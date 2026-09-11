#!/usr/bin/env python3
"""当前 Qwen3 block_diff 的离线泄漏检查；中文操作说明见同目录 attention_leakage_说明.md。

方法一：固定 noisy 输入，逐个替换 clean block，比较每个 noisy block 的全部 logits。
方法二：截获每层真实 attention kernel 入口，检查完整 Q×K 可见关系。
方法三：对 noisy 输出求输入位置 embedding 梯度，检查直接及跨层间接依赖。
方法四：替换当前 clean block 的整段秘密 token，展示预测并比较全部 logits。

只运行 forward/autograd.grad，不创建优化器、不写模型权重、不启动双机训练。
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path", type=Path, help="本地已合并 HF checkpoint，必须含 config.json 和完整权重")
    source.add_argument("--tiny", action="store_true", help="随机小模型：仅检查代码路径，不代表训练 checkpoint")
    p.add_argument("--diffusion-config", type=Path, help="显式补充训练时使用的 diffusion JSON；只改内存配置")
    p.add_argument("--device", default="npu:0", help="例如 npu:0 / cuda:0 / cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--backend", choices=["auto", "eager", "sdpa"], default="auto", help="auto 沿用加载时的选择；报告实际值")
    p.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    p.add_argument("--block-size", type=int, help="默认沿用 checkpoint；覆盖时报告，不改 checkpoint")
    p.add_argument("--num-blocks", type=int, default=3, help="至少三个 block，默认覆盖首/中/末块")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--noise-ratio", type=float, default=1.0, help="默认全 MASK；也可用 0.5 检查部分加噪")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--atol", type=float, default=1e-6, help="logits 绝对误差阈值；不会自动放宽")
    p.add_argument("--grad-atol", type=float, default=1e-8)
    p.add_argument("--gradient-probes", type=int, default=2, help="每个 noisy block 的独立随机输出投影数")
    p.add_argument("--use-fused-rmsnorm", action="store_true")
    p.add_argument("--use-fused-rotary-pos-emb", action="store_true")
    p.add_argument("--inject-leak", choices=["same", "future"], help="仅用于验证检测器：在内存中故意放开禁止边，正常应 FAIL")
    p.add_argument("--report", type=Path, default=Path("attention_leakage_report.json"))
    a = p.parse_args()
    if (a.num_blocks < 3 or a.batch_size < 1 or a.gradient_probes < 1
            or not 0 < a.noise_ratio <= 1 or a.atol < 0 or a.grad_atol < 0
            or not math.isfinite(a.atol) or not math.isfinite(a.grad_atol)
            or (a.block_size is not None and a.block_size < 1)):
        p.error("block/batch/probe 必须为正，num-blocks≥3，noise-ratio∈(0,1]，阈值非负")
    return a


class Reporter:
    def __init__(self, args):
        self.args = args
        self.data = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "metadata": {}, "checks": []}

    def add(self, name, status, **details):
        self.data["checks"].append(dict(name=name, status=status, **details))
        print(f"[{status}] {name}: {json.dumps(details, ensure_ascii=False)}", flush=True)

    def save(self):
        states = {x["status"] for x in self.data["checks"]}
        status = "ERROR" if "ERROR" in states else "FAIL" if "FAIL" in states else "INCONCLUSIVE" if "INCONCLUSIVE" in states else "PASS"
        self.data["status"] = status
        self.args.report.parent.mkdir(parents=True, exist_ok=True)
        self.args.report.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"\n总结果：{status}；完整报告：{self.args.report.resolve()}", flush=True)
        return {"PASS": 0, "FAIL": 1, "ERROR": 2, "INCONCLUSIVE": 3}[status]


def expected_allowed(length, block_size, torch, device):
    """独立 oracle：不调用被测 mask 函数，不复用其向量表达式。
    布局 [noisy, clean]；clean 因果且不看 noisy；noisy 只看同块 noisy 和更早 clean。
    """
    rows = []
    for q in range(2 * length):
        row = []
        for k in range(2 * length):
            if q >= length:
                visible = k >= length and k <= q
            elif k < length:
                visible = q // block_size == k // block_size
            else:
                visible = (k - length) // block_size < q // block_size
            row.append(visible)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device)[None, None]


def run(args, report):
    import torch
    import transformers
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401；注册 NPU 设备及算子，使用训练服务器现有环境
        torch.npu.set_device(args.device)
    from transformers import Qwen3Config
    from mindspeed_llm.fsdp2.models.qwen3_diffusion.modeling_qwen3_diffusion import Qwen3DiffusionForCausalLM as Model

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    Model.register_patches(SimpleNamespace(use_fused_rmsnorm=args.use_fused_rmsnorm,
                                          use_fused_rotary_pos_emb=args.use_fused_rotary_pos_emb))
    # 与 ModelFactory 一样先注册补丁，再导入 attention（其 RoPE 函数是模块级导入）。
    from mindspeed_llm.fsdp2.models.qwen3 import qwen3_attention as attention_module
    if args.tiny:
        config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False)
        config.mask_token_id = 127
        config.block_size = 4
        config.dlm_paradigm = "block_diff"
    else:
        if not args.model_path.is_dir():
            raise ValueError("--model-path 必须是本地 HF checkpoint 文件夹")
        config = Qwen3Config.from_pretrained(str(args.model_path), local_files_only=True)
    if args.diffusion_config:
        extra = json.loads(args.diffusion_config.read_text(encoding="utf-8"))
        config.update(extra)
    if args.block_size is not None:
        config.block_size = args.block_size
    if getattr(config, "dlm_paradigm", None) != "block_diff":
        raise ValueError("checkpoint 没有 dlm_paradigm=block_diff；请显式提供训练时 diffusion-config")
    block = getattr(config, "block_size", 0)
    if not isinstance(block, int) or block < 1:
        raise ValueError("配置缺少有效 block_size")
    if not 0 <= getattr(config, "mask_token_id", -1) < config.vocab_size:
        raise ValueError("mask_token_id 不在已保存词表内；请检查 checkpoint，测试不自动扩词表")
    if args.backend != "auto":
        config._attn_implementation = args.backend
    if args.tiny:
        model = Model(config).to(dtype=dtype)
    else:
        # 当前子类 from_pretrained 会将 embedding 复制到 lm_head，覆盖训练后的输出头。
        # 直接调用父类加载器，仍实例化当前 Model；严格检查缺失/多余/形状不符的权重。
        model, info = super(Model, Model).from_pretrained(
            str(args.model_path), config=config, local_files_only=True,
            torch_dtype=dtype, output_loading_info=True)
        problems = {k: info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(k)}
        if problems:
            raise RuntimeError(f"checkpoint 非完整匹配，拒绝用随机补齐权重做通过判定：{problems}")
    model.to(args.device)
    model.requires_grad_(False)  # 只求输入 embedding 的梯度，节省显存，不更新参数
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    # 保留 train 分支，但关闭随机 dropout；不能用 eval 代替真实 train 路径。
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    backend = model.model.layers[0].self_attn.config._attn_implementation
    report.data["metadata"].update(torch=torch.__version__, transformers=transformers.__version__,
        model_class=type(model).__name__, backend=backend, block_size=block,
        source_root=str(ROOT), weight_source="随机小模型" if args.tiny else str(args.model_path.resolve()),
        scope="单设备、完整层数、无 FSDP/TP/CP 包装；train 保留原分支但关闭 dropout/recompute")
    for relative in ("mindspeed_llm/fsdp2/models/qwen3_diffusion/modeling_qwen3_diffusion.py",
                     "mindspeed_llm/fsdp2/models/qwen3/qwen3_attention.py"):
        report.data["metadata"][relative + ":sha256"] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    print(json.dumps(report.data["metadata"], ensure_ascii=False, indent=2), flush=True)
    if backend == "eager":
        real_kernel = attention_module.eager_attention_forward
    else:
        real_kernel = attention_module.ALL_ATTENTION_FUNCTIONS[backend]

    length = args.num_blocks * block
    expected = expected_allowed(length, block, torch, args.device)
    modes = ["train", "eval"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for seed in args.seeds:
            tag = f"{mode}/seed={seed}"
            print(f"\n===== {tag}；noisy=[0,{length})，clean=[{length},{2*length}) =====", flush=True)
            try:
                run_case(args, report, model, attention_module, real_kernel, backend,
                         torch, expected, mode, seed, length, block, tag)
            except Exception as exc:
                report.add(tag, "ERROR", error=str(exc), traceback=traceback.format_exc())


def run_case(args, report, model, attention_module, real_kernel, backend,
             torch, expected, mode, seed, length, block, tag):
    torch.manual_seed(seed)
    model.train(mode == "train")
    model.bd_mask = None
    model.block_diff_position_ids = None
    # 先在 CPU 生成，再搬到设备，避免设备 RNG 差异影响实验输入。
    clean = torch.randint(0, model.config.vocab_size - 1, (args.batch_size, length))
    clean[clean == model.mask_token_id] = (model.mask_token_id + 1) % model.config.vocab_size
    clean = clean.to(args.device)
    selected = torch.rand(args.batch_size, length).lt(args.noise_ratio).to(args.device)
    selected[:, ::block] = True  # 每个 block 至少有一个 MASK，正对照始终包含预测位置
    noisy = torch.where(selected, model.mask_token_id, clean).detach()
    probability = torch.full_like(clean, args.noise_ratio, dtype=torch.float32)
    half_pos = torch.arange(length, device=args.device)[None].expand(args.batch_size, -1)
    # mg 数据侧传入 L 个位置；train attention 将 Q/K 分半后分别应用同一份 RoPE。
    # eval attention 不分半，所以显式传 [0..L-1,0..L-1]，保持逻辑位置一致。
    positions = half_pos if mode == "train" else torch.cat([half_pos, half_pos], dim=1)
    report.data.setdefault("cases", {})[tag] = {"clean_ids": clean.cpu().tolist(), "noisy_ids": noisy.cpu().tolist(),
        "masked_indices": selected.cpu().tolist(), "position_ids": positions.cpu().tolist()}

    def frozen_noise(input_ids, eps=1e-3, loss_mask=None):
        # 关键：返回固定 noisy tensor，不能只固定随机种子再从变动后的 clean 重建 noisy。
        return noisy.clone(), selected.clone(), probability.clone()

    audit = {"active": False, "layers": set()}

    def kernel(module, query, key, value, attention_mask=None, **kwargs):
        if audit["active"]:
            layer = module.layer_idx
            audit["layers"].add(layer)
            ok = False
            details = {"layer": layer, "q_shape": list(query.shape), "k_shape": list(key.shape),
                       "mask_shape": None if attention_mask is None else list(attention_mask.shape),
                       "is_causal": kwargs.get("is_causal"), "dropout": kwargs.get("dropout")}
            if attention_mask is not None and attention_mask.ndim == 4 and attention_mask.shape[-2:] == (2*length, 2*length):
                # 本仓库应传 additive float mask：0 可见，-inf 或 finfo.min 屏蔽。
                # 不把 bool mask 或任意负小数当成正确 mask。
                if attention_mask.dtype.is_floating_point:
                    allowed = attention_mask == 0
                    blocked = torch.isneginf(attention_mask) | (attention_mask <= torch.finfo(attention_mask.dtype).min / 2)
                    target = expected.expand_as(attention_mask)
                    valid = torch.where(target, allowed, blocked)
                    ok = bool(valid.all()) and kwargs.get("is_causal") is False and kwargs.get("dropout", 0) == 0
                    details["wrong_entries"] = int((~valid).sum())
                    details["first_wrong_qk"] = (~valid).nonzero()[:10].cpu().tolist()
                    if layer == 0:
                        details["noisy_block1_query"] = block
                        details["visible_key_positions"] = allowed[0, 0, block].nonzero().flatten().cpu().tolist()
            report.add(f"{tag}/方法二/实际 kernel mask/层{layer}", "PASS" if ok else "FAIL", **details)
        return real_kernel(module, query, key, value, attention_mask=attention_mask, **kwargs)

    def forward(x):
        # 保留 labels 才能进入真实 block_diff 拼接与 mask 逻辑；labels 已在数据侧 shift。
        labels = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        result = model(input_ids=x, labels=labels, position_ids=positions,
                       use_cache=False, return_dict=True)
        logits = result.logits
        if logits.shape[:2] != x.shape or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("noisy logits 形状错误或出现 NaN/Inf")
        return logits

    def compare(name, a, b, should_change):
        delta = (a.float() - b.float()).abs()
        maximum = delta.max().item()
        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError(f"{name} 出现非有限差异")
        status = ("PASS" if maximum > args.atol else "INCONCLUSIVE") if should_change else ("PASS" if maximum <= args.atol else "FAIL")
        report.add(f"{tag}/{name}", status, max_abs=maximum, mean_abs=delta.mean().item(),
                   l2=delta.norm().item(), threshold=args.atol, expected="应变化（正对照）" if should_change else "应不变")

    # 只包裹当前 attention 模块查到的 kernel，保留原始算子及其参数。
    with patch.object(model, "forward_process", side_effect=frozen_noise), \
         patch.object(attention_module, "eager_attention_forward", kernel), \
         patch.object(attention_module, "ALL_ATTENTION_FUNCTIONS", {backend: kernel}):
        if args.inject_leak:
            # 检测器负对照：仅当前进程的缓存 mask 被修改，不写入源代码或 checkpoint。
            model.bd_mask = model.make_block_diff_mask(length, block, args.batch_size,
                next(model.parameters()).dtype, args.device).clone()
            key_block = 1 if args.inject_leak == "same" else 2
            model.bd_mask[:, :, block:2*block, length+key_block*block:length+(key_block+1)*block] = 0

        print("方法一/步骤1：重复 baseline，检查固定输入是否可复现；同时执行方法二。", flush=True)
        with torch.no_grad():
            audit["active"] = True
            base = forward(clean).detach()
            audit["active"] = False
            required_layers = set(range(len(model.model.layers)))
            report.add(f"{tag}/方法二/所有层 kernel 均被观察", "PASS" if audit["layers"] == required_layers else "FAIL",
                       layers=sorted(audit["layers"]), expected_layers=len(required_layers))
            repeat = forward(clean).detach()
            jitter = (repeat.float() - base.float()).abs().max().item()
            report.add(f"{tag}/方法一/重复性", "PASS" if jitter <= args.atol else "INCONCLUSIVE", max_abs=jitter, threshold=args.atol)
            if jitter > args.atol:
                print("baseline 噪声超过阈值：本轮无法排除泄漏，停止差异判定。", flush=True)
                return
            print("方法一/步骤2：逐个替换 clean block，对所有 noisy block 比较（包含首尾边界）。", flush=True)
            for kb in range(args.num_blocks):
                altered = clean.clone()
                altered[:, kb*block:(kb+1)*block] = (altered[:, kb*block:(kb+1)*block] + 1) % model.config.vocab_size
                changed = forward(altered).detach()
                for qb in range(args.num_blocks):
                    sl = slice(qb*block, (qb+1)*block)
                    relation = "前缀" if kb < qb else "当前" if kb == qb else "未来"
                    compare(f"方法一/clean块{kb}→noisy块{qb}/{relation}", base[:, sl], changed[:, sl], kb < qb)

            print("方法四：把当前 clean Block1 整段秘密 token 替换，noisy 和其他 clean 块固定。", flush=True)
            secret = clean.clone()
            secret[:, block:2*block] = (secret[:, block:2*block] + 17) % model.config.vocab_size
            secret_logits = forward(secret).detach()
            report.data["cases"][tag]["secret_demo"] = {
                "说明": "词表 ID 序列作为秘密；无需 tokenizer，不将 argmax 不变单独当作通过证据",
                "before_secret": clean[:, block:2*block].cpu().tolist(),
                "after_secret": secret[:, block:2*block].cpu().tolist(),
                "before_prediction": base[:, block:2*block].argmax(-1).cpu().tolist(),
                "after_prediction": secret_logits[:, block:2*block].argmax(-1).cpu().tolist()}
            print(json.dumps(report.data["cases"][tag]["secret_demo"], ensure_ascii=False), flush=True)
            compare("方法四/秘密替换", base[:, block:2*block], secret_logits[:, block:2*block], False)
            del repeat, changed, secret_logits, base

        print("方法三：对每个 noisy block 的输出做随机投影，求各输入位置 embedding 梯度。", flush=True)
        captured = {}

        def embedding_hook(module, inputs, output):
            # 这是拼接后 [B,2L,H] 的位置张量；不能对共享 embedding.weight 求梯度来区分位置。
            leaf = output.detach().requires_grad_(True)
            captured["embedding"] = leaf
            return leaf

        handle = model.get_input_embeddings().register_forward_hook(embedding_hook)
        try:
            for qb in range(args.num_blocks):
                for probe in range(args.gradient_probes):
                    # 每次独立图，避免 retain_graph 长时间占用显存；不对 loss 反传，排除标签直连。
                    logits = forward(clean)
                    part = logits[:, qb*block:(qb+1)*block].float()
                    direction = torch.randn(part.shape, device="cpu").to(args.device)
                    score = (part * direction).sum() / part.numel() ** 0.5
                    gradient, = torch.autograd.grad(score, captured["embedding"])
                    if not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError("embedding 梯度出现 NaN/Inf")
                    norms = gradient.float().norm(dim=-1)
                    # 允许：自身 noisy block、过去 clean block；其余 clean/noisy 一律禁止。
                    allowed = expected[0, 0, qb*block]
                    forbidden_max = norms[:, ~allowed].max().item()
                    report.add(f"{tag}/方法三/noisy块{qb}/投影{probe}/禁止位置", "PASS" if forbidden_max <= args.grad_atol else "FAIL",
                               max_norm=forbidden_max, threshold=args.grad_atol,
                               per_position_norm=norms.cpu().tolist())
                    if qb > 0:
                        prefix_max = norms[:, length:length+qb*block].max().item()
                        report.add(f"{tag}/方法三/noisy块{qb}/投影{probe}/前缀正对照", "PASS" if prefix_max > args.grad_atol else "INCONCLUSIVE",
                                   max_norm=prefix_max, threshold=args.grad_atol)
                    del logits, part, score, gradient, norms
        finally:
            handle.remove()


def main():
    args = arguments()
    report = Reporter(args)
    try:
        run(args, report)
    except Exception as exc:
        report.add("初始化或运行异常", "ERROR", error=str(exc), traceback=traceback.format_exc())
    return report.save()


if __name__ == "__main__":
    sys.exit(main())
