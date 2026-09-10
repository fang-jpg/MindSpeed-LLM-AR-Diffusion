# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
  --input ./dataset/AlpacaHistroy-oaast-selected-0000.parquet \
  --tokenizer-name-or-path ./model_from_hf/qwen2_5_7b_hf/ \
  --output-prefix ./finetune_dataset/alpaca \
  --handler-name AlpacaStyleInstructionHandler \
  --tokenizer-type PretrainedFromHF \
  --workers 4 \
  --log-interval 1000 \
  --prompt-type qwen \
  --map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'
# 多轮对话数据转换，--map-keys按实际数据集修改