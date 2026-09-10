# Demo提供的是Alpaca单轮对话数据集处理脚本，多轮对话场景下需要更换数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

# Alpaca单轮对话数据集示例
python ./preprocess_data.py \
	--input ./dataset/instruct.json \
	--tokenizer-name-or-path ./model_from_hf/llama31_hf \
	--output-prefix ./finetune_dataset/alpaca \
	--handler-name AlpacaStyleInstructionHandler \
	--tokenizer-type PretrainedFromHF \
	--workers 4 \
	--seq-length 4096 \
	--log-interval 1000 \
	--prompt-type llama3 \
	--pack \
	--neat-pack

# 若使用Alpaca多轮对话数据集需要增加以下参数
# -map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'
# 多轮对话建议使用Sharegpt格式数据集

# Sharegpt多轮对话数据集示例
#python ./preprocess_data.py \
#	--input ./dataset/hermes_de_sharegpt.json \
#	--tokenizer-name-or-path ./model_from_hf/llama31_hf \
#	--output-prefix ./finetune_dataset/sharegpt \
#	--handler-name SharegptStyleInstructionHandler \
#	--tokenizer-type PretrainedFromHF \
#	--workers 4 \
#	--seq-length 4096 \
#	--log-interval 1000 \
#	--prompt-type llama3 \
#	--pack \
#	--neat-pack \
#	--map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}'
