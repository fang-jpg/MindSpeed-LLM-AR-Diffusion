# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
	--input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
	--tokenizer-name-or-path ./model_from_hf/qwen25_hf/ \
	--output-prefix ./finetune_dataset/alpaca \
	--handler-name AlpacaStyleInstructionHandler \
	--tokenizer-type PretrainedFromHF \
	--workers 4 \
	--log-interval 1000 \
	--prompt-type qwen \
	--pack \
	--neat-pack \
	--seq-length 4096 \
#  demo提供的是单轮数据集，若使用多轮数据需要修改以下参数：
#  --input ./dataset/多轮数据集
#  --map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'