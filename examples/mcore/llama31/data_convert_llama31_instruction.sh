# Demo提供的是Alpaca单轮对话数据集，多轮对话场景下需要更换数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
	--input ./dataset/train-00000-of-00001-d9b93805488c263e.parquet\
	--tokenizer-name-or-path ./model_from_hf/llama31_hf \
	--output-prefix ./finetune_dataset/alpaca \
	--handler-name AlpacaStyleInstructionHandler \
	--tokenizer-type PretrainedFromHF \
	--workers 4 \
	--log-interval 1000 \
	--prompt-type llama3 \
# 若使用Alpaca多轮对话数据集需要增加以下参数
# -map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'
