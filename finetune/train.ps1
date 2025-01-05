# 设置变量
$DATA_PATH="E:\xz\pywork\2024PhageLS\DNABERT2\finetune\data\576"
$MAX_LENGTH=200
$LR=3e-5

# 训练命令
python train.py `
    --model_name_or_path "E:\xz\pywork\2024PhageLS\DNABERT2\DNABERT-2-117M" `
    --data_path $DATA_PATH `
    --kmer -1 `
    --run_name "DNABERT2_$($DATA_PATH)" `
    --model_max_length $MAX_LENGTH `
    --per_device_train_batch_size 4 `
    --per_device_eval_batch_size 4 `
    --gradient_accumulation_steps 1 `
    --learning_rate $LR `
    --num_train_epochs 5 `
    --save_steps 200 `
    --output_dir "E:\xz\pywork\2024PhageLS\DNABERT2\finetune\output" `
    --evaluation_strategy steps `
    --eval_steps 200 `
    --warmup_steps 50 `
    --logging_steps 100 `
    --overwrite_output_dir `
    --log_level info `
    --find_unused_parameters False
