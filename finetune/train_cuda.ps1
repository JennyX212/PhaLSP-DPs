# 设置变量
$LR=3e-5
$MAX_LENGTH=200
$SEQ_LENGTH=600
$DATA_PATH="G:\Git\Github\DNABERT2\finetune\data\$($SEQ_LENGTH)"

# 训练命令
python train.py `
    --model_name_or_path "G:\Git\Github\DNABERT2\DNABERT-2-117M" `
    --data_path $DATA_PATH `
    --kmer -1 `
    --run_name "DNABERT2_$($DATA_PATH)" `
    --model_max_length $MAX_LENGTH `
    --per_device_train_batch_size 8 `
    --per_device_eval_batch_size 8 `
    --gradient_accumulation_steps 1 `
    --learning_rate $LR `
    --num_train_epochs 20 `
    --fp16 `
    --save_steps 210 `
    --output_dir "G:\Git\Github\DNABERT2\finetune\output\$($SEQ_LENGTH)" `
    --evaluation_strategy steps `
    --eval_steps 210 `
    --warmup_steps 50 `
    --logging_steps 100 `
    --overwrite_output_dir `
    --log_level info `
    --find_unused_parameters False
