# model_name=t5-small
# model_name=facebook/bart-base
model_name=gpt2
# GPT2 일때만 max len 256
model_name=gpt2
python3 ../main.py \
    -load_model_path $model_name \
    -load_tokenizer_path $model_name \
    -gpu_id 1 \
    -project_name syllogistic-generation \
    -group_name debug-save_generation \
    -num_epochs 3 \
    -augment_mode false \
    -batch_size 16 \
    -valid_batch_size 32 \
    -train_mode kfold \
    -kfold_idx 0 \
    -num_kfold 5 \
    -max_len 256 \
    -save_final_model false
