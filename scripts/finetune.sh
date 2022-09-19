kfold_idx=-1
num_kfold=0 
export bert_ceil=0.99
export bert_floor=0.95
export rouge_ceil=0.9

for seed in {42..50}
python3 ../main.py \
    -seed $seed \
    -load_model_path t5-base \
    -load_tokenizer_path t5-base \
    -gpu_id 0 \
    -project_name syllogistic-generation \
    -group_name t5-$seed\
    -num_epochs 30 \
    -augment_mode false \
    -batch_size 16 \
    -valid_batch_size 32 \
    -train_mode finetuning \
    -kfold_idx $kfold_idx \
    -num_kfold $num_kfold \
    -learning_rate 3e-4 \
    -save_final_model false