for fold in {0..4}
do
python3 ../main.py \
    -learning_rate 3e-4 \
    -load_model_path gpt2 \
    -load_tokenizer_path gpt2 \
    -gpu_id 0 \
    -project_name syllogistic-generation \
    -group_name gpt2-LM \
    -num_epochs 15 \
    -augment_mode false \
    -batch_size 8 \
    --accumulation_steps 2 \
    -valid_batch_size 32 \
    -train_mode kfold \
    -kfold_idx $fold \
    -num_kfold 5 \
    -max_len 256 \
    -save_final_model false
done