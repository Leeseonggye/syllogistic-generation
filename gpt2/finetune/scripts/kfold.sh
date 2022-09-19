for kfold_idx in {0..4}
do
python3 ../main.py \
    -seed 48 \
    -load_model_path t5-base \
    -load_tokenizer_path t5-base \
    -gpu_id 0 \
    -project_name syllogistic-generation \
    -group_name t5-bsz8\
    -num_epochs 10 \
    -augment_mode false \
    -batch_size 8 \
    -valid_batch_size 32 \
    -train_mode kfold \
    -kfold_idx $kfold_idx \
    -num_kfold 5 \
    -learning_rate 3e-4 \
    -save_final_model false
done

# for kfold_idx in {0..4}
# do
# python3 ../main.py \
#     -load_model_path facebook/bart-base \
#     -load_tokenizer_path facebook/bart-base \
#     -gpu_id 1 \
#     -project_name syllogistic-generation \
#     -group_name bart-base \
#     -num_epochs 30 \
#     -augment_mode false \
#     -batch_size 16 \
#     -valid_batch_size 32 \
#     -train_mode kfold \
#     -kfold_idx $kfold_idx \
#     -num_kfold 5 \
#     -save_final_model false
# done