for kfold in {0..4}
do
python3 ../main.py \
    -gpu_id 0 \
    -project_name syllogistic-generation \
    -group_name bart_vanilla_kfold \
    -num_epochs 30 \
    -augment_mode false \
    -batch_size 16 \
    -valid_batch_size 32 \
    -train_mode kfold \
    -kfold_idx $kfold \
    -num_kfold 5 \
    -save_final_model false
done