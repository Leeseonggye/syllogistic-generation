for kfold in {0..4}
do
python3 ../main.py \
    -project_name syllogistic-generation \
    -group_name bart_vanilla_kfold \
    -num_epochs 30 \
    -valid_batch_size 32 \
    -train_mode kfold \
    -kfold_idx $kfold \
    -num_kfold 5 \
    -save_final_model false
done
