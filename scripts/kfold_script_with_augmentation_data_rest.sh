#!/bin/bash

python3 ../main.py -group_name KFOLD_VALIDATION_ori_mix_num_augmentation_3_non_filtering \
            -num_augmentation 3 \
            -bertscore_ceil 0.99 \
            -bertscore_floor 0.95 \
            -kfold_idx 4 \
            -num_epochs 30 \
            # -save_epochs False \
            # -save_final_model False


for aug_method in aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do
        for fold_idx in {0..4}
        do
        python3 ../main.py -group_name KFOLD_VALIDATION_${aug_method}_num_augmentation_${num_augmentation}_non_filtering \
            -num_augmentation $num_augmentation \
            -bertscore_ceil 0.99 \
            -bertscore_floor 0.95 \
            -kfold_idx $fold_idx \
            -augmentation_method $aug_method \
            -num_epochs 30 \
            # -save_epochs False \
            # -save_final_model False
        done
    done
done

