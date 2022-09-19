#!/bin/bash
for fold_idx in {0..4}
do
python3 ../main.py -group_name KFOLD_VALIDATION_aug_mix_num_augmentation_1 \
    -num_augmentation 1 \
    -bertscore_ceil 0.99 \
    -bertscore_floor 0.95 \
    -kfold_idx $fold_idx \
    -augmentation_method aug_mix \
    -num_epochs 10
done

for fold_idx in {0..4}
do
python3 ../main.py -group_name KFOLD_VALIDATION_aug_pair_num_augmentation_3 \
    -num_augmentation 3 \
    -bertscore_ceil 0.99 \
    -bertscore_floor 0.95 \
    -kfold_idx $fold_idx \
    -augmentation_method aug_pair \
    -num_epochs 10
done
