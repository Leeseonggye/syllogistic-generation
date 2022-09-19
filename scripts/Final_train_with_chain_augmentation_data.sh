#!/bin/bash

for aug_method in ori_mix aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do
        for filtering in true false
        do
            for best_epoch_method in rouge12L rougeL
            do
            python3 ../main.py -group_name A_final_test_30epochs_${best_epoch_method}_${aug_method}_num_augmentation_${num_augmentation}_filtered_${filtering} \
                -save_final_model true \
                -num_kfold 0 \
                -num_augmentation $num_augmentation \
                -bertscore_ceil 0.99 \
                -bertscore_floor 0.95 \
                -filtering $filtering \
                -augmentation_method $aug_method \
                -best_epoch_method $best_epoch_method \
                # -save_epochs False \
                # -save_final_model False
            done
        done    
    done
done

