#!/bin/bash

python3 ../main.py \
    -project_name syllogistic-generation \
    -group_name A_final_rougeL_ori_mix_1_filtering_true_30epoch \
    -num_kfold 0 \
    -num_augmentation 1 \
    -filtering true \
    -augmentation_method ori_mix \
    -best_epoch_method rougeL \
    -bertscore_ceil 0.99 \
    -bertscore_floor 0.95 \
    -save_final_model true

for aug_method in aug_mix
do
    for num_augmentation in 1
    do
        for filtering in true
        do
            for best_epoch_method in rouge12L rougeL
            do
            python3 ../main.py -group_name A_final_${best_epoch_method}_${aug_method}_${num_augmentation}_filtering_${filtering}_30epoch \
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

for aug_method in aug_pair
do
    for num_augmentation in 1
    do
        for filtering in false
        do
            for best_epoch_method in rouge12L rougeL
            do
            python3 ../main.py -group_name A_final_${best_epoch_method}_${aug_method}_${num_augmentation}_filtering_${filtering}_30epoch \
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

for aug_method in aug_pair
do
    for num_augmentation in {2..3}
    do
        for filtering in true
        do
            for best_epoch_method in rouge12L rougeL
            do
            python3 ../main.py -group_name A_final_${best_epoch_method}_${aug_method}_${num_augmentation}_filtering_${filtering}_30epoch \
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

