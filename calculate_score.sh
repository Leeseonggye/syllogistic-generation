#!/bin/bash

for aug_method in ori_mix aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do
        for filtering in true false
        do
            for best_epoch_method in rouge12L rougeL
            do         
                python3 calculate_scores.py -num_augmentation $num_augmentation \
                    -method $aug_method \
                    -best_epoch_method $best_epoch_method \
                    -filtering $filtering
            done
        done
    done
done

