#!/bin/bash

python3 ../main.py \
    -project_name syllogistic-generation \
    -group_name bart_vanilla_kfold \
    -valid_batch_size 32 \
    -train_mode kfold \
    -num_kfold 0 \
    -save_final_model true

