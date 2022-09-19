# for premise in premise1 premise2
# do 
#     for fold_idx in {0..4}
#     do
#     python3 main.py \
#         -project_name Backtranslation-finetuning \
#         -group_name $premise-use_all \
#         -augmented_data $premise \
#         -num_epochs 20 \
#         -batch_size 8 \
#         -valid_batch_size 32 \
#         -train_mode kfold \
#         -kfold_idx $fold_idx \
#         -bertscore_ceil 1 \
#         -bertscore_floor 0 \
#         -rouge_ceil 1 
#     done
# done

for fold_idx in {0..4}
    do
    python3 main.py \
        -project_name Backtranslation-finetuning \
        -group_name premise-not-use \
        -augmented_data premise1 \
        -num_epochs 20 \
        -batch_size 8 \
        -valid_batch_size 32 \
        -train_mode kfold \
        -kfold_idx $fold_idx \
        -bertscore_ceil 0 \
        -bertscore_floor 1 \
        -rouge_ceil 0
    done