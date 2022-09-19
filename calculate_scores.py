import datasets
import pandas as pd
import numpy as np
import argparse
import pprint
import os

# python3 calculate_scores.py -num_augmentation $num_augmentation \
#         -method $aug_method \
# /root/syllogistic/syllogistic-generation/seq2seq/finetune/generation_log/docker_test_train_aug_mix_num_augmentation_1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default ="/root/syllogistic/syllogistic-generation/seq2seq/finetune/generation_log/")
    parser.add_argument("-num_augmentation", type=int)
    parser.add_argument("-method", type=str)
    parser.add_argument('-filtering',type = str,default = "kfold",help = "best epoch 선택 방법을 blue+rougeL or bleu+rouge1,2,L",choices = ['true', 'false'])
    parser.add_argument("-log_path", type=str, default = "/root/syllogistic/syllogistic-generation/seq2seq/finetune/logs/final_log/log.csv")
    parser.add_argument('-best_epoch_method',type = str,default = "kfold",help = "best epoch 선택 방법을 blue+rougeL or bleu+rouge1,2,L",choices = ['rougeL', 'rouge12L'])

    args = parser.parse_args()

    if os.path.isdir("/root/syllogistic/syllogistic-generation/seq2seq/finetune/logs/final_log/"):
        pass
    else:
        os.mkdir("/root/syllogistic/syllogistic-generation/seq2seq/finetune/logs/final_log/")

    model_name = f"A_final_{args.best_epoch_method}_{args.method}_{args.num_augmentation}_filtering_{args.filtering}"
    data_name = f"A_final_{args.best_epoch_method}_{args.method}_{args.num_augmentation}_filtering_{args.filtering}_30epoch/A_final_{args.best_epoch_method}_{args.method}_{args.num_augmentation}_filtering_{args.filtering}_30epoch.html"
    model_name = args.data_name
    data = pd.read_html(os.path.join(args.data_path,data_name))[0]
    label = data['label'].tolist()
    generation = data['generation'].tolist()
        
    bert_scorer = datasets.load_metric('bertscore')
    rouge_scorer = datasets.load_metric('rouge')
    bleu_scorer = datasets.load_metric('bleu')
        
    bert_score = bert_scorer.compute(
            references = label, 
            predictions = generation, 
            lang = 'en', 
            verbose = True
            )
    bert_score = np.mean(bert_score["f1"])

    rouge_score = rouge_scorer.compute(
            references = label, 
            predictions = generation, 
            )
        
    bleu_prediciton = [pred.split() for pred in generation]
    bleu_label = [[lab.split()] for lab in label]
    bleu_score = bleu_scorer.compute(
            references = bleu_label, 
            predictions = bleu_prediciton, 
            )
        
    score_dict = {
            "model_name" : model_name,
            'bert_score': bert_score,
            'rouge-1': rouge_score["rouge1"].mid.fmeasure,
            'rouge-2': rouge_score["rouge2"].mid.fmeasure,
            'rouge-L': rouge_score["rougeL"].mid.fmeasure,
            'bleu': bleu_score["bleu"]
        }
        
    pprint.pprint(score_dict)

    result_pd = pd.DataFrame(score_dict, index = [0])
    log_df = load_log_df(args)
    log_df = log_df.append(result_pd, ignore_index=True)
    log_df.to_csv(os.path.join(args.log_path), index = False)
 
def load_log_df(args):
    
    if not os.path.exists(os.path.join(args.log_path)):
        log_df = pd.DataFrame(columns = ["model_name", "bert_score", "rouge-1", "rouge-2", "rouge-L", "bleu"])
        log_df.to_csv(os.path.join(args.log_path), index = False)
    else :
        log_df = pd.read_csv(args.log_path)
    return log_df

if __name__ == "__main__":
    main()
