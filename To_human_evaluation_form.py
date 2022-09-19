import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default ="/root/syllogistic/syllogistic-generation/seq2seq/finetune/generation_log/")
    parser.add_argument("-num_augmentation", type=int)
    parser.add_argument("-method", type=str)
    parser.add_argument('-filtering', type = lambda x : x.lower() == 'true', default = "False", help = "augmented data filtering 유무")
    parser.add_argument('-drop_duplication', type = lambda x : x.lower() == 'true', default = "False", help = "augmented data duplication 제거 유무")

    # parser.add_argument("-model_name", type=str)
    parser.add_argument("-log_path", type=str, default = "/root/syllogistic/syllogistic-generation/seq2seq/finetune/logs/final_log/log_vanilla_BART.csv")
    args = parser.parse_args()

    if os.path.isdir("/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form"):
        pass
    else:
        os.mkdir("/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form")

    data = pd.read_html("/root/syllogistic/syllogistic-generation/seq2seq/finetune/generation_log/bart_vanilla_kfold/bart_vanilla_kfold.html")[0]
    diff_list = []
    for idx in range(len(data)):
        if data['generation'][idx] != data['label'][idx]:
            diff_list.append(idx)
        else:
            pass
    
    human_evaluate_list = data.iloc[diff_list]
    # human_evaluate_list.to_csv(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/{args.method}_{args.num_augmentation}_filtering_{args.filtering}_drop_duplicate{args.drop_duplication}.csv")
    # human_evaluate_list.to_html(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/{args.method}_{args.num_augmentation}_filtering_{args.filtering}_drop_duplicate{args.drop_duplication}.html")
    human_evaluate_list.to_csv(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/bart_vanilla.csv")
    human_evaluate_list.to_html(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/bart_vanilla.html")
    human_evaluate_list = pd.read_csv(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/bart_vanilla.csv", index_col= 0)
    human_evaluate_list.to_csv(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/bart_vanilla.csv", index=False)
    human_evaluate_list.to_html(f"/root/syllogistic/syllogistic-generation/seq2seq/finetune/human_evaluation_form/bart_vanilla.html", index=False)
    

    
    

if __name__ == "__main__":
    main()