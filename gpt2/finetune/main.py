from random import sample
from arg_parser import make_parser
from train_utils import train
from utils import set_seed

import torch
from torch.utils.data import DataLoader


import os
import pandas as pd
import wandb

def main(args = None):
    if args is None:
        args = make_parser()
    print("current working directory : ", os.getcwd())
    print("gpu device for usage : ", torch.cuda.get_device_name(args.gpu_id))
    
    print("="*20, "Arguments", "="*20)
    print('\n'.join(f'{k:20} : {v}' for k, v in vars(args).items()))
    print("="*45)

    fold_name = f"{args.group_name}-kfold({args.kfold_idx})" if args.num_kfold != 0 else f"{args.group_name}-finetune"
    args.fold_name = fold_name
    wandb.init(
        project = args.project_name,
        name = args.fold_name, 
        reinit = True,
        group = args.group_name)
    

    # Gradient Accumulation
    if args.accumulation_steps > 1 :
        print("Gradient Accumulation is enabled")
        print("Accumulation steps : ", args.accumulation_steps)
        print("Batch size : ", args.batch_size)
        print("Accumulated Batch size : ", args.batch_size * args.accumulation_steps)
        
    set_seed(args.seed)      

    # 저장여부 확인 및 경로 설정
    if args.save_final_model:
        if not os.path.exists(os.path.join(args.generation_path, args.fold_name)) :
            os.mkdir(os.path.join(args.generation_path, args.fold_name))
            os.mkdir(os.path.join(args.model_path, args.fold_name))
    else: 
        print("="*30)
        print("not save at all")
        print("="*30)

    if not os.path.exists(os.path.join(args.log_dir, args.fold_name)) :
        os.mkdir(os.path.join(args.log_dir, args.fold_name))

    model, tokenizer, TrainDataset, TestDataset, collate_fn = prepare_model_datasets(args)

    # Finetune 시에는 kfold 고려 / pretrain 시에는 고려 X
    if args.train_mode == "kfold":
        df = pd.read_csv(os.path.join(args.data_path, "Avicenna_train.csv"), encoding = "Windows-1252")
        X = df[df["Syllogistic relation"] == "yes"]["Premise 1"].to_list()
        y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()  

        if args.num_kfold != 0 :
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits = args.num_kfold, shuffle = True, random_state = args.seed)
            kfold_split = [(train_idx, val_idx) for train_idx, val_idx in splitter.split(X, y)]
            train_idx, val_idx = kfold_split[args.kfold_idx]
            train_idx = df[df["Syllogistic relation"] == "yes"].index[train_idx].to_list()
            val_idx = df[df["Syllogistic relation"] == "yes"].index[val_idx].to_list()

        else:
            train_idx = df[df["Syllogistic relation"] == "yes"].index.to_list()
            
        train_datasets = TrainDataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = train_idx)
        train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
    
    else : 
        train_datasets = TrainDataset(args, "Avicenna_train.csv", tokenizer)
        train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)


    args.data_length = len(train_datasets)
    wandb.config.update(args)
    

    print(f"""
    ============================
    Train Data Volume : {len(train_datasets)}
    ============================
    """)
    
    if args.num_kfold != 0 :
        print("=========================", "Train with Validation Set", "=========================")
        val_datasets = TestDataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = val_idx)
        val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
        train(args, model, train_dataloader, tokenizer, val_dataloader, wandb = wandb)
    else:
        print("=========================", "Train with Test Set", "=========================")
        test_dataset = TestDataset(args, "Avicenna_test.csv", tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False) 
        train(args, model, train_dataloader, tokenizer, test_dataloader, wandb = wandb)


def prepare_model_datasets(args):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    from data import CustomSyllogismGPT2TrainDataset, CustomSyllogismGPT2TestDataset, collate_fn_gpt
    additional_token_list = ['<|and|>','<|so|>']
    model = GPT2LMHeadModel.from_pretrained(args.load_model_path)
    model.config.max_length = args.max_len
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.load_tokenizer_path, 
        bos_token = '<|startoftext|>', 
        additional_special_tokens = additional_token_list)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    TrainDataset = CustomSyllogismGPT2TrainDataset
    TestDataset = CustomSyllogismGPT2TestDataset
    collate_fn = collate_fn_gpt

    return model, tokenizer, TrainDataset, TestDataset, collate_fn




if __name__ == "__main__":
    main()