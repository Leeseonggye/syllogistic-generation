from arg_parser import make_parser
from train_utils import train
from utils import set_seed, merge_data, extract_premise

import torch
from torch.utils.data import DataLoader

import os
import pandas as pd
import wandb

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
        if not os.path.exists(os.path.join(args.generation_path, args.group_name)) :
            os.mkdir(os.path.join(args.generation_path, args.group_name))
            os.mkdir(os.path.join(args.model_path, args.group_name))
    else: 
        print("="*30)
        print("not save at all")
        print("="*30)

    if not os.path.exists(os.path.join(args.log_dir, args.group_name)) :
        os.mkdir(os.path.join(args.log_dir, args.group_name))

    model, tokenizer, TrainDataset, TestDataset, collate_fn = prepare_model_datasets(args)

    if args.num_augmentation == 0:

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
    
    elif args.num_augmentation > 0:

        if os.path.isfile(os.path.join(args.data_path,f"{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}.csv")):
            pass
        else:
            extract_premise(args)
            merge_data(args,"Avicenna_train.csv")
        
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
                
            train_datasets = TrainDataset(args, f"{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}.csv", tokenizer, kfold_idx = train_idx)
            train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
        
        else : 
            train_datasets = TrainDataset(args, f"{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}.csv", tokenizer)
            train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)


        args.data_length = len(train_datasets)
        wandb.config.update(args)
        

        print(f"""
        ============================
        Train Data Volume : {len(train_datasets)}
        ============================
        """)
        
        if args.num_kfold != 0 :
            print("=========================", f"Train with Validation Set with {args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}", "=========================")
            val_datasets = TestDataset(args, f"{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}.csv", tokenizer, kfold_idx = val_idx)
            val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
            train(args, model, train_dataloader, tokenizer, val_dataloader, wandb = wandb)
        else:
            print("=========================", f"Train with Test Set with {args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}", "=========================")
            test_dataset = TestDataset(args, "Avicenna_test.csv", tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False) 
            train(args, model, train_dataloader, tokenizer, test_dataloader, wandb = wandb)


def prepare_model_datasets(args):
    if "bart" in args.load_model_path :
        args.model = "bart"
    elif "t5" in args.load_model_path :
        args.model = "t5"
    elif "gpt" in args.load_model_path : 
        args.model = "gpt"
    else : 
        print(f"""
        =============================================================
        ============  잘못된 모델 입력 (BART, T5, GPT) ==============
        ===========      {args.load_model_path}     =================
        =============================================================
        """)

    if args.model == "bart" :
        from transformers import BartTokenizerFast, BartForConditionalGeneration
        from data import CustomSyllogismBARTDataset, collate_fn_bart
        model = BartForConditionalGeneration.from_pretrained(args.load_model_path)
        tokenizer = BartTokenizerFast.from_pretrained(args.load_tokenizer_path)
        
        TrainDataset = CustomSyllogismBARTDataset
        TestDataset = CustomSyllogismBARTDataset
        collate_fn = collate_fn_bart

    elif args.model == "t5" :
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from data import CustomSyllogismT5Dataset, collate_fn_t5
        model = T5ForConditionalGeneration.from_pretrained(args.load_model_path)
        tokenizer = T5Tokenizer.from_pretrained(args.load_tokenizer_path)
        special_tokens_dict = {"sep_token": "<sep>"}
        tokenizer.add_special_tokens(special_tokens_dict)

        TrainDataset = CustomSyllogismT5Dataset
        TestDataset = CustomSyllogismT5Dataset
        collate_fn = collate_fn_t5

    return model, tokenizer, TrainDataset, TestDataset, collate_fn




if __name__ == "__main__":
    main()