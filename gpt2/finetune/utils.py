import torch
import numpy as np
import random
import os

import datasets
import nltk.translate.bleu_score as bleu
import tensorflow as tf
import tensorflow_text as text
from numpy import mean

from tqdm import tqdm
import pandas as pd
import gc

"""
Batch Generation For GPT2 : https://github.com/huggingface/transformers/pull/7552#issue-497255933
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def set_special_token(tokenizer, model, additional_special_tokens):
    special_token_dict = {'additional_special_tokens' : additional_special_tokens}
    tokenizer.add_special_tokens(special_token_dict)
    model.resize_token_embeddings(len(tokenizer))
    print("-"*15, "Special Token 추가 완료", "-"*15)

def train_iter(wandb, args, model, tokenizer, model_inputs, optimizer, step, num_iter, epoch, device, scheduler = None) :
    model_inputs = merge_input_label(model_inputs, tokenizer)
    model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
    outputs = model(**model_inputs)

    loss = outputs.loss/args.accumulation_steps
    loss.backward()

    if (num_iter+1) % args.accumulation_steps == 0 :
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        loss = loss.detach().cpu().item()
        if scheduler:
            scheduler.step()

        wandb.log({"train_loss" : loss, "epoch" : epoch, "steps" : step})
    return step

def log_validation(args, model, val_dataloader, tokenizer, wandb, epoch = 0) :
    if args.valid_epochs:            
        val_bleu, val_rouge, val_generation, val_loss = test(args, model, val_dataloader, tokenizer)
        
        log = {'val_bleu' : val_bleu, 'val_rouge' : val_rouge, 'val_loss' : val_loss}
        wandb.log(log)
        save_log(args, log, epoch)

    # if args.save_epochs:
    #     print("-"*10, "model & tokenizer & generation result saved", "-"*10)
    #     save_generation(val_generation, args, epoch = epoch)
    #     save_model(model, tokenizer, args, epoch = epoch)
    if (args.save_final_model) & (epoch == args.num_epochs - 1):
        print("-"*10, "final model & tokenizer & generation result saved", "-"*10)
        save_generation(val_generation, args, epoch = epoch)
        save_model(model, tokenizer, args, epoch = epoch)

    print(f"valid bleu : {val_bleu}")
    print(f"valid rouge : {val_rouge}")
    print(f"valid loss : {val_loss}")
    print("-"*50)

def test(args, model, dataloader, tokenizer) :
    model.eval()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    prediction_list = []
    label_list = []
    input_list = []
    loss_list = []

    scorer = Score_Calculator(tokenizer)

    with torch.no_grad():
        print("-"*15, f"Test", "-"*15)
        for model_inputs in tqdm(dataloader) :
            model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
            model_inputs_for_loss = merge_input_label(model_inputs, tokenizer)
            # calculate loss
            loss = model(**model_inputs_for_loss).loss.detach().cpu().item()
            model.zero_grad()
            loss_list.append(loss)
            
            # mask for conclusion (after <conc> token)
            # conc_token_id = tokenizer.additional_special_tokens[1]
            # conc_idxs = (model_inputs["input_ids"] == conc_token_id).nonzero()

            # generation_input_ids = torch.full(model_inputs["input_ids"].shape[0], int(args.max_len/2), fill_value=tokenizer.pad_token_id) # (batch_size, max_len/2) 0 : pad_token_id
            # for row, idx in conc_idxs : 
            #     generation_input_ids[row, -idx+1:] = model_inputs["input_ids"][row, :idx+1] # premise 1 + premise 2 + <conc>
            # generation_attn_mask = generation_input_ids != tokenizer.pad_token_id
            # # calculate bleu, rouge score
            prediction = model.generate(
                input_ids = model_inputs["decoder_input_ids"], 
                attention_mask = model_inputs["decoder_attention_mask"],
                max_length = args.max_len
                ).cpu()  
            labels = model_inputs["label_input_ids"].cpu().squeeze(0).tolist()
            labels = [[token for token in label if token != -100] for label in labels]
            
            scorer.update_score(labels, prediction)

            # save generation & label string
            input_str_batch = tokenizer.batch_decode(model_inputs["decoder_input_ids"].cpu().tolist(), skip_special_tokens = True)
            input_list.extend(input_str_batch)

            prediction_str_batch = tokenizer.batch_decode(prediction.tolist(), skip_special_tokens = True)
            prediction_list.extend(prediction_str_batch)

            label_str_batch = tokenizer.batch_decode(labels, skip_special_tokens = True)
            label_list.extend(label_str_batch)

    model.train()

    generation_df = pd.DataFrame({
        "input" : input_list,
        "generation" : prediction_list,
        "label" : label_list})
    
    valid_loss = sum(loss_list)/len(dataloader)

    bleu, rouge = scorer.get_final_score()
    del scorer, model_inputs, loss_list, prediction_list, label_list, input_list
    gc.collect()
    torch.cuda.empty_cache()
    
    return bleu, rouge, generation_df, valid_loss


class Score_Calculator():
    def __init__(self, tokenizer) :
        """
        calculator = Score_Calculator(tokenizer)
        for batch in test_dataloader:
            generation = model.generate(**batch)
            labels = batch["labels"]

            calculator.update_score(labels, generation)
        bleu, rouge = calculator.get_final_score()
        """
        self.rouge_scorer = text.metrics.rouge_l
        self.rouge_score = 0

        self.bleu_scorer = bleu.sentence_bleu
        self.bleu_score = 0

        self.tokenizer = tokenizer
        self.special_tokens = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]
        
        self.rouge_count = 0
        self.bleu_count = 0
    def update_score(self, label : torch.tensor, prediction : torch.tensor) :
        bleu_label = self.tokenizer.batch_decode(label, skip_special_tokens = True)
        bleu_prediction = self.tokenizer.batch_decode(prediction, skip_special_tokens = True)
        for lab, pred in zip(bleu_label, bleu_prediction):
            self.bleu_score += self.bleu_scorer([lab], pred)
            self.bleu_count += 1

        rouge_label = [tf.ragged.constant([[self.tokenizer.decode(token) for token in sent if token not in self.special_tokens]]) for sent in label]
        rouge_prediction = [tf.ragged.constant([[self.tokenizer.decode(token) for token in sent if token not in self.special_tokens]]) for sent in prediction]
        self.rouge_score += mean([self.rouge_scorer(label, prediction).f_measure.numpy()[0] for label, prediction in zip(rouge_label, rouge_prediction)])
        self.rouge_count += 1 

    def get_final_score(self):
        return self.bleu_score/self.bleu_count, self.rouge_score/self.rouge_count
    
    def compute(self, label : torch.tensor, prediction : torch.tensor) :
        bleu_label = self.tokenizer.batch_decode(label, skip_special_tokens = True)
        bleu_prediction = self.tokenizer.batch_decode(prediction, skip_special_tokens = True)
        bleu_score = 0
        for lab, pred in zip(bleu_label, bleu_prediction):
            bleu_score += self.bleu_scorer([lab], pred)
        bleu_score /= len(bleu_label)

        rouge_label = [tf.ragged.constant([[token for token in sent if token not in self.special_tokens]]) for sent in label]
        rouge_prediction = [tf.ragged.constant([[token for token in sent if token not in self.special_tokens]]) for sent in prediction]
        rouge_score = mean([self.rouge_scorer(label, prediction).f_measure.numpy()[0] for label, prediction in zip(rouge_label, rouge_prediction)])
        return bleu_score, rouge_score


def save_generation(file, args, epoch = None) :
    file.to_html(os.path.join(args.generation_path, args.fold_name, f"{args.fold_name}.html"), index = False)

# 모델 저장 코드 수정 
def save_model(model, tokenizer, args, epoch = None):
    save_path = os.path.join(args.model_path, args.fold_name)
    model.save_pretrained(os.path.join(save_path, f"{args.model_save_name}_{args.fold_name}"))
    tokenizer.save_pretrained(save_path)

def load_log_df(args):
    if not os.path.exists(os.path.join(args.log_dir, args.fold_name, "log.csv")):
        log_df = pd.DataFrame(columns = ["epoch", "val_bleu", "val_rouge", "val_loss"])
        log_df.to_csv(os.path.join(args.log_dir, args.fold_name, "log.csv"), index = False)
    else :
        log_df = pd.read_csv(os.path.join(args.log_dir, args.fold_name, "log.csv"))
    return log_df

def save_log(args, log, epoch = None):
    log["epoch"] = epoch
    log["data_length"] = args.data_length
    if args.num_kfold != 0 :
        log["kfold"] = args.kfold_idx
    log_df_new = pd.DataFrame(log, index = [0])
    log_df = load_log_df(args)
    log_df = log_df.append(log_df_new, ignore_index = True)
    log_df.to_csv(os.path.join(args.log_dir, args.fold_name, "log.csv"), index = False)

def merge_input_label(model_inputs, tokenizer) :
    input_ids = torch.cat([model_inputs["decoder_input_ids"], model_inputs["label_input_ids"]], dim = 1)
    attention_mask = torch.cat([model_inputs["decoder_attention_mask"], model_inputs["label_attention_mask"]], dim = 1)
    labels = torch.where(input_ids != tokenizer.pad_token_id, input_ids, -100) # -100 : ignore index in CE loss
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def merge_data(args, original_data_name,) :
    original_data_path = os.path.join(args.data_path, original_data_name)
    original_data = pd.read_csv(original_data_path, encoding = "Windows-1252")
    original_data["index"] = original_data.index

    augmented_data_path = os.path.join(args.data_path, f"Avicenna_{args.augmented_data}_generation.csv")
    augmented_data = pd.read_csv(augmented_data_path)
    augmented_data["Syllogistic relation"] = "yes"

    condition = (augmented_data.bertscore < args.bertscore_ceil) & (augmented_data.bertscore > args.bertscore_floor) & (augmented_data.rouge < args.rouge_ceil)
    augmented_data = augmented_data[condition][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]

    total_data = pd.concat([original_data, augmented_data])

    return total_data

