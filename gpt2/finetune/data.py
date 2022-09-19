import torch
import os
from torch.utils.data import Dataset
from utils import merge_data

import pandas as pd

class CustomSyllogismDatasetMixin(Dataset):
    def __init__(self, args, data_name, tokenizer, kfold_idx = None):
        if (args.augment_mode) & ("train" in data_name):
            self.raw_data = merge_data(args, data_name)
        else:
            data_path = os.path.join(args.data_path, data_name)
            self.raw_data = pd.read_csv(data_path, encoding = 'Windows-1252')
            self.raw_data["index"] = self.raw_data.index
            
        self.raw_data = self.raw_data[self.raw_data["Syllogistic relation"] == "yes"]
        self.max_len = args.max_len
        self.batch_size = args.batch_size

        self.tokenizer = tokenizer
        self.index = self.raw_data["index"]

        self.prem1 = self.raw_data["Premise 1"].to_list()
        self.prem2 = self.raw_data["Premise 2"].to_list()
        self.label = self.raw_data["Conclusion"].to_list()

        if kfold_idx is not None:
            self.prem1 = [sent for num, sent in zip(self.index, self.prem1) if num in kfold_idx]
            self.prem2 = [sent for num, sent in zip(self.index, self.prem2) if num in kfold_idx]
            self.label = [sent for num, sent in zip(self.index, self.label) if num in kfold_idx]


        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.label),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n Label : {len(self.label)}"
    
    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx])

    def __len__(self):
        return len(self.prem1)

    def __preprocess(self, input_text) :
        encoder_text, decoder_target = input_text

        encoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        decoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        decoder_target_ids = torch.full((1, self.max_len), fill_value = self.target_pad_id)

        encoder_attn_mask = torch.zeros((1, self.max_len))
        decoder_attn_mask = torch.zeros((1, self.max_len))

        encoder_tokens = self.tokenizer.encode(encoder_text, add_special_tokens = False, return_tensors = 'pt')
        decoder_tokens = self.tokenizer.encode(decoder_target, add_special_tokens = False, return_tensors = 'pt')
        
        encoder_token_ids[0, :encoder_tokens.shape[1]] = encoder_tokens
        decoder_token_ids[0, :decoder_tokens.shape[1]-1] = decoder_tokens[:, :-1]
        decoder_target_ids[0, :decoder_tokens.shape[1]-1] = decoder_tokens[:, 1:]

        encoder_attn_mask[0, :encoder_tokens.shape[1]] = 1
        decoder_attn_mask[0, :decoder_tokens.shape[1]-1] = 1

        return encoder_token_ids, encoder_attn_mask, decoder_token_ids, decoder_attn_mask, decoder_target_ids

    def __get_special_tokens(self, args) :
        pass
    
    def __set_input_text(self) :
        pass

class CustomSyllogismGPT2TrainDataset(CustomSyllogismDatasetMixin) :
    def __init__(self, args, data_name, tokenizer, kfold_idx = None):
        super().__init__(args, data_name, tokenizer, kfold_idx)
        self.__get_special_tokens(args)
        self.__set_input_text()
    
    def __get_special_tokens(self, args):
        self.bos = self.tokenizer.bos_token
        self.eos = self.tokenizer.eos_token
        self.and_token = self.tokenizer.additional_special_tokens[0] # '<|and|>'
        self.conc = self.tokenizer.additional_special_tokens[1] #'<|so|>'

        self.input_pad_id = self.tokenizer.eos_token_id
        self.target_pad_id = -100

    def __set_input_text(self) :
        input_text = [self.bos + p1 + self.and_token + p2 + self.conc for p1, p2 in zip(self.prem1, self.prem2)]
        label_text = [label + self.eos for label in self.label]
        self.input_text = [(inp, lab) for inp, lab in zip(input_text, label_text)]

    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx])
    
    def __preprocess(self, input_text): # premise 1 + premise 2 + conclusion LM Train version
        decoder_text, label_text = input_text
        half_max_len = self.max_len // 2

        self.tokenizer.padding_side = "left"
        decoder_inputs = self.tokenizer(decoder_text, return_tensors = 'pt', add_special_tokens = False, max_length = half_max_len, padding="max_length")

        self.tokenizer.padding_side = "right"
        label_inputs = self.tokenizer(label_text, return_tensors = 'pt', add_special_tokens = False, max_length = half_max_len, padding="max_length")

        return decoder_inputs, label_inputs 

    # def __preprocess(self, input_text): # conclusion Only LM Train version
    #     decoder_text = input_text
        
    #     decoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
    #     label_token_ids = torch.full((1, self.max_len), fill_value = self.target_pad_id)
    #     decoder_attn_mask = torch.ones((1, self.max_len))

    #     decoder_tokens = self.tokenizer.encode(decoder_text, add_special_tokens = False, return_tensors = 'pt')

    #     conc_idx = (decoder_tokens == self.conc).nonzero()[0] 
    #     label_tokens = decoder_tokens
    #     label_tokens[:, :conc_idx + 1] = self.target_pad_id # ignore premise 1 and 2 when calculating loss

    #     decoder_token_ids[0, :decoder_tokens.shape[1]] = decoder_tokens[:, :] 
    #     label_token_ids[0, :label_tokens.shape[1]] = label_tokens[:, :]

    #     return decoder_token_ids, decoder_attn_mask, label_token_ids # input, attn_mask, label

class CustomSyllogismGPT2TestDataset(CustomSyllogismDatasetMixin) : # GPT2에 맞추어 모든 토큰에 대해 계산할지, Label에 대해서만 계산할지에 따라 코드 수정하기 
    def __init__(self, args, data_name, tokenizer, kfold_idx = None): 
        super().__init__(args, data_name, tokenizer, kfold_idx)
        self.__get_special_tokens(args)
        self.__set_input_text()

    def __get_special_tokens(self, args):
        self.bos = self.tokenizer.bos_token
        self.eos = self.tokenizer.eos_token
        self.and_token = self.tokenizer.additional_special_tokens[0] # '<|and|>'
        self.conc = self.tokenizer.additional_special_tokens[1] #'<|so|>'

        self.input_pad_id = self.tokenizer.eos_token_id
        self.target_pad_id = -100
        
    # def __set_input_text(self) :
    #     input_text = [self.bos + p1 + self.and_token + p2 + self.conc for p1, p2 in zip(self.prem1, self.prem2)]
    #     label_text = [label + self.eos for label in self.label]
    #     self.input_text = [(inp, lab) for inp, lab in zip(input_text, label_text)]

    def __set_input_text(self) :
        input_text = [self.bos + p1 + self.and_token + p2 + self.conc for p1, p2 in zip(self.prem1, self.prem2)]
        label_text = [label + self.eos for label in self.label]
        self.input_text = [(inp, lab) for inp, lab in zip(input_text, label_text)]

    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx])
    
    def __preprocess(self, input_text): # premise 1 + premise 2 + conclusion LM Train version
        decoder_text, label_text = input_text
        half_max_len = self.max_len // 2

        self.tokenizer.padding_side = "left"
        decoder_inputs = self.tokenizer(decoder_text, return_tensors = 'pt', add_special_tokens = False, max_length = half_max_len, padding="max_length")

        self.tokenizer.padding_side = "right"
        label_inputs = self.tokenizer(label_text, return_tensors = 'pt', add_special_tokens = False, max_length = half_max_len, padding="max_length")

        return decoder_inputs, label_inputs 

def collate_fn_gpt(batch) :
    
    input_ids = []
    input_attn_mask = []
    label_ids = []
    label_attn_mask = []
    
    for decoder_inputs, label_inputs in batch:
        input_ids.append(decoder_inputs["input_ids"])
        input_attn_mask.append(decoder_inputs["attention_mask"])
        label_ids.append(label_inputs["input_ids"])
        label_attn_mask.append(label_inputs["attention_mask"])

    model_inputs = {
        "decoder_input_ids" : torch.cat(input_ids, dim = 0),
        "decoder_attention_mask" : torch.cat(input_attn_mask, dim = 0),
        "label_input_ids" : torch.cat(label_ids, dim = 0),
        "label_attention_mask" : torch.cat(label_attn_mask, dim = 0)
    }
    return model_inputs
