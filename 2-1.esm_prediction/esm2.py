"""Source:
https://www.kaggle.com/code/daehunbae/esm-2-pseudo-perplexity-ranking
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np

import transformers
from transformers import EsmTokenizer, EsmForMaskedLM
import torch


def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    
    # mask one by one except [CLS] and [SEP]
    mask = torch.ones(tensor_input.size(-1) -1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.no_grad():
        loss = model(masked_input.to(device), labels=labels.to(device)).loss
    return np.exp(loss.item())



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('/home/jupyter-yehlin/DMSV2/dataset_final')

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

df = pd.read_csv('dmsv2_aa_seq.csv')
new_df = pd.DataFrame(columns=list(df.columns) + ['esm_pp'])
processed_names = set()

for index, row in df.iterrows():
    name = row['name']
    if name not in processed_names:
        score_val = score(model=model, tokenizer=tokenizer, sentence=row['aa_seq'])
        row['esm_pp'] = score_val
        new_df = new_df.append(row, ignore_index=True)
        
new_df.to_csv('dmsv2_aa_seq_esm.csv', index=False)