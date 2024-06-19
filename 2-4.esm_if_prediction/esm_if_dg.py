import os

import time,subprocess,re,sys,shutil
import torch
import numpy as np
import pandas as pd

import subprocess


def format_pytorch_version(version):
  return version.split('+')[0]

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)
CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

IF_model_name = "esm_if1_gvp4_t16_142M_UR50.pt"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

os.chdir('/home/gridsan/ylcho/DMSV2/2-4.esm_if_prediction')

import esm

from esm.inverse_folding.util import load_structure, extract_coords_from_structure,CoordBatchConverter
from esm.inverse_folding.multichain_util import extract_coords_from_complex,_concatenate_coords,load_complex_coords


print("importing the model")

model, alphabet = esm.pretrained.load_model_and_alphabet(IF_model_name)
model.eval().cuda().requires_grad_(False)

print("--> Installations succeeded")


def run_model(coords,sequence,model,cmplx=False,chain_target='A'):

    device = next(model.parameters()).device

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)

    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    logits_swapped=torch.swapaxes(logits,1,2)
    token_probs = torch.softmax(logits_swapped, dim=-1)

    return token_probs

def score_variants(sequence,token_probs,alphabet):

    aa_list=[]
    wt_scores=[]
    skip_pos=0

    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    for i,n in enumerate(sequence):
      aa_list.append(n+str(i+1))
      score_pos=[]
      for j in range(1,21):
          score_pos.append(masked_absolute(alphabetAA_D_L[j],i, token_probs, alphabet))
          if n == alphabetAA_D_L[j]:
            WT_score_pos=score_pos[-1]

      wt_scores.append(WT_score_pos)

    return aa_list, wt_scores

def masked_absolute(mut, idx, token_probs, alphabet):

    mt_encoded = alphabet.get_idx(mut)

    score = token_probs[0,idx, mt_encoded]
    return score.item()

a = 0.10413378327743603
b = 0.6162549378400894

# Initialize an empty DataFrame
df = pd.DataFrame(columns=['name', 'dg_IF', 'dg_kcalmol'])

pdb_ls = os.listdir('/home/gridsan/ylcho/DMSV2/dataset/af2_predicted')

# Load the existing CSV file
try:
    df = pd.read_csv('esm_if_dg.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['name', 'dg_IF', 'dg_kcalmol'])

# Create a set of processed PDBs
processed_pdbs = set(df['name'])

for pdb in pdb_ls:
    name = pdb.replace('.pdb', '')
    print(name)
    
    # Check if the PDB has already been processed
    if name in processed_pdbs:
        continue  # Skip already processed PDBs
    
    chain_id = 'A'
    structure = load_structure(f"/home/gridsan/ylcho/DMSV2/dataset/af2_predicted/{pdb}", chain_id)
    coords_structure, sequence_structure = extract_coords_from_structure(structure)
    
    prob_tokens = run_model(coords_structure, sequence_structure, model, chain_target=chain_id)
    aa_list, wt_scores = score_variants(sequence_structure, prob_tokens, alphabet)

    dg_IF = np.nansum(wt_scores)
    dg_kcalmol = a * dg_IF + b
    
    # Append a new row to the DataFrame
    # new_row = {'name': name, 'dg_IF': dg_IF, 'dg_kcalmol': dg_kcalmol}
    # df = df.append(new_row, ignore_index=True)
    
    new_row = {'name': name, 'dg_IF': dg_IF, 'dg_kcalmol': dg_kcalmol}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    
    # Save the updated DataFrame to the CSV file
    df.to_csv('esm_if_dg.csv', index=False)



# for pdb in pdb_ls:
#     chain_id = 'A'
#     structure = load_structure(f"esmfold_structures_recyc3/{pdb}", chain_id)
#     coords_structure, sequence_structure = extract_coords_from_structure(structure)
#     name = pdb.replace('.pdb', '')
    
#     prob_tokens = run_model(coords_structure, sequence_structure, model, chain_target=chain_id)
#     aa_list, wt_scores = score_variants(sequence_structure, prob_tokens, alphabet)

#     dg_IF = np.nansum(wt_scores)
#     dg_kcalmol = a * dg_IF + b
    
#     # Append a new row to the DataFrame
#     df = df.append({'name': name, 'dg_IF': dg_IF, 'dg_kcalmol': dg_kcalmol}, ignore_index=True)
#     df.to_csv('esm_if_dg.csv', index=False)
