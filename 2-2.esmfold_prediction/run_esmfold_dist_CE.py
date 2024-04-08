import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
from scipy.special import softmax
import gc
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import *



df = pd.read_csv('all_4_ID.csv')
df.loc[df['name'].str.startswith('r1_'), 'sequence'] = df.loc[df['name'].str.startswith('r1_'), 'sequence'].apply(lambda x: x[:-1])
df.loc[df['name'].str.startswith('r2_'), 'sequence'] = df.loc[df['name'].str.startswith('r2_'), 'sequence'].apply(lambda x: x[5:-6])
df.loc[df['name'].str.startswith('r3_'), 'sequence'] = df.loc[df['name'].str.startswith('r3_'), 'sequence'].apply(lambda x: x[10:-11])
df.loc[df['name'].str.startswith('r4_'), 'sequence'] = df.loc[df['name'].str.startswith('r4_'), 'sequence'].apply(lambda x: x[13:-13])
df.loc[df['name'].str.startswith('r5_'), 'sequence'] = df.loc[df['name'].str.startswith('r5_'), 'sequence'].apply(lambda x: x[3:-3])
df.loc[df['name'].str.startswith('r6_'), 'sequence'] = df.loc[df['name'].str.startswith('r6_'), 'sequence'].apply(lambda x: x[8:-8])

df.loc[df['name'].str.startswith('r7_'), 'sequence'] = df.loc[df['name'].str.startswith('r7_'), 'sequence'].apply(lambda x: x[12:-12])
df.loc[df['name'].str.startswith('r8_'), 'sequence'] = df.loc[df['name'].str.startswith('r8_'), 'sequence'].apply(lambda x: x[4:-5])
df.loc[df['name'].str.startswith('r9_'), 'sequence'] = df.loc[df['name'].str.startswith('r9_'), 'sequence'].apply(lambda x: x[1:-2])
df.loc[df['name'].str.startswith('r10_'), 'sequence'] = df.loc[df['name'].str.startswith('r10_'), 'sequence'].apply(lambda x: x[12:-13])
df.loc[df['name'].str.startswith('r11_'), 'sequence'] = df.loc[df['name'].str.startswith('r11_'), 'sequence'].apply(lambda x: x[11:-12])
df.loc[df['name'].str.startswith('r12_'), 'sequence'] = df.loc[df['name'].str.startswith('r12_'), 'sequence'].apply(lambda x: x[9:-9])


df.loc[df['name'].str.startswith('r13_'), 'sequence'] = df.loc[df['name'].str.startswith('r13_'), 'sequence'].apply(lambda x: x[4:-4])
df.loc[df['name'].str.startswith('r14_'), 'sequence'] = df.loc[df['name'].str.startswith('r14_'), 'sequence'].apply(lambda x: x[2:-3])
df.loc[df['name'].str.startswith('r15_'), 'sequence'] = df.loc[df['name'].str.startswith('r15_'), 'sequence'].apply(lambda x: x[6:-6])
df.loc[df['name'].str.startswith('r16_'), 'sequence'] = df.loc[df['name'].str.startswith('r16_'), 'sequence'].apply(lambda x: x[4:-5])
df.loc[df['name'].str.startswith('r17_'), 'sequence'] = df.loc[df['name'].str.startswith('r17_'), 'sequence'].apply(lambda x: x[11:-12])
df.loc[df['name'].str.startswith('r18_'), 'sequence'] = df.loc[df['name'].str.startswith('r18_'), 'sequence'].apply(lambda x: x[10:-10])

df.loc[df['name'].str.startswith('r19_'), 'sequence'] = df.loc[df['name'].str.startswith('r19_'), 'sequence'].apply(lambda x: x[3:-4])
df.loc[df['name'].str.startswith('r20_'), 'sequence'] = df.loc[df['name'].str.startswith('r20_'), 'sequence'].apply(lambda x: x[9:-9])


def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0,:,1]

    bins = np.append(0,np.linspace(2.3125,21.6875,63))
    sm_contacts = softmax(output["distogram_logits"],-1)[0]
    sm_contacts = sm_contacts[...,bins<8].sum(-1)
    xyz = output["positions"][-1,0,:,1]
    mask = output["atom37_atom_exists"][0,:,1] == 1
    o = {"pae":pae[mask,:][:,mask],
             "plddt":plddt[mask],
             "sm_contacts":sm_contacts[mask,:][:,mask],
             "xyz":xyz[mask],
             "distogram_logits": output["distogram_logits"]}
    return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)


def esmfold_plddt(model, sequence, num_recycles=0, chain_linker=2, masking_rate= 0.5, get_LM_contacts= True, samples = 8):

    best_pdb_str = None
    best_ptm = 0
    best_output = None
    traj = []
    stochastic_mode = "LM"


    num_samples = 1 if samples is None else samples
    plddt_ls=[]
    ptm = []
    for seed in range(num_samples):
        torch.cuda.empty_cache()
        if samples is None:
            seed = "default"
            mask_rate = 0.0
            model.train(False)
        else:
            torch.manual_seed(seed)
            mask_rate = masking_rate if "LM" in stochastic_mode else 0.0
            model.train("SM" in stochastic_mode)

        output = model.infer(sequence,
                                                num_recycles=num_recycles,
                                                chain_linker="X"*chain_linker,
                                                residue_index_offset=512,
                                                mask_rate=mask_rate,
                                                return_contacts=get_LM_contacts)

        pdb_str = model.output_to_pdb(output)[0].cpu()
        output = tree_map(lambda x: x.cpu().numpy(), output)
        ptm = output["ptm"][0]
        plddt = output["plddt"][0,:,1].mean()
        plddt_ls.append(plddt)
        ptm_ls.append(ptm)
        
        traj.append(parse_output(output))
        print(f'{seed} ptm: {ptm:.3f} plddt: {plddt:.1f}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return plddt_ls, ptm_ls
        
        
def esmfold_dist_CE(model, sequence, query_pdb, num_recycles=0, chain_linker=2, masking_rate= 0.5, get_LM_contacts= True, samples = 8):

    epsilon = 1e-8
    log_CE_ls = []
    CE_ls = []
    plddt_ls = []
    ptm_ls = []
    stochastic_mode = "LM"
    
    query_dist = prep_input(query_pdb,  chain='A')["feat"]

    with torch.no_grad():
        for seed in range(samples):
            if samples is None:
                seed = "default"
                mask_rate = 0.0
                model.train(False)
            else:
                torch.manual_seed(seed)
                mask_rate = masking_rate if "LM" in stochastic_mode else 0.0
                model.train("SM" in stochastic_mode)

            output = model.infer(sequence,
                                num_recycles=num_recycles,
                                chain_linker="X" * chain_linker,
                                residue_index_offset=512)
                                # mask_rate=mask_rate,
                                            # return_contacts=get_LM_contacts
                    
            ptm = output["ptm"].cpu().numpy()[0]
            plddt = output["plddt"].cpu().numpy()[0,:,1].mean()
            plddt_ls.append(plddt)
            ptm_ls.append(ptm)
            


            output_dist_tensor = output["distogram_logits"][0].cpu()
            output_dist_numpy = output_dist_tensor.numpy()
            output_dist = softmax(output_dist_tensor, -1)
            
            log_p= -np.sum(np.log(output_dist+epsilon)*query_dist, axis=-1)
            log_CE = np.sum(log_p)/(len(log_p)*len(log_p))
            log_CE_ls.append(log_CE)
            p = np.sum(output_dist*query_dist, axis=-1)
            CE= np.sum(p)/(len(p)*len(p))
            CE_ls.append(CE)
            print(f'{seed} log_CE: {log_CE:.3f} CE: {CE:.3f}')
            
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return log_CE_ls, CE_ls, plddt_ls, ptm_ls


class CustomDataset(Dataset):
    def __init__(self, df_file):
        self.df = df_file

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.loc[idx, 'sequence']
        name = self.df.loc[idx, 'name']
        ID = self.df.loc[idx, 'ID']
        return {'name': name, 'sequence': sequence, 'ID':ID}


def run_seq_sampling_and_save_CE(model_dir, input_csv, output_csv, num_recycles=0, chain_linker=2, masking_rate=0.5, get_LM_contacts=True, samples=8,  batch_size=1):
    custom_dataset = CustomDataset(input_csv)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Adjust num_workers based on your system
    
    model = torch.load(model_dir)
    model.cuda().requires_grad_(False)
    model.trunk.set_chunk_size(128)

    
    for batch_index, batch in enumerate(data_loader):
        try:
            seq = batch['sequence']
            name = batch['name']
            # ID = batch['ID'][0]
            pdb_dir = f'af2_predicted/{name[0]}_unrelaxed_model_1.pdb'
            print(name)
            # log_CE_ls, CE_ls, plddt_ls, ptm_ls = esmfold_dist_CE(model, seq, pdb_dir, num_recycles, chain_linker, masking_rate, get_LM_contacts, samples)
            plddt_ls, ptm_ls = esmfold_plddt(model, seq, num_recycles, chain_linker, masking_rate, get_LM_contacts, samples)
            if samples ==1:
                # result_df = pd.DataFrame({'name': name, 'sequence': seq, 'log_CE': log_CE_ls,'CE': CE_ls, 'PLDDT': plddt_ls,'PTM': ptm_ls})
                result_df = pd.DataFrame({'name': name, 'sequence': seq, 'PLDDT': plddt_ls,'PTM': ptm_ls})

            else:        
                result_df = pd.DataFrame({'name': name, 'sequence': seq, 'log_CE': [log_CE_ls],'CE': [CE_ls], 'PLDDT': [plddt_ls],'PTM': [ptm_ls]})

            if os.path.exists(output_csv):
                result_df.to_csv(output_csv, mode='a', header=False, index=False)
            else:
                result_df.to_csv(output_csv, index=False)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e: 
            print(f"An error occurred: {e}")
            continue
            
            
            
# run_seq_sampling_and_save_CE('esmfold.model', df, 'all_4_full_seq_CE_af2.csv', samples=1)
df = pd.read_csv('/home/jupyter-yehlin/DMSV2/dataset/PafA/Protein_sequence_AP_orthos.csv')
run_seq_sampling_and_save_CE('esmfold.model', df, '/home/jupyter-yehlin/DMSV2/dataset/PafA/Protein_sequence_AP_orthos_ESMFold_plddt.csv', samples=1)