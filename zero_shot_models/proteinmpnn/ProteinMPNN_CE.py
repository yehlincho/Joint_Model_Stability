# Standard library imports
import ast
import math
import os
import pickle
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio.PDB import *
from IPython.display import HTML
from scipy.special import softmax
import tqdm.notebook
import colabdesign
from colabdesign.mpnn import mk_mpnn_model, clear_mem
from colabdesign.shared.protein import pdb_to_string

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

import warnings, os, re
warnings.simplefilter(action='ignore', category=FutureWarning)
os.system("mkdir -p output")


restypes = ['A','R','N','D','C','Q','E','G','H', 'I','L','K','M','F','P','S','T','W','Y','V']


def get_pdb(pdb_code="", directory_path=""):
    pdb_file_path = os.path.join(directory_path, f"{pdb_code}")
    if os.path.isfile(pdb_file_path):
        return pdb_file_path
    else:
        return None
    
def MPNN_prediction(pdb, directory_path, chains="A",model_name = "v_48_020", homooligomer = False,fix_pos = "", inverse = False, rm_aa = "", unconditional_logits=False):
    if fix_pos == "": fix_pos = None
    rm_aa = ",".join(list(re.sub("[^A-Z]+","",rm_aa.upper())))
    if rm_aa == "": rm_aa = None
    pdb_path = get_pdb(pdb, directory_path)
    print(pdb_path)

    mpnn_model = mk_mpnn_model(model_name)

    mpnn_model.prep_inputs(pdb_filename=pdb_path,
                           chain=chains, homooligomer=homooligomer,
                           fix_pos=fix_pos, inverse=inverse,
                           rm_aa=rm_aa, verbose=True)
    if unconditional_logits:
        logits = mpnn_model.get_unconditional_logits()
    else:
        logits = mpnn_model.get_logits()
    
    # print("logits", logits)
    pssm = softmax(logits, -1)
    return pssm

def MPNN_sequence_design(pdb, directory_path, chains="A",model_name = "v_48_020", homooligomer = False,fix_pos = "", inverse = False, rm_aa = "", temperature = 0.1):
    if fix_pos == "": fix_pos = None
    rm_aa = ",".join(list(re.sub("[^A-Z]+","",rm_aa.upper())))
    if rm_aa == "": rm_aa = None
    pdb_path = get_pdb(pdb, directory_path)
    print(pdb_path)

    mpnn_model = mk_mpnn_model()

    mpnn_model.prep_inputs(pdb_filename=pdb_path,
                           chain=chains, homooligomer=homooligomer,
                           fix_pos=fix_pos, inverse=inverse,
                           rm_aa=rm_aa, verbose=True)
    
    logits = mpnn_model.sample(temperature=temperature)
    pssm = softmax(logits, -1)
    return pssm

def extract_sequence(pdb_code, directory_path, target_chain='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', os.path.join(directory_path, f"{pdb_code}"))
    
    # Dictionary to map three-letter codes to one-letter codes
    aa_mapping = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    
    # Initialize an empty sequence string
    sequence = ""
    
    for model in structure:
        for chain in model:
            # Check if the chain matches the target_chain
            if chain.id == target_chain:
                for residue in chain:
                    # Check if the residue is an amino acid
                    if is_amino_acid(residue):
                        # Convert three-letter code to one-letter code and append to sequence
                        sequence += aa_mapping[residue.get_resname()]
    
    return sequence

def is_amino_acid(residue):
    # List of three-letter codes for common amino acids
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
                   "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
                   "TYR", "VAL"]
    return residue.get_resname() in amino_acids

def calculate_CE(sequence, pssm):
    indices = [restypes.index(residue) if residue in restypes else -1 for residue in sequence]
    probability_list = [res[indices[n]] for n, res in enumerate(pssm)]
    CE = -sum(math.log(probability) for probability in probability_list)/len(probability_list)
    return CE


if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--Date', type = str,default='Debug',help='Date of experiment')
    parser.add_argument('--data_dir', type= str,default='/data/yehlin/af2_predicted',help='path of template data')
    parser.add_argument('--save_dir', type= str,default= '/home/jupyter-yehlin/Joint_Model_Stability/zero_shot_models/proteinmpnn/conditional_MPNN_pssm_v_48_010.csv',help='path of template data')
    parser.add_argument('--model_name', type= str,default='v_48_010',help='path of template data')
    parser.add_argument('--unconditional_logits',  default=False, help="unconditional logits") 
    
    print("running it ")

    args = parser.parse_args()
    config = vars(args)
    directory_path = config['data_dir']
    savedir =  config['save_dir']
    unconditional_logits = config['unconditional_logits']
    filenames = os.listdir(directory_path)
    
    loaded_list = []
    count = 0
    for itr, pdb_file in enumerate(filenames):  
        try:
            if count > 10:
                break
            seq = extract_sequence(pdb_file, directory_path = directory_path)
            pssm = MPNN_prediction(pdb_file, directory_path = directory_path , model_name=config['model_name'], unconditional_logits=unconditional_logits)
            CE = calculate_CE(seq, pssm)
            loaded_list.append([pdb_file, seq, CE])
            count += 1
        except Exception as e:
            print(f"An error occurred for file {pdb_file}: {e}. Continuing to the next file...")
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(loaded_list, columns=['name', 'sequence', 'CE'])
    df['name'] = df['name'].str.replace('_unrelaxed_model_1.pdb', '')
    df.to_csv(savedir, index=False)