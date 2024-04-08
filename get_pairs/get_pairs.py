import os
# import colabdesign
# from colabdesign.mpnn import mk_mpnn_model, clear_mem
# from colabdesign.shared.protein import pdb_to_string
os.chdir('/home/gridsan/ylcho/')
import os
# import colabdesign
# from colabdesign.mpnn import mk_mpnn_model, clear_mem
# from colabdesign.shared.protein import pdb_to_string
os.chdir('/home/gridsan/ylcho/DMSV2/')
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import pandas as pd
import tqdm.notebook
import seaborn
import re, tempfile

import argparse
# from colabdesign.af.contrib import predictz
# from colabdesign.shared.protein import _np_rmsd

if "hhsuite" not in os.environ['PATH']:
    os.environ['PATH'] += ":/home/gridsan/ylcho/DMSV2/"
    
    
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
import shutil
import random



order = "ACDEFGHIKLMNPQRSTVWYX"
amino_acid_to_number = {aa: i for i, aa in enumerate(order)}

def parse_PDB_biounits(x, atoms=["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"], chain='A'):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'



def get_pairs_single(S, contact_pairwise_index):
    index_1 = contact_pairwise_index[:, 0].astype(np.int64)
    index_2 = contact_pairwise_index[:, 1].astype(np.int64)
    
    print(index_1)
    print(index_2)
    
    wt_1 = S[index_1]
    print(wt_1)
    wt_2 = S[index_2]
    print(wt_2)
    wt_pairwise = (wt_1 * 21) + wt_2
    
    true_pairwise = wt_pairwise.flatten()

    true_pairwise_count = np.bincount(true_pairwise).astype(int)

    return true_pairwise_count

def calculate_contacts_and_get_pairwise(xyz, seq, distance_threshold,pairwise_threshold=6):

    dict_coord = {}  # dict to store coordinates. dict_coord[res][atom] = (x, y, z)
    contacting_pairs = []  # List to store pairs of contacting residues as [ires, jres]

    # Precompute distance threshold squared
    distance_threshold_sq = distance_threshold ** 2

    for i, (xyz, res) in enumerate(zip(xyz, seq)):
        for atom_num, coords in enumerate(xyz):
            if not np.isnan(coords).any():  # Skip atoms with NaN coordinates
                x, y, z = coords
                res = i + 1  # Numeric index for the residue
                if res not in dict_coord:
                    dict_coord[res] = []
                dict_coord[res].append((x, y, z))

    res_list = list(dict_coord.keys())
    res_coords = [np.array(dict_coord[res]) for res in res_list]

    for i, ires_coords in enumerate(res_coords):
        for j in range(i + 1, len(res_list)):
            jres_coords = res_coords[j]
            distances = np.sum((ires_coords[:, np.newaxis] - jres_coords) ** 2, axis=-1)
            close_atoms = np.argwhere(distances < distance_threshold_sq)
            if close_atoms.size > 0 and abs(i-j)>=6:
                contacting_pairs.append([res_list[i]-1, res_list[j]-1])
                
    print(contacting_pairs)
                
    S = np.array([amino_acid_to_number[aa] for aa in seq])
    contacting_pairs= np.array(contacting_pairs)
    test_true_pairs = np.zeros(441, dtype=int)
    true_pairwise_count = get_pairs_single(S, contacting_pairs)
    test_true_pairs[:len(true_pairwise_count)] += true_pairwise_count

    return test_true_pairs



def main(directory, output_csv, distance):
    for file in os.listdir(directory):
        try:
            if file.endswith('.pdb'):
                file_path = os.path.join(directory, file)
                xyz, seq = parse_PDB_biounits(file_path)
                pairwise_count = calculate_contacts_and_get_pairwise(xyz, seq[0],distance)

                df = pd.DataFrame({'name': [file], 'pairwise_count': [pairwise_count]})

                # Check if the output CSV file exists
                if os.path.exists(output_csv):
                    # Append the new data to the existing CSV file without header
                    df.to_csv(output_csv, mode='a', header=False, index=False)
                else:
                    # Create a new CSV file without header
                    df.to_csv(output_csv, index=False)
        except Exception as e:
            print(f"An error occurred for file '{file}': {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDB files and calculate pairwise counts.')
    parser.add_argument('--directory', type=str, default='dataset/native_protein', help='Directory containing PDB files')
    parser.add_argument('--output_csv', type=str, default='pairwise_counts_native_5A.csv', help='Output CSV file')
    parser.add_argument('--distance', type=int, default=5, help='contact distance')
    args = parser.parse_args()

    main(args.directory, args.output_csv, args.distance)