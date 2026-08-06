# JointDesign
A comprehensive toolkit for protein design using joint TrROS/TrMRF models with zero-shot stability prediction using ESMFold, ESM2, and ProteinMPNN models.

## System Requirements

### Hardware Requirements
- NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- Minimum 16GB RAM
- 50GB free disk space


### Tested Environments
- CUDA 12.5
- Python 3.7
- Tested on:
  - NVIDIA A100 80GB

## Installation Guide
1. Clone the repository:
```bash
git clone https://github.com/yehlincho/Joint_Model_Stability.git
cd Joint_Model_Stability
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a conda environment with Python 3.7
- Install all required Python packages
- Download pre-trained models (ESMFold, TrRosetta, TrMRF)
- Download stability dataset
- Set up Jupyter kernel

## Model Weights

The pre-trained design-model weights are hosted on Zenodo.
Download and unzip them into `design_models/` before running any design notebook.

| Bundle | Contents | Consumed by |
|--------|----------|-------------|
| `TrMRF_weights.zip` (~19 MB) | `model_TrMRF_A.npy` (5-block), `model_TrMRF_seqid_retrain_3blocks.npy` (3-block), `model_TrMRF_seqid_retrain_5blocks.npy` (5-block retrain) | `design_models/TrMRF.ipynb`, `models.py` `TrMRF()` |
| `TrROS_weights.zip` (~162 MB) | trRosetta ensemble `models/model_xa[a-e].npy` (5 models) + background models `bkgr_models/bkgr0[1-5].npy` (5 models) | `design_models/TrROS.ipynb`, `models.py` `mk_design_model()` |

**Zenodo record:** https://sandbox.zenodo.org/records/582524 &nbsp;|&nbsp; **DOI:** [10.5072/zenodo.582524](https://doi.org/10.5072/zenodo.582524)

```bash
cd design_models
# after downloading both zips from Zenodo into design_models/
unzip -o TrMRF_weights.zip   # -> model_TrMRF_*.npy
unzip -o TrROS_weights.zip   # -> models/ and bkgr_models/
```

After unzipping, `design_models/` should contain the three `model_TrMRF_*.npy` files plus the `models/` and `bkgr_models/` directories, matching the default `DB_DIR="."` paths in `models.py`.

> Note: the TrROS bundle contains the standard public trRosetta / trDesign weights (originally from `files.ipd.uw.edu`). They are re-hosted here only for a self-contained download; the commented `wget` cell in `TrROS.ipynb` fetches the same files.

## Demo

### Running the Demo

1. Activate the conda environment:
```bash
conda activate joint_design1
```
2. Launch Jupyter:
```bash
jupyter notebook
```
3. Open `design_models/joint_models.ipynb` for protein design demo:
- This notebook demonstrates joint protein design using TrMRF and TrORS models
- Output: Designed protein sequences and structures

4. For zero shot stability prediction, use notebooks in `zero_shot_models/`:
- `esmfold/`: ESMFold distogram cross entropy and pLDDT based prediction
- `esm/`: ESM2 pseudo perplexcity based prediction
- `proteinmpnn/`: ProteinMPNN unconditional/conditional cross entropy based prediction

## Usage Instructions
### Protein Design

1. Using Joint Models:
```python
from models import mk_design_model
# Initialize model
model = mk_design_model(add_pdb=True, 
                       add_TrMRF=True, 
                       add_TrROS=True,
                       msa_design=False,
                       serial=True)

# Design protein
design = model.design(inputs={"pdb": pdb_features,
                            "I": sequence_features}, 
                    opt_iter=100,
                    hard=False, 
                    hard_switch=[50],
                    num=100,
                    return_traj=False, 
                    verbose=True, 
                    seqid=1.0)
```

2. Using Individual Models:
- TrMRF: Use `design_models/TrMRF.ipynb`
- TrROS: Use `design_models/TrROS.ipynb`


## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or inquiries, please contact Yehlin Cho at yehlin@mit.edu