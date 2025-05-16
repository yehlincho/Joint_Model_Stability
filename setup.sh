#!/bin/bash

# Create and activate conda environment
conda create -n joint_design1 python=3.7 -y
conda activate joint_design1

# Install basic requirements
pip install -r requirements.txt
pip3 install torch torchvision torchaudio
conda install "jaxlib=*=*cuda*" jax -c conda-forge

# Download ESMFold model
wget https://colabfold.steineggerlab.workers.dev/esm/esmfold.model

# Install ColabDesign and create symlink
pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.0
ln -s /usr/local/lib/python3.7/dist-packages/colabdesign colabdesign

# Install specific package versions
pip uninstall dm-haiku -y
pip install dm-haiku==0.0.9
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# Install transformers from specific branch
git clone -b add_esm-proper --single-branch https://github.com/liujas000/transformers.git
pip -q install ./transformers

# Install packaging
pip install packaging==21.3

# Setup Jupyter kernel
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=joint_design1


## Download stability dataset
gdown https://drive.google.com/file/d/1VRBjHc0HTlLsYj4BaecipTpflz6LOU1-/view?usp=drive_link

cd design_models

# Download TrRosetta models
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/model_TrMRF_A.npy
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/model_TrMRF_seqid_retrain_3blocks.npy
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/models.zip
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/bkgr_models.zip
unzip -qqo models.zip
unzip -qqo bkgr_models.zip
wget -qnc https://files.ipd.uw.edu/krypton/TrRosetta/model_TrMRF_seqid_retrain_5blocks.npy
wget -qnc https://files.ipd.uw.edu/krypton/for_gabe.zip
unzip -qqo for_gabe.zip

# Go back to root directory
cd ..

echo "Setup completed successfully!" 
