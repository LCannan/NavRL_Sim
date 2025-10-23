#!/bin/bash

# Exit immediately if a command fails
set -e

# Define environment name
ENV_NAME="NavRL"

# Load Conda environment handling
eval "$(conda shell.bash hook)"
# conda create -n $ENV_NAME python=3.10 -c conda-forge

# Re-activate the environment
conda activate $ENV_NAME

# Step 5: Install TensorDict and dependencies
echo -e "\033[0;31mInstalling TensorDict dependencies...\033[0m"
pip uninstall -y tensordict
pip uninstall -y tensordict
pip install tomli  # If missing 'tomli'
cd ./third_party/tensordict
python setup.py develop

# Step 6: Install TorchRL
echo -e "\033[0;31mInstalling TorchRL...\033[0m"
cd ../rl
python setup.py develop

# Check which torch is being used
python -c "import torch; print(torch.__path__)"

echo -e "\033[0;31mSetup completed successfully!\033[0m"

