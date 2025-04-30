#!/usr/bin/env bash

#SBATCH --job-name=pytorch-test
#SBATCH --mail-type=fail,end
#SBATCH --time=00-01:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# #SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:l40s:1

# shellcheck disable=SC1091
source load_env.sh

# Clone the GitHub repository
REPO_URL="https://github.com/Nivocoegi/NNDL_Project_Repo.git"
REPO_DIR="/cfs/earth/scratch/${USER}/cloned_NNDL_Project_Repo_from_git"

if [[ ! -d ${REPO_DIR} ]]; then
    git clone ${REPO_URL} ${REPO_DIR}
fi

hostname
# ## get GPU info
nvidia-smi

echo
echo "#########################################   Tensorflow Info"
echo

micromamba run -n pytorch python ${REPO_DIR}/pytorch_info.py

echo
echo "#########################################   DL part"
echo

# Run the Jupyter Notebook
micromamba run -n pytorch jupyter nbconvert --to notebook --execute ${REPO_DIR}/main.ipynb --output ${REPO_DIR}/main_output.ipynb

echo
echo "#########################################   Jupyter Notebook"
echo

micromamba run -n pytorch jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''

