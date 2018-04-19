#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=80:00:00
#SBATCH --mem=64GB
##SBATCH --gres=gpu:1
#SBATCH --job-name=vggTest
#SBATCH --mail-type=END
##SBATCH --mail-user=xz2139@nyu.edu
#SBATCH --output=vgg_%j.out
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
module load nltk/python3.5/3.2.4
which python3
python3 model_draft.py


