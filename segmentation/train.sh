#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=480:00:00
#SBATCH --partition=pasteur
#SBATCH --account=pasteur
#SBATCH --output=/pasteur/u/aunell/zzoutput_gemini_inside.txt
#SBATCH --exclude=pasteur1,pasteur2,pasteur3,pasteur4

python /pasteur/u/aunell/cryoViT/segmentation/train.py --log_status True &> /pasteur/u/aunell/cryoViT/segmentation/output_train.txt