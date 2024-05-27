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

module load python

base_output_dir="/pasteur/u/aunell/cryoViT/ablation0524"

for i in {7..8}
do
    image_path="/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_00${i}_16.png"

    for backbone in dinov2_vitg14_reg
    do
        output_dir="${base_output_dir}/backbone_${backbone}"
        mkdir -p $output_dir
        image_dir="${output_dir}/image_${i}.png"
        python /pasteur/u/aunell/cryoViT/features/main.py --output_dir $image_dir --image $image_path --backbone $backbone &> ${output_dir}/log_${i}.txt
    done
done
