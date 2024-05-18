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

base_output_dir="/pasteur/u/aunell/cryoViT/ablation0517"
image_path="/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_001_16.png"

for crop_size in 448
do
    for dimensionality in Both
    do
        for backbone in dinov2_vitg14_reg 
        do
            for include_hsv in False
            do
                output_dir="${base_output_dir}/backbone_${backbone}/crop_${crop_size}/_dim_${dimensionality}_hsv_${include_hsv}"
                mkdir -p $output_dir
                python /pasteur/u/aunell/cryoViT/features/main.py --output_dir $output_dir --image $image_path --crop_size $crop_size --dimensionality $dimensionality --backbone $backbone --include_hsv $include_hsv &> ${output_dir}/log.txt
            done
        done
    done
done