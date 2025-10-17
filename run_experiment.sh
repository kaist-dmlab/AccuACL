#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Path to the file you want to run

file_path="acl/multi_round_baseline.py"
configs="--scenario SplitCIFAR10 --data CIFAR10 --n_classes 10 --epochs 100 --batch_size 16 --wandb --mem_size 1000 --num_workers 0 --al_strategy AccuACL"

for seed in 0 1 2
do  
    for cl in ER DER ER_ACE
    do
        additional_options="--cl_strategy $cl --seed $seed"
        command="python $file_path --nStart 100 --nQuery 100 --nEnd 1000 $configs $additional_options"
        
        echo "Running command: $command"
        eval $command
    done
done