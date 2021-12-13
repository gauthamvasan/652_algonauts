#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out

module load python/3.8
module load cuda/11.2.2
source /home/vasan/src/rl_algorithms/rtrl/bin/activate

export PYTHONPATH="$PYTHONPATH:/home/vasan/scratch/pytorch-i3d:/home/vasan/scratch/652_algonauts"

for ind in {1..7}
do
for roi in 'LOC' 'FFA' 'STS' 'EBA' 'PPA' 'V1' 'V2' 'V3' 'V4';
do
for sub in '01' '02' '03' '04' '05' '06' '07' '08' '09' '10';
do
        SECONDS=0
        python i3d_train.py --roi $roi --sub $sub --layer_ind $ind
	echo "Baseline job $seed took $SECONDS"
done
done
done
