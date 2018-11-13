#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=10:00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j_1000.out  # %N for node name, %j for jobID
#SBATCH --account=def-dpmeger

#SBATCH --mail-user=melissa.mozifian@mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
echo 'Training policy...'
module load cuda cudnn python/3.5
source ~/projects/def-dpmeger/melfm24/tensorflow/bin/activate
python train_bc_policy.py Humanoid-v2 --num_epochs 1000
echo 'Done.'
