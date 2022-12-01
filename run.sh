#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=varun.chhangani@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=log1.txt
#SBATCH -c 30
#SBATCH -w gnode039

set -x

echo "Loading Modules"
module load u18/python/3.7.4 u18/cuda/11.6 u18/cudnn/8.4.0-cuda-11.6



#echo "Setting Environment"
#source ~/env/bin/activate


bash ~/Research-Starter-Kit/jp.sh 18989 8989 10.2.16.76 varun
cd /ssd_scratch/cvit/
mkdir varunc
cd varunc

python ~/train_wink.py


