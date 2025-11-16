#!/bin/bash
#SBATCH -n 24
#SBATCH --mem=256G
#SBATCH -t 3:00:00
#SBATCH -p batch
#SBATCH -o logs/alignment_%j.log
#SBATCH -e logs/alignment_%j.err

source venv/bin/activate

python alignment.py --dataset-fname $1 --target returns
# python alignment.py --dataset-fname $1 --x_include_actions --target next_obs