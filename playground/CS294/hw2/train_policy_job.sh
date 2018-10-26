#!/bin/bash
python train_pg_f18.py HalfCheetah-v2 -ep 1500 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $1 -lr $2 -rtg --exp_name hc_b10000_r0.005
echo 'Done.'


