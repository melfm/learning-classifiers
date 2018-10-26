#!/bin/bash

batches=( 10000 30000 50000)
learning_rates=(0.005 0.01 0.02)
for i in "${batches[@]}"
do
    for j in "${learning_rates[@]}"
do
    echo 'Running with batch'
    echo i
    echo 'And learning rate '
    echo j
    sh train_policy_job.sh $i $j
done

