#!/bin/bash

for beta in 1000 500 100 50 10 5 2
do
  for seed in 7 13 39 42 51
  do
    mkdir checkpoints/densenet121_cifar100/"beta${beta}_seed${seed}"
    cp params_densenet121.py checkpoints/densenet121_cifar100/"beta${beta}_seed${seed}"/params.py
    python train.py --directory "beta${beta}_seed${seed}" --beta $beta --seed $seed
  done
done
