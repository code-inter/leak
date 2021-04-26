#!/bin/bash

lead_nets(){
GPU=$1
index=$2
nets=$3

CUDA_VISIBLE_DEVICES=$GPU python  main.py  --index $index --nets $nets
}



lead_nets  1 25  resnet56
