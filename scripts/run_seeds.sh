#!/bin/bash

lead_seed(){
GPU=$1
index=$2

CUDA_VISIBLE_DEVICES=$GPU python  main.py  --index $index
}





(lead_seed  0 20 &)
(lead_seed  0 21 &)
(lead_seed  0 22 &)
(lead_seed  0 23 &)
(lead_seed  0 24 &)
(lead_seed  0 25 &)
(lead_seed  0 26 &)
(lead_seed  0 27 &)
(lead_seed  0 28 &)
(lead_seed  0 29 &)
(lead_seed  0 30 &)
