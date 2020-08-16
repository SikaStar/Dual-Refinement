#!/bin/sh
TARGET=$1
MODEL=$2

if [ $# -ne 2 ]
  then
    echo "Arguments error: <TARGET> <MODEL PATH>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 \
python examples/test_model.py -b 256 -j 8 -a resnet50 --dataset-target ${TARGET} --resume ${MODEL}
