#!/bin/sh
SOURCE=$1
TARGET=$2
SEED=$3

if [ $# -ne 3 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <SEED>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/baseline_train.py -ds ${SOURCE} -dt ${TARGET} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 --dropout 0 --lambda-value 0 \
	--init logs/${SOURCE}TO${TARGET}/source-pretrain-${SEED}/model_best.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/Baseline/seed${SEED} \
	--seed ${SEED}
	# --rr-gpu
