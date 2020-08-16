#!/bin/sh
SOURCE=dukemtmc
TARGET=market1501
seed=1
alpha=0.5
mu=0.1
R=5
ins=4
b=64
rho=1.6e-3
knn=6
ARCH=resnet50


CUDA_VISIBLE_DEVICES=1,0,2,3 \
python examples/dual_refine_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --seed ${seed} \
    --num-instances ${ins} --lr 0.00035 --iters 400 -b ${b} --dropout 0 --lambda-value 0 \
	  --init logs/${SOURCE}TO${TARGET}/source-pretrain-1/model_best.pth.tar \
	  --logs-dir logs/${SOURCE}TO${TARGET}/DualRefinement/seed${seed} \
	  --alpha ${alpha} --mu ${mu} --knn ${knn} --R ${R} \
	  --epochs 40 --epochs_decay 20 --rho ${rho}
	  # --rr-gpu
