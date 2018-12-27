#!/usr/bin/env bash

MODULE=lachebeque

MODEL=$1
MODEL_VERSION="0_2"

OUTDIR=${PWD}/trained/${MODEL_VERSION}
PKG_PATH=${PWD}/${MODULE}

rm -rf ${OUTDIR}

gcloud ml-engine local train \
   --module-name=${MODULE}.task \
   --package-path=${PKG_PATH} \
   -- \
   --output_dir=${OUTDIR} \
   --train_data_path=${PWD}/data/train.data \
   --eval_data_path=${PWD}/data/eval.data \
   --learning_rate=0.01 \
   --train_steps=100 \
   --model=${MODEL}