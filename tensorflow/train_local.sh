#!/usr/bin/env bash

MODULE=lachebeque

MODEL="${1}"
MODEL_VERSION="${2}"

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
   --train_steps=100 \
   --model=${MODEL} \
   --learning_rate=0.001 \
   --cell_size=64 \
   --hidden_layer_size=192