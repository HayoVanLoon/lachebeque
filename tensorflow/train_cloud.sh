#!/usr/bin/env bash

PROJECT=$(gcloud config get-value project)
REGION=us-east1

TFVERSION=1.12

MODULE=lachebeque

MODEL="${1}"
MODEL_VERSION="${2}"

OUTDIR=gs://${PROJECT}/${MODEL}/trained/${MODEL_VERSION}
PKG_PATH=${PWD}/${MODULE}

JOBNAME=${MODEL}_${MODEL_VERSION}_$(date -u +%y%m%d_%H%M%S)

gsutil -m rm -rf ${OUTDIR}

gcloud ml-engine jobs submit training ${JOBNAME} \
    --module-name=${MODULE}.task \
    --package-path=${PKG_PATH} \
    --region=${REGION} \
    --job-dir=${OUTDIR} \
    --staging-bucket=gs://${PROJECT} \
    --scale-tier=BASIC_GPU \
    --runtime-version=${TFVERSION} \
    -- \
    --output_dir=${OUTDIR} \
    --train_data_path="gs://${PROJECT}/${MODULE}/data/train.data" \
    --eval_data_path="gs://${PROJECT}/${MODULE}/data/eval.data"  \
    --train_steps=3000 \
    --model=${MODEL} \
    --learning_rate=0.001 \
    --cell_size=64 \
    --hidden_layer_size=192 \
