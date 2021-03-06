#!/usr/bin/env bash

PROJECT=$(gcloud config get-value project)
REGION=us-east1

TFVERSION=1.12

MODULE=lachebeque

MODEL="${1}"
MODEL_VERSION="${2}"

OUTDIR=gs://${PROJECT}/${MODEL}/trained/${MODEL_VERSION}

MODEL_LOCATION=$(gsutil ls ${OUTDIR}/export/exporter | tail -1)

echo "Deleting and deploying $MODEL $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL}
#gcloud ml-engine models delete ${MODEL}

gcloud ml-engine models create ${MODULE} \
    --regions ${REGION}
gcloud ml-engine versions create "${MODEL}_${MODEL_VERSION}" \
    --model ${MODULE} \
    --origin ${MODEL_LOCATION} \
    --runtime-version=${TFVERSION}