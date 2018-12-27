#!/usr/bin/env bash

gcloud ml-engine local predict \
    --model-dir=trained \
    --text-instances switzerland.txt \
    --verbosity debug
