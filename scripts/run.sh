#! /bin/bash

folder="s3://hengenlab/yolo_videos/"

files=$(aws s3 --endpoint https://s3-central.nrp-nautilus.io ls $folder | awk '{print $4}' | head -n 10)

for file in $files; do
    export DATANAME=${file}
    export ID=${RANDOM}
    kubectl apply -f k8s/yolo_detect.yml
    echo "Submitting jobs for ${DATANAME}"
done