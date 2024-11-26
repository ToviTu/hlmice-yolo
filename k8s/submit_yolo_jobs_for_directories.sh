#!/bin/bash

directories=$(aws --endpoint https://s3-central.nrp-nautilus.io s3 ls s3://hengenlab/yolo_videos/ | awk '{print $2}' | grep "prey_capture/")
count=8
for d in $directories;
do 
    export TAG=$count
    export SUBDIR="${d%/}"

    echo "Submitting jobs for $SUBDIR"
    ((count++))

    envsubst <./k8s/yolo_detect.yml> job-filled.yml
    kubectl apply -f job-filled.yml
    rm job-filled.yml
done
