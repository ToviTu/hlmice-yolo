#!/bin/bash

ANIMALS="JMG1 JMG4 JMG5 JMG6 JMG7 JMG8" #ANIMALS="JMG8"
count=0
for a in $ANIMALS;
do 
    export TAG="$( echo ${a} | awk '{print tolower($0)}')"
    export ANIMAL="${a}"

    echo "Submitting jobs for $a"
    ((count++))

    envsubst <./k8s/yolo_detect.yml> job-filled.yml
    kubectl apply -f job-filled.yml
    rm job-filled.yml
done
