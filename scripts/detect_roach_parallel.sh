#! /bin/bash

ANIMAL=${ANIMAL}
MAX_JOBS=${MAX_JOBS}
running_jobs=0

files=$(aws s3 --endpoint https://s3-central.nrp-nautilus.io ls ${VIDEODIR}${SUBDIR}/ | grep mp4 | grep  ${ANIMAL} | awk '{print $4}')
echo "Copying model checkpoint ${MODELNAME}"
aws s3 --endpoint $ENDPOINT cp s3://hengenlab/t.tovi/${MODELNAME} /models/${MODELNAME}

for DATANAME in $files; do
    echo "Copying data ${DATANAME}"
    aws s3 --endpoint $ENDPOINT cp ${VIDEODIR}${SUBDIR}/${DATANAME} /datasets/${DATANAME}
    yolo predict model="/models/${MODELNAME}" source="/datasets/${DATANAME}" save_txt=True save=False save_conf=True device=0 name=${DATANAME}
    rm -f /datasets/${DATANAME} &
    ((running_jobs++))

    if [ "$running_jobs" -ge "$MAX_JOBS" ]; then
        wait -n
        ((running_jobs--))
    fi
done

wait

python scripts/compile_roach.py

for file in $(ls | grep npy);do
    aws --endpoint $ENDPOINT s3 rm s3://hengenlab/yolo/results/roach/${SUBDIR}/${file}
    aws --endpoint $ENDPOINT s3 cp ./${file} s3://hengenlab/yolo/results/roach/${SUBDIR}/
done

