#! /bin/bash

MAX_JOBS=3
running_jobs=0
files=$(aws s3 --endpoint https://s3-central.nrp-nautilus.io ls ${VIDEODIR}${SUBDIR}/ | grep mp4 | awk '{print $4}')
echo "Copying model checkpoint ${MODELNAME}"
aws s3 --endpoint $ENDPOINT cp s3://hengenlab/yolo/model/${MODELNAME} /models/${MODELNAME}

for DATANAME in $files; do
    echo "Copying data ${DATANAME}"
    aws s3 --endpoint $ENDPOINT cp ${VIDEODIR}${SUBDIR}/${DATANAME} /datasets/${DATANAME}
    python yolov5/detect.py --exist-ok --weights /models/${MODELNAME} --source /datasets/${DATANAME} --save-csv --save-txt --device 0 --nosave &
    ((running_jobs++))

    if [ "$running_jobs" -ge "$MAX_JOBS" ]; then
    wait -n
    ((running_jobs--))
    fi
done
wait
zip -r ${SUBDIR}_result.zip yolov5/runs/detect/exp
aws s3 --endpoint $ENDPOINT cp ${SUBDIR}_result.zip s3://hengenlab/yolo/results/${SUBDIR}/