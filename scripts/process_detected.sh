#! /bin/bash

aws --endpoint $ENDPOINT s3 cp s3://hengenlab/yolo/results/prey_capture/prey_capture_result.zip ./
unzip prey_capture_result.zip

wait

mkdir pdata
python scripts/post_process.py yolov5/runs/detect/exp/labels/ pdata/
python scripts/compile_post_processed.py

mkdir processed_arrays

for file in $(ls | grep npy);do
    #aws --endpoint $ENDPOINT s3 cp ./${file} s3://hengenlab/yolo/results/${SUBDIR}/
    cp ./${file} ./processsed_arrays/
done

zip processed_arrays

aws --endpoint $ENDPOINT s3 cp ./processed_arrays s3://hengenlab/yolo/results/${SUBDIR}/

