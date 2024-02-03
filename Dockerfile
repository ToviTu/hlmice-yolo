# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /usr/src/app

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir /dataset/
RUN mkdir /models/

ENV ENDPOINT="https://s3-central.nrp-nautilus.io"
ENV S3_ENDPOINT="s3-central.nrp-nautilus.io"
ENV S3_USE_HTTPS=1
ENV AWS_LOG_LEVEL=3
COPY . .

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    wget \
    curl \
    dnsutils \
    nano \
    zip \
    unzip \
    git \
    s3cmd \
    ffmpeg \
    screen \
    fonts-freefont-ttf \
    inotify-tools \
    parallel \
    pciutils \
    ncdu \
    libbz2-dev \
    gettext \
    apt-transport-https \
    gnupg2 \
    time \
    openssl \
    redis-tools \
    ca-certificates \
    hdf5-tools

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/ultralytics/yolov5

