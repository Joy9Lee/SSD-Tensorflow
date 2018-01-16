#!/bin/sh

python tf_convert_data.py \
  --dataset_name=pascalvoc \
  --dataset_dir=./datasets/VOCdevkit/VOC2007/ \
  --output_name=voc_2007_test \
  --output_dir=./datasets/tfrecords/
