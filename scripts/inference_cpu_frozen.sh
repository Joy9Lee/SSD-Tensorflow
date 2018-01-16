#!/bin/sh

python ${SSD_ROOTDIR}/infer_detection.py \
     --eval_dir=${SSD_ROOTDIR}/detection_log/ \
     --model_name=ssd_300_vgg \
     --batch_size=1 \
     --inference_graph_path=${SSD_ROOTDIR}/ssd_vgg_frozen.pb \
     -input_tfrecord_paths=${SSD_ROOTDIR}/datasets/tfrecords/voc_2007_test_000.tfrecord

