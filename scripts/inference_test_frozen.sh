#!/bin/sh

python ${SSD_ROOTDIR}/infer_ssd_network.py \
     --eval_dir=${SSD_ROOTDIR}/infer_log/ \
     --dataset_dir=${SSD_ROOTDIR}/datasets/tfrecords/ \
     --dataset_name=pascalvoc_2007 \
     --dataset_split_name=test \
     --model_name=ssd_300_vgg \
     --batch_size=1 \
     --inference_graph_path=${SSD_ROOTDIR}/ssd_vgg_frozen.pb
