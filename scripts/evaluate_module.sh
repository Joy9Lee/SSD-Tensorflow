#!/bin/sh

python eval_ssd_network.py \
--eval_dir=${SSD_ROOTDIR}/eval_log/ \
--dataset_dir=${SSD_ROOTDIR}/datasets/tfrecords/ \
--dataset_name=pascalvoc_2007  \
--dataset_split_name=test \
--model_name=ssd_300_vgg  \
--checkpoint_path=${SSD_ROOTDIR}/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt \
--batch_size=1
