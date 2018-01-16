#!/bin/sh

python ${TENSORFLOW_ROOTDIR}/tensorflow/python/tools/freeze_graph.py \
--input_checkpoint=${SSD_ROOTDIR}/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt \
--input_graph=${SSD_ROOTDIR}/ssd_vgg_graph.pbtxt \
--output_node_names=\
ssd_300_vgg/softmax/Reshape_1,\
ssd_300_vgg/softmax_1/Reshape_1,\
ssd_300_vgg/softmax_2/Reshape_1,\
ssd_300_vgg/softmax_3/Reshape_1,\
ssd_300_vgg/softmax_4/Reshape_1,\
ssd_300_vgg/softmax_5/Reshape_1,\
ssd_300_vgg/block4_box/Reshape,\
ssd_300_vgg/block7_box/Reshape,\
ssd_300_vgg/block8_box/Reshape,\
ssd_300_vgg/block9_box/Reshape,\
ssd_300_vgg/block10_box/Reshape,\
ssd_300_vgg/block11_box/Reshape,\
ssd_300_vgg/block4_box/Reshape_1,\
ssd_300_vgg/block7_box/Reshape_1,\
ssd_300_vgg/block8_box/Reshape_1,\
ssd_300_vgg/block9_box/Reshape_1,\
ssd_300_vgg/block10_box/Reshape_1,\
ssd_300_vgg/block11_box/Reshape_1 \
--output_graph=${SSD_ROOTDIR}/ssd_vgg_frozen.pb
