#!/bin/sh

cd ${TENSORFLOW_ROOTDIR}
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=${SSD_ROOTDIR}/ssd_vgg_frozen.pb \
--out_graph=${SSD_ROOTDIR}/ssd_vgg_quantize.pb \
--outputs=\
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
--inputs='eval_batch:0' \
--transforms='add_default_attributes strip_unused_nodes(type=float, shape="1,300,300,3")  
remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms quantize_weights strip_unused_nodes sort_by_execution_order'
