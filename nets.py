import tensorflow as tf
import numpy as np

import tf_cylindrical as cylin

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

class pose_exp_net:
    def __init__(self, do_wrap=True):
        self.do_wrap = do_wrap
        self.cnv1 = cylin.conv2d(16,  [7, 7], stride=2, padding=pad, name='pose_exp_net/cnv1', 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv2 = cylin.conv2d(32,  [5, 5], stride=2, padding=pad, name='pose_exp_net/cnv2', 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv3 = cylin.conv2d(64,  [3, 3], stride=2, padding=pad, name='pose_exp_net/cnv3', 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv4 = cylin.conv2d(128, [3, 3], stride=2, padding=pad, name='pose_exp_net/cnv4', 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv5 = cylin.conv2d(256, [3, 3], stride=2, padding=pad, name='pose_exp_net/cnv5', 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.pose_pred = cylin.conv2d(cnv7, 6*num_source, [1, 1], padding=pad, scope='pose_pred',
            stride=1, normalizer_fn=None, activation_fn=None)
        self.layers = [self.cnv1, self.cnv2, self.cnv3, self.cnv4, self.cnv5, self.cnv6, self.cnv7, self.pose_pred]
    def __call__(self, tgt_image, src_image_stack):
        inputs = tf.concat([tgt_image, src_image_stack], axis=3)
        cnv1_o = self.cnv1(inputs)
        cnv2_o = self.cnv2(cnv1_o)
        cnv3_o = self.cnv2(cnv2_o)
        cnv4_o = self.cnv2(cnv3_o)
        cnv5_o = self.cnv2(cnv4_o)
        cnv6_o = self.cnv6(cnv5_o)
        cnv7_o = self.cnv6(cnv6_o)
        pose_pred_o = self.pose_pred(cnv7_o)
        pose_avg = tf.reduce_mean(pose_pred_o, [1, 2])
        # Empirically we found that scaling by a small constant
        # facilitates training.
        pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
        return pose_final


class disp_net:
    def __init__(self, do_wrap=True):
        self.do_wrap = do_wrap
        self.cnv1 = step_stop(32)("cnv1", width = 7, stride = 4, kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv2 = step_stop(64)("cnv2", width = 5, stride = 3, kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv3 = step_stop(128)("cnv3", kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.cnv4 = step_stop(256)("cnv4", kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.processing_stage = resnet(cylin.conv2d, "processing", kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.icnv2 = cylin.conv2d(128,  [width, width], stride=1, padding=pad, name = "icnv2", 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.icnv1 = cylin.conv2d(1,  [width, width], stride=1, padding=pad, name = "icnv1", 
            kernel_regularizer = tf.keras.regularizers.l2(0.05), activation = tf.nn.relu)
        self.layers = self.cnv1.layers + self.cnv2.layers + self.cnv3.layers + self.cnv4.layers + self.processing_stage.layers + [self.icnv2] + [self.icnv1]
    def __call__(self, tgt_image):
        pad = 'CYLIN' if self.do_wrap else 'SAME'
        cnv1_o = self.cnv1(tgt_image)
        cnv2_o = self.cnv2(cnv1_o)
        cnv3_o = self.cnv2(cnv2_o)
        cnv4_o = self.cnv2(cnv3_o)
        proc = self.processing_stage(cnv4_o)
        proc = self.processing_stage(proc)
        proc = self.processing_stage(proc)
        proc = self.processing_stage(proc)
        upcnv1 = resize_like(proc, cnv2_o)
        i1_in  = tf.concat([upcnv1, cnv2_o], axis=-1)
        icnv1_o = self.iconv1(i1_in)
        upcnv2 = resize_like(icnv2_o, tgt_image)
        i2_in  = tf.concat([upcnv2, tgt_image], axis=-1)
        icnv2_o = self.iconv2(i2_in)
        return icnv2_o

