import tensorflow as tf
import numpy as np

import sys
import os
sys.path.insert(0,os.path.split(os.path.abspath(__file__))[0] + '/..')
import tf_cylindrical as cylin


DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(x, like):
    like_shape = like.shape[1:3]
    return tf.image.resize(x, like_shape)

class depth_ego_net_compatible(tf.keras.Model):
    def __init__(self, opt):
        super(depth_ego_net_compatible, self).__init__()
        self.flags = opt
        depth_net_name = "depth_net/"
        self.c1_d = cylin.conv2d(32, [7, 7], stride=2, padding='CYLIN', name = depth_net_name + "cnv1")
        self.c1b_d = cylin.conv2d(32, [7, 7], stride=1, padding='CYLIN', name = depth_net_name + "cnv1b")
        self.c2_d = cylin.conv2d(64, [5, 5], stride=2, padding='CYLIN', name = depth_net_name + "cnv2")
        self.c2b_d = cylin.conv2d(64, [5, 5], stride=1, padding='CYLIN', name = depth_net_name + "cnv2b")
        self.c3_d = cylin.conv2d(128, [3, 3], stride=2, padding='CYLIN', name = depth_net_name + "cnv3")
        self.c3b_d = cylin.conv2d(128, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "cnv3b")
        self.c4_d = cylin.conv2d(256, [3, 3], stride=2, padding='CYLIN', name = depth_net_name + "cnv4")
        self.c4b_d = cylin.conv2d(256, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "cnv4b")
        self.c5_d = cylin.conv2d(512, [3, 3], stride=2, padding='CYLIN', name = depth_net_name + "cnv5")
        self.c5b_d = cylin.conv2d(512, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "cnv5b")
        self.c6_d = cylin.conv2d(512, [3, 3], stride=2, padding='CYLIN', name = depth_net_name + "cnv6")
        self.c6b_d = cylin.conv2d(512, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "cnv6b")
        self.c7_d = cylin.conv2d(512, [3, 3], stride=2, padding='CYLIN', name = depth_net_name + "cnv7")
        self.c7b_d = cylin.conv2d(512, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "cnv7b")
        
        self.uc7_d = cylin.conv2dTranspose(512, [3,3], stride = 2, name = depth_net_name + "upcnv7")
        self.ic7_d = cylin.conv2d(512, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv7")
        self.uc6_d = cylin.conv2dTranspose(512, [3,3], stride = 2, name = depth_net_name + "upcnv6")
        self.ic6_d = cylin.conv2d(512, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv6")
        self.uc5_d = cylin.conv2dTranspose(256, [3,3], stride = 2, name = depth_net_name + "upcnv5")
        self.ic5_d = cylin.conv2d(256, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv5")
        self.uc4_d = cylin.conv2dTranspose(128, [3,3], stride = 2, name = depth_net_name + "upcnv4")
        self.ic4_d = cylin.conv2d(128, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv4")
        self.uc3_d = cylin.conv2dTranspose(64, [3,3], stride = 2, name = depth_net_name + "upcnv3")
        self.ic3_d = cylin.conv2d(64, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv3")
        self.uc2_d = cylin.conv2dTranspose(32, [3,3], stride = 2, name = depth_net_name + "upcnv2")
        self.ic2_d = cylin.conv2d(32, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv2")
        self.uc1_d = cylin.conv2dTranspose(16, [3,3], stride = 2, name = depth_net_name + "upcnv1")
        self.ic1_d = cylin.conv2d(16, [3, 3], stride=1, padding='CYLIN', name = depth_net_name + "icnv1")

        self.d4_d = cylin.conv2d(1, [3,3], stride = 1, padding = "CYLIN", L2Regularization = 0, 
                activation = tf.sigmoid, name = depth_net_name + "disp4")
        self.d3_d = cylin.conv2d(1, [3,3], stride = 1, padding = "CYLIN", L2Regularization = 0, 
                activation = tf.sigmoid, name = depth_net_name + "disp3")
        self.d2_d = cylin.conv2d(1, [3,3], stride = 1, padding = "CYLIN", L2Regularization = 0, 
                activation = tf.sigmoid, name = depth_net_name + "disp2")
        self.d1_d = cylin.conv2d(1, [3,3], stride = 1, padding = "CYLIN", L2Regularization = 0, 
                activation = tf.sigmoid, name = depth_net_name + "disp1")


        pose_net_name = "pose_exp_net"
        self.num_sources = 2
        self.c1_p = cylin.conv2d(16,[7,7], stride=2, padding = "CYLIN", name = pose_net_name + "cnv1")
        self.c2_p = cylin.conv2d(32,[3,5], stride=2, padding = "CYLIN", name = pose_net_name + "cnv2")
        self.c3_p = cylin.conv2d(64,[3,3], stride=2, padding = "CYLIN", name = pose_net_name + "cnv3")
        self.c4_p = cylin.conv2d(128,[3,3], stride=2, padding = "CYLIN", name = pose_net_name + "cnv4")
        self.c5_p = cylin.conv2d(256,[3,3], stride=2, padding = "CYLIN", name = pose_net_name + "cnv5")
        self.c6_p = cylin.conv2d(256,[3,3], stride=2, padding = "CYLIN", name = pose_net_name + "pose/cnv6")
        self.c7_p = cylin.conv2d(256,[3,3], stride=2, padding = "CYLIN", name = pose_net_name + "pose/cnv7")
        self.pred_p = cylin.conv2d(6*opt.num_source,[1,1], stride=1, padding = "CYLIN", name = pose_net_name + "pose/pred")

        self.last_outs = None
        self.saliency_tracked_layers = self.layers
        '''[self.c1_d, self.c1b_d, self.c2_d, self.c2b_d, self.c3_d, self.c3b_d, 
            self.c4_d, self.c4b_d, self.c5_d, self.c5b_d, self.c6_d, self.c6b_d, self.c7_d, self.c7b_d,
            self.uc]'''

    def parameter_count(self):
        return tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in self.variables])

    def getDepth(self, x):
        H = x.shape[1]
        W = x.shape[2]

        y = self.c1_d(x)
        y1b = self.c1b_d(y)
        y = self.c2_d(y1b)
        y2b = self.c2b_d(y)
        y = self.c3_d(y2b)
        y3b = self.c3b_d(y)
        y = self.c4_d(y3b)
        y4b = self.c4b_d(y)
        y = self.c5_d(y4b)
        y5b = self.c5b_d(y)
        y = self.c6_d(y5b)
        y6b = self.c6b_d(y)
        y = self.c7_d(y6b)
        y = self.c7b_d(y)

        y = self.uc7_d(y)
        y = resize_like(y, y6b)
        y = tf.concat([y, y6b], axis = 3)
        y = self.ic7_d(y)
        y = self.uc6_d(y)
        y = resize_like(y, y5b)
        y = tf.concat([y, y5b], axis = 3)
        y = self.ic6_d(y)
        y = self.uc5_d(y)
        y = resize_like(y, y4b)
        y = tf.concat([y, y4b], axis = 3)
        y = self.ic5_d(y)
        y = self.uc4_d(y)
        y = resize_like(y, y3b)
        y = tf.concat([y, y3b], axis = 3)
        y = self.ic4_d(y)
        disp4 = DISP_SCALING * self.d4_d(y) + MIN_DISP
        disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])
        y = self.uc3_d(y)
        y = tf.concat([y, y2b, disp4_up], axis = 3)
        y = self.ic3_d(y)
        disp3 = DISP_SCALING * self.d3_d(y) + MIN_DISP
        disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])
        y = self.uc2_d(y)
        y = tf.concat([y, y1b, disp3_up], axis = 3)
        y = self.ic2_d(y)
        disp2 = DISP_SCALING * self.d2_d(y) + MIN_DISP
        disp2_up = tf.image.resize_bilinear(disp2, [H, W])
        y = self.uc1_d(y)
        y = tf.concat([y, disp2_up], axis = 3)
        y = self.ic1_d(y)
        disp1 = DISP_SCALING * self.d1_d(y) + MIN_DISP
        

        self.last_outs = [p.last_out for p in self.saliency_tracked_layers]
        return [disp1, disp2, disp3, disp4]

    def getPoseFromTo(self, fromImages, toImage):
        x = tf.concat(fromImages, toImage, axis = 3)

        c1 = self.c1_p(x)
        c2 = self.c2_p(c1)
        c3 = self.c2_p(c2)
        c4 = self.c2_p(c3)
        c5 = self.c2_p(c4)
        c6 = self.c2_p(c5)
        c7 = self.c2_p(c6)
        pred = self.pred_p(c7)
        pred = tf.reduce_mean(pred, [1,2])
        pose_f = 0.01 * tf.reshape(pred, [-1,self.num_sources, 6])
        return pose_f
 
 
    '''
        x - [batch, sequence, width, height, rgb]
    '''
    def call(self, x):
        x_shape = x.shape
        depths = []
        poses = []
        for i in range(0,x_shape[1]-1):
            depth = self.getDepth(x[i])
            pose = self.getPoseFromTo(x[i],x[-1])
            depths.append(depth)
            poses.append(pose)
        return depths, poses

    def prune(self, metrics, kill_fraction = 0.1):
        assert(len(metrics) == len(self.saliency_tracked_layers))
        last_mask = None
        for dl, met in zip(self.layers[:-1], metrics):
            last_mask = dl.prune(met, last_mask, kill_fraction)
        self.layers[-1].prune(None, last_mask, kill_fraction)