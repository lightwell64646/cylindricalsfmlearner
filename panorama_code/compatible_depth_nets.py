import tensorflow as tf

import sys
sys.path.insert(0,"..")
import tf_cylindrical as cylin

class depth_net(tf.keras.Model):
    def __init__(self):
        super(depth_net, self).__init__()
        self.c1_d = cylin.conv2d(64, [5, 5], stride=2, padding='CYLIN', name = "c1")
        self.c2_d = cylin.conv2d(128, [5, 5], stride=2, padding='CYLIN', name = "c2")
        self.proc_d = cylin.conv2d(128, [3, 3], stride=1, padding='CYLIN', name = "proc")
        self.last_outs = None

    def parameter_count(self):
        return tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in self.variables])

    def getDepth(self, x):

    def getPoseFromTo(self, fromImages, toImage):
        x = tf.concat(fromImages, toImage)


    '''
        x - [batch, sequence, width, height, rgb]
    '''
    def call(self, x):
        x_shape = x.shape
        depths = []
        for i in range(1,x_shape[1]):
            depth = self.getDepth(x[i])
            pose = self.getPoseFromTo([x[i],x[0]])
            depths.append(depth)
            poses.append(pose)
        return depths, poses
        y = self.c1(x)
        y = self.c2(y)
        y = self.proc(y)
        y = self.proc(y)
        y = self.proc(y)
        y = tf.reduce_mean(tf.reduce_mean(y, axis = 2), axis = 1)
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [self.c1.last_out, self.c2.last_out, self.proc.last_out, self.l1.last_out]
        return y

    def prune(self, metrics, kill_fraction = 0.1):
        assert(len(metrics) == len(self.layers) - 1)
        last_mask = None
        for dl, met in zip(self.layers[:-1], metrics):
            last_mask = dl.prune(met, last_mask, kill_fraction = kill_fraction)
        self.layers[-1].prune(None, last_mask)