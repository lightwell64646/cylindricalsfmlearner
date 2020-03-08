import tensorflow as tf
import sys
sys.path.insert(0, "..")
import tf_cylindrical as cylin

class mnist_net(tf.keras.Model):
    def __init__(self, opt):
        super(mnist_net, self).__init__()
        self.c1 = cylin.conv2d(250, [3, 3], stride=1, padding='VALID', name = "c1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.c2 = cylin.conv2d(250, [3, 3], stride=1, padding='VALID', name = "c2", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.proc = cylin.conv2d(250, [3, 3], stride=1, padding='VALID', name = "proc", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.flatten = cylin.flatten(name = "flattening")
        self.l1 = cylin.linear(500, name = "l1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.l2 = cylin.linear(10, name = "l2", L2Regularization=opt.l2_weight_reg)
        self.last_outs = None

    def parameter_count(self):
        return tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in self.variables])

    def call(self, x):
        y = tf.keras.layers.MaxPool2D()(self.c1(x))
        y = self.c2(y)
        y = self.proc(y)
        y = self.proc(y)
        y = self.proc(y)
        y = self.flatten(y)
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


class mnist_net_strong_initialization(tf.keras.Model):
    def __init__(self, opt):
        super(mnist_net_strong_initialization, self).__init__()
        self.c1 = cylin.conv2d(50, [5, 5], stride=1, padding='VALID', name = "c1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.proc = cylin.conv2d(20, [5, 5], stride=1, padding='VALID', name = "proc", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.l1 = cylin.linear(50, name = "l1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.l2 = cylin.linear(10, name = "l2", L2Regularization=opt.l2_weight_reg)
        self.last_outs = None

    def parameter_count(self):
        return tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in self.variables])

    def call(self, x):
        y = tf.keras.layers.MaxPool2D()(self.c1(x))
        #y = self.c2(y)
        y = tf.keras.layers.MaxPool2D()(self.proc(y))
        #y = self.proc(y)
        #y = self.proc(y)
        y = tf.reshape(y, [-1, y.shape[1] * y.shape[2] * y.shape[3]])
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [self.c1.last_out, self.proc.last_out, self.l1.last_out]
        return y

    def prune(self, metrics, kill_fraction = 0.1):
        assert(len(metrics) == len(self.layers) - 1)
        last_mask = None
        for dl, met in zip(self.layers[:-1], metrics):
            last_mask = dl.prune(met, last_mask, kill_fraction = kill_fraction)
        self.layers[-1].prune(None, last_mask)
