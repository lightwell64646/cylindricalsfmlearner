import tensorflow as tf
import sys

sys.path.insert(0, "..")
from utils import get_loss_categorical, parameter_count
import tf_cylindrical as cylin


'''
A Keras model that predicts on MNIST

since we want to run in eager mode layer.output isn't available.
We use self.last_outs instead for calculating second derivative loss.
    Note: We write code in eager mode and use tf.function to run in 
        graph mode, but the code still has to compile as eager.

    Note: If you try to add optimizer to the model parameters it expects
        a fully compiled model which is (to my knowledge) not compatible
        with a custom training loop. Therefore optimizer is stored in the
        mnist_prune_trainer classes (single GPU and distributed).
'''
class mnist_net(tf.keras.Model):
    def __init__(self, opt):
        super(mnist_net, self).__init__()
        self.c1 = cylin.conv2d(60, [3, 3], stride=1, padding='VALID', name = "c1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.c2 = cylin.conv2d(60, [3, 3], stride=1, padding='VALID', name = "c2", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.c3 = cylin.conv2d(60, [3, 3], stride=1, padding='VALID', name = "c3", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.num_convolutional_layers = 3

        self.flatten = cylin.flatten(name = "flattening")
        self.l1 = cylin.linear(100, name = "l1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.l2 = cylin.linear(10, name = "l2", L2Regularization=opt.l2_weight_reg)

        self.last_outs = None
        self.saliency_tracked_layers = [self.c1, self.c2, self.c3, self.l1]

    def scramble(self):
        self.c1.scramble()
        self.c2.scramble()
        self.c3.scramble()
        self.l1.scramble()

    def call(self, x):
        y = self.c1(x)
        y = tf.keras.layers.MaxPool2D()(y)
        y = self.c2(y)
        y = self.c3(y)
        y = self.flatten(y)
        #y = tf.reduce_sum(y, [1,2])
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [p.last_out for p in self.saliency_tracked_layers]
        return y

    def prune(self, metrics, kill_fraction = 0.1, kill_low = True, const_percent = False):
        assert(len(metrics) == len(self.saliency_tracked_layers))
        masks = []
        decision_points = []

        last_mask = None
        i = 0
        for dl, met in zip(self.saliency_tracked_layers, metrics):
            #if (i == 0 or i == 1 or i == 3 or i == 2):
            last_mask, decision_pt = dl.prune(met, last_mask, kill_fraction, kill_low, const_percent)
            '''else:
                dl.prune(None, last_mask, kill_fraction, kill_low, const_percent)
                last_mask = tf.constant([i for i in range(dl.units)])
                decision_pt = None'''
            masks.append(last_mask)
            decision_points.append(decision_pt)
            i += 1
            if i == self.num_convolutional_layers:
                last_mask = self.flatten.prune(None, last_mask)
        self.layers[-1].prune(None, last_mask, kill_fraction)
        return masks, decision_points

class mnist_net_prune_fin_shape(tf.keras.Model):
    def __init__(self, opt):
        super(mnist_net_prune_fin_shape, self).__init__()
        self.c1 = cylin.conv2d(23, [3, 3], stride=1, padding='VALID', name = "c1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.c2 = cylin.conv2d(33, [3, 3], stride=1, padding='VALID', name = "c2", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.c3 = cylin.conv2d(6, [3, 3], stride=1, padding='VALID', name = "c3", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.num_convolutional_layers = 3

        self.flatten = cylin.flatten(name = "flattening")
        self.l1 = cylin.linear(71, name = "l1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.l2 = cylin.linear(10, name = "l2", L2Regularization=opt.l2_weight_reg)

        self.last_outs = None
        self.saliency_tracked_layers = [self.c1, self.c2, self.c3, self.l1]

    def scramble(self):
        self.c1.scramble()
        self.c2.scramble()
        self.c3.scramble()
        self.l1.scramble()

    def call(self, x):
        y = self.c1(x)
        y = tf.keras.layers.MaxPool2D()(y)
        y = self.c2(y)
        y = self.c3(y)
        y = self.flatten(y)
        #y = tf.reduce_sum(y, [1,2])
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [p.last_out for p in self.saliency_tracked_layers]
        return y

    def prune(self, metrics, kill_fraction = 0.1, kill_low = True, const_percent = False):
        assert(len(metrics) == len(self.saliency_tracked_layers))
        masks = []
        decision_points = []

        last_mask = None
        i = 0
        for dl, met in zip(self.saliency_tracked_layers, metrics):
            #if (i == 0 or i == 1 or i == 3 or i == 2):
            last_mask, decision_pt = dl.prune(met, last_mask, kill_fraction, kill_low, const_percent)
            '''else:
                dl.prune(None, last_mask, kill_fraction, kill_low, const_percent)
                last_mask = tf.constant([i for i in range(dl.units)])
                decision_pt = None'''
            masks.append(last_mask)
            decision_points.append(decision_pt)
            i += 1
            if i == self.num_convolutional_layers:
                last_mask = self.flatten.prune(None, last_mask)
        self.layers[-1].prune(None, last_mask, kill_fraction)
        return masks, decision_points

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

    def call(self, x):
        y = tf.keras.layers.MaxPool2D()(self.c1(x))
        y = tf.keras.layers.MaxPool2D()(self.proc(y))
        y = tf.reshape(y, [-1, y.shape[1] * y.shape[2] * y.shape[3]])
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [self.c1.last_out, self.proc.last_out, self.l1.last_out]
        return y