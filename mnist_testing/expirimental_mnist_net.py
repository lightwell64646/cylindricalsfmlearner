import tensorflow as tf
import sys
from utils import get_loss_categorical, parameter_count

sys.path.insert(0, "..")
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
        self.c1 = cylin.conv2d(250, [3, 3], stride=1, padding='VALID', name = "c1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.c2 = cylin.conv2d(250, [3, 3], stride=1, padding='VALID', name = "c2", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.rt = cylin.reverseAttention()

        self.last_outs = None
        self.saliency_tracked_layers = [self.c1, self.c2, self.proc, self.l1]

    def call(self, x):
        y = self.c1(x)
        y = tf.keras.layers.MaxPool2D()(y)
        img_processed = self.c2(y)
        
        img_processed = tf.concat(tf.meshgrid())
        y = self.rt(img_processed)       
        y = self.ut1(y, img_processed) 
        y = self.ut2(y, img_processed)
        y = self.t1(y)
        y = self.t2(y)
        self.last_outs = [p.last_out for p in self.saliency_tracked_layers]
        return y

    def prune(self, metrics, kill_fraction = 0.1, kill_low = True):
        assert(len(metrics) == len(self.saliency_tracked_layers))
        last_mask = None
        i = 0
        for dl, met in zip(self.saliency_tracked_layers, metrics):
            last_mask = dl.prune(met, last_mask, kill_fraction, kill_low)
            i += 1
            if i == self.num_convolutional_layers:
                last_mask = self.flatten.prune(None, last_mask, kill_fraction)
        self.layers[-1].prune(None, last_mask, kill_fraction)
