import tensorflow as tf
import sys
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
        self.proc = cylin.conv2d(250, [3, 3], stride=1, padding='VALID', name = "proc", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.num_convolutional_layers = 3

        self.flatten = cylin.flatten(name = "flattening")
        self.l1 = cylin.linear(500, name = "l1", 
            activation = tf.nn.relu, L2Regularization=opt.l2_weight_reg)
        self.l2 = cylin.linear(10, name = "l2", L2Regularization=opt.l2_weight_reg)

        self.last_outs = None
        self.prunable_layers = [self.c1, self.c2, self.proc, self.l1]

    def call(self, x):
        y = tf.keras.layers.MaxPool2D()(self.c1(x))
        y = self.c2(y)
        y = self.proc(y)
        y = self.proc(y)
        y = self.proc(y)
        y = self.flatten(y)
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [p.last_out for p in self.prunable_layers]
        return y

    def prune(self, metrics, kill_fraction = 0.1):
        assert(len(metrics) == len(self.prunable_layers))
        last_mask = None
        metrics.insert(self.num_convolutional_layers, None) #for flatten
        for dl, met in zip(self.layers[:-1], metrics):
            last_mask = dl.prune(met, last_mask, kill_fraction)
        self.layers[-1].prune(None, last_mask, kill_fraction)


def get_loss_categorical(net, probabilities, labels, global_batch_size):
    regularization_loss = tf.nn.scale_regularization_loss(net.losses)
    answer_loss = tf.nn.compute_average_loss(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels = labels, logits = probabilities), 
                    global_batch_size)
    return answer_loss, regularization_loss

def parameter_count(net):
    return tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in net.variables])

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
        #y = self.c2(y)
        y = tf.keras.layers.MaxPool2D()(self.proc(y))
        #y = self.proc(y)
        #y = self.proc(y)
        y = tf.reshape(y, [-1, y.shape[1] * y.shape[2] * y.shape[3]])
        y = self.l1(y)
        y = self.l2(y)
        self.last_outs = [self.c1.last_out, self.proc.last_out, self.l1.last_out]
        return y
