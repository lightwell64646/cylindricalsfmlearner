import tensorflow as tf

def get_loss_categorical(probabilities, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
                        labels = labels, logits = probabilities)

def parameter_count(net):
    return tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in net.variables])