import tensorflow as tf

import sys
sys.path.insert(0, "..")
import tf_cylindrical as cylin
#definitions may not use all inputs, but standardization helps for framework

def grad2_saliency(loss, tape, net):
    with tape as tape2:
        grad1 = tape.gradient(loss, net.last_outs)
    return tape2.gradient(grad1, net.last_outs)

def grad1_saliency(loss, tape, net):
    return tape.gradient(loss, net.last_outs)

def activity_saliency(loss, tape, net):
    return [tf.reduce_mean(o, [i for i in range(len(o.shape)-1)])
        for o in net.last_outs]

def active_grad_saliency(loss, tape, net):
    g1s = grad1_saliency(loss, tape, net)
    a1s = activity_saliency(loss, tape, net)
    return [g*a for g,a in zip(g1s,a1s)]

def l2_saliency(loss, tape, net):
    weights = average_kernels(net.saliency_tracked_layers)
    return [w*w for w in weights]

def l1_saliency(loss, tape, net):
    weights = average_kernels(net.saliency_tracked_layers)
    return [tf.abs(w) for w in weights]

def average_kernels(layers):
    res = []
    for l in layers:
        k = l.saliency_weight
        if (type(l) is cylin.conv2dTranspose):
            res.append(tf.reduce_mean(k, [i for i in range(len(k.shape) -2)]+[len(k.shape)-1]))
        else:
            res.append(tf.reduce_mean(k, [i for i in range(len(k.shape)-1)]))
    return res