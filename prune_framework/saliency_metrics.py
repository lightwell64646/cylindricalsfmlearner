import tensorflow as tf
#definitions may not use all inputs, but standardization helps for framework

def grad2_saliency(loss, tape, net):
    with tf.GradientTape() as tapeScope:
        grad1 = tape.gradient(loss, net.last_outs)
    return tapeScope.gradient(grad1, net.last_outs)

def grad1_saliency(loss, tape, net):
    return tape.gradient(loss, net.last_outs)

def activity_saliency(loss, tape, net):
    return average_kernels([o for o in net.last_outs])

def l2_saliency(loss, tape, net):
    weights = average_kernels([l.saliency_weight for l in net.saliency_tracked_layers])
    return [w*w for w in weights]

def l1_saliency(loss, tape, net):
    weights = average_kernels([l.saliency_weight for l in net.saliency_tracked_layers])
    return [tf.abs(w) for w in weights]

def average_kernels(kernels):
    return [tf.reduce_mean(k, [i for i in range(len(k.shape)-1)]) for k in kernels]