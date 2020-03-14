

# -*- coding: utf-8 -*-

"""Convolutional layers for cylindrical data.

@@convolution2d
@@conv2d
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from tf_cylindrical.pad import wrap, wrap_pad


class convolution2d(Layer):
    def __init__(self, units, kernel_size, stride=1, padding='CYLIN', L2Regularization = 0.05, activation = None, **kwargs):
        super(convolution2d, self).__init__(**kwargs)
        # kernel size 1D -> 2D
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.units = units
        self.kernel_size = kernel_size
        self.stride = stride
        self.L2Regularization = L2Regularization
        self.activation = activation
        self.padding = padding
        
        self.last_out = None

    def build(self, input_shape):
        super(convolution2d, self).build(input_shape)
        self.kernel = self.add_weight(name = "kernel",
                                      shape = self.kernel_size + [input_shape[-1], self.units],
                                      initializer = 'random_normal', 
                                      regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                      trainable = True)
        self.bias = self.add_weight(name = "bias",
                                    shape = [self.units],
                                    initializer = 'random_normal',
                                    trainable = True)

        print("built", [w.shape for w in self.get_weights()])

    def call(self, x):
        # maintain original behavior
        if self.padding=='SAME' or self.padding=='VALID':
            result = tf.nn.conv2d(x, self.kernel, self.stride, self.padding) + self.bias

        # W=(W−F+2P)/S+1
        elif self.padding=='CYLIN':
            size = x.get_shape()
            height = size[1]
            width = size[2]
            wrap_padding = [k-1 for k in self.kernel_size]
            wrapped_inputs = wrap_pad(x, wrap_padding)
            result = tf.nn.conv2d(wrapped_inputs, self.kernel, self.stride, 'VALID') + self.bias
        else: 
            raise('Not a valid padding: {}'.format(self.padding))
        
        if self.activation != None:
            result = self.activation(result)
        self.last_out = result
        return result

    def prune(self, metric, input_mask = None, kill_fraction = 0.1):
        if (input_mask is not None): 
            pruned_kernel = tf.gather(self.kernel, input_mask, axis = 2)
            self.kernel = self.add_weight(name = "weight", shape = pruned_kernel.shape,
                                     initializer = tf.constant_initializer(pruned_kernel.numpy()),
                                     regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                     trainable = True)
        if metric is not None:
            decision_point = np.sort(metric)[int(metric.shape[0]*kill_fraction)]
            output_mask = [i for i, m in enumerate(metric) if m > decision_point]
            output_mask = tf.constant(output_mask, dtype = tf.int32)
            pruned_kernel = tf.gather(self.kernel, output_mask, axis = 3)
            pruned_bias = tf.gather(self.bias, output_mask)
            self.kernel = self.add_weight(name = "weight", shape = pruned_kernel.shape,
                                     initializer = tf.constant_initializer(pruned_kernel.numpy()),
                                     regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                     trainable = True)
            self.bias = self.add_weight(name = "b", shape = pruned_bias.shape,
                                     initializer = tf.constant_initializer(pruned_bias.numpy()),
                                     trainable = True)
            self.units = len(output_mask)
            return output_mask
        return None


# Aliases
conv2d = convolution2d

class linear(Layer):
    def __init__(self, units, L2Regularization = 0.05, activation = None, **kwargs):
        super(linear, self).__init__(**kwargs)
        self.units = units
        self.L2Regularization = L2Regularization
        self.activation = activation

        self.last_out = None

    def build(self, input_shape):
        super(linear, self).build(input_shape)
        self.w = self.add_weight(name = "weight", 
                                 shape = (input_shape[-1], self.units),
                                 initializer = 'random_normal',
                                 regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                 trainable = True)
        self.bias = self.add_weight(name = "b", shape = [self.units],
                                    initializer = 'random_normal',
                                    trainable = True)

    def call(self, x):
        out = x @ self.w + self.bias
        if self.activation != None:
            out = self.activation(out)
        self.last_out = out
        return out
    def prune(self, metric, input_mask = None, kill_fraction = 0.1):
        if (input_mask is not None): 
            pruned_w = tf.gather(self.w, input_mask, axis = 0)
            self.w = self.add_weight(name = "weight", shape = pruned_w.shape,
                                        initializer = tf.constant_initializer(pruned_w.numpy()),
                                        regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                        trainable = True)
        if metric is not None:
            decision_point = np.sort(metric)[int(metric.shape[0]*kill_fraction)]
            output_mask = [i for i, m in enumerate(metric) if m > decision_point]
            output_mask = tf.constant(output_mask, dtype = tf.int32)
            pruned_w = tf.gather(self.w, output_mask, axis = 1)
            pruned_bias = tf.gather(self.bias, output_mask)
            self.w = self.add_weight(name = "weight", shape = pruned_w.shape,
                                        initializer = tf.constant_initializer(pruned_w.numpy()),
                                        regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                        trainable = True)
            self.bias = self.add_weight(name = "b", shape = pruned_bias.shape,
                                        initializer = tf.constant_initializer(pruned_bias.numpy()),
                                        trainable = True)
            self.units = len(output_mask)
            #print("prune linear output_mask", output_mask.shape, metric.shape, input_mask.shape)
            return output_mask
        return None

'''
tiles the pruning metrics to ensure clean transition
'''
class flatten(Layer):
    def __init__(self, **kwargs):
        super(flatten, self).__init__(**kwargs)
        self.mirrored_inputs = 1
        self.units = 1
    def build(self, input_shape):
        super(flatten, self).build(input_shape)
        self.mirrored_inputs = tf.math.reduce_prod(input_shape[1:-1])
        self.units = tf.reduce_sum(input_shape[-1])
    def call(self, x):
        return tf.reshape(x, [-1, self.mirrored_inputs*self.units])

    def prune(self, empty, input_mask = None, kill_fraction = 0.1):
        self.units = tf.reduce_sum(input_mask.shape)
        print("flatten shortened to ", input_mask.shape)
        res = [input_mask + i*self.units for i in range(self.mirrored_inputs)]
        res = tf.concat(res,0)
        return res

class resnet(Layer):
    def __init__(self, units, layer = conv2d, depth = 2, **kwargs):
        super(resnet, self).__init__(**kwargs)
        self.layer = layer
        self.units = units
        self.layers = [layer(units = units, **kwargs) for i in range(depth)]
        self.depth = depth

        self.last_out = None

    def call(self, x):
        out = x
        for l in self.layers:
            out = out + l(out)
        self.last_out = out
        return out

    def prune(self, metric, input_mask = None, kill_fraction = 0.1):
        for l in self.layers:
            input_mask = l.prune(metric, input_mask)
        self.units = len(input_mask)
        return input_mask


class step_stop(Layer):
    def __init__(self, features, pad = 'CYLIN', width = 3, stride = 2, **kwargs):
        super(step_stop, self).__init__()
        self.step = conv2d(32,  [width, width], 
            stride=stride, padding=pad, **kwargs)
        self.stop = conv2d(32,  [width, width], 
            stride=1, padding=pad, **kwargs)
        self.layers = [self.step, self.stop]
    def __call__(self, x):
        return self.stop(self.step(x))

    '''def prune(self, metric, input_mask = None, kill_fraction = 0.1):
        # since the output of the intermediate isn't reported to the top layer of the module, this can't be done well. Suggest not useing
        for l in self.layers[:-1]:
            l.prune(None, input_mask)
        self.layers[-1].prune(metric, None)
        return output_mask'''
