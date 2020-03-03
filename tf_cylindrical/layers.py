

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
        self.kernel = self.add_weight(name = "kernel",
                                      shape = self.kernel_size + [input_shape[-1], self.units],
                                      initializer = 'random_normal', 
                                      regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                      trainable = True)
        self.bias = self.add_weight(name = "bias",
                                    shape = [self.units],
                                    initializer = 'random_normal',
                                    trainable = True)

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
        self.w = self.add_weight(name = "weight", 
                                 shape = (input_shape[-1], self.units),
                                 initializer = 'random_normal',
                                 regularizer=tf.keras.regularizers.l2(self.L2Regularization),
                                 trainable = True)
        self.bias = self.add_weight(name = "b", shape = [self.units],
                                    initializer = 'random_normal',
                                    trainable = True)

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
            return output_mask
        return None

    def call(self, x):
        out = x @ self.w + self.bias
        if self.activation != None:
            out = self.activation(out)
        self.last_out = out
        return out


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

# -*- coding: utf-8 -*-

"""Convolutional layers for cylindrical data.

@@convolution2d
@@conv2d
"""
'''
import tensorflow as tf

from tf_cylindrical.pad import wrap, wrap_pad


class convolution2d:
    def __init__(num_outputs, kernel_size, stride=1, padding='CYLIN', **kwargs):
        # kernel size 1D -> 2D
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_outputs = num_outputs
        self.kwargs = **kwargs
        self.layer = tf.keras.layers.Conv2D(num_outputs,
                            kernel_size,
                            stride,
                            'VALID' if (padding == 'CYLIN') else padding,
                            **kwargs)

        print(padding)
        raise('Not a valid padding: {}'.format(padding))

    def prune(self, metric, input_mask = None, kill_fraction = 0.1):
        decision_point = np.sort(metric)[int(metric.shape[0]*kill_fraction)]
        output_mask = [i for m, i in enumerate(metric) if m > decision_point]
        if (input_mask != None): 
            kernel = tf.gather(self.layer.weights[0], input_mask, axis = 2)
        pruned_kernel = tf.gather(kernel, output_mask, axis = 3)
        pruned_bias = tf.gather(self.layer.weights[1], output_mask, axis = -1)
        if input_mask == None:
            input_size = [self.layers.weights[0].shape[2]]
        else:
            input_size = [len(input_mask)]

        self.layer = tf.keras.layers.Conv2D(metric.shape[0] - int(metric.shape[0]*kill_fraction)),
                            kernel_size,
                            stride,
                            self.padding,
                            self.kwargs)
        self.layer(tf.ones([1] + self.kernel_size + input_size))
        self.layer.set_weights([pruned_kernel, pruned_bias])


    def __call__(self, x):
        # maintain original behavior
        if self.padding=='SAME' or self.padding=='VALID':
            return self.layer(inputs)

        # W=(W−F+2P)/S+1
        elif self.padding=='CYLIN':
            size = inputs.get_shape()
            height = size[1]
            width = size[2]
            wrap_padding = [k-1 for k in kernel_size]
            wrapped_inputs = wrap_pad(inputs, wrap_padding)
            return self.layer(wrapped_inputs)



# Aliases
conv2d = convolution2d

self.layer.weights[0]
'''