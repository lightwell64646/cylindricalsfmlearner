from keras.datasets import mnist
import tensorflow as tf

def read_and_preprocess_mnist(features, labels):
    num_classes = 10

    features = tf.cast(tf.reshape(features, [28,28]), tf.float32) / 255
    features = tf.expand_dims(features, axis=-1)
    labels = tf.cast(tf.one_hot(labels,num_classes), tf.int32)
    return features, labels

# set the batch dimension to a constant size. This makes TPUs happy.
def _set_shapes(batch_size, features, labels, transpose_input = False, extra_dims = 0):
    """Statically set the batch_size dimension."""
    """needed to make batch_size read as a fixed value to make TPUs happy"""
    
    def _set_shape_transpose(x, batch_size = batch_size):
        x.set_shape(x.get_shape().merge_with(tf.TensorShape([None] * extra_dims + [batch_size])))
        
    def _set_shape(x, extra_dims = 0, batch_size = batch_size):
        x.set_shape(x.get_shape().merge_with(tf.TensorShape([batch_size] + [None] * extra_dims)))

    if transpose_input:
        _set_shape_transpose(features, 3)
        _set_shape_transpose(labels)
    else:
        _set_shape(features, 3)
        _set_shape(labels)
    return features, labels

def get_mnist_datset(batch_size, is_training = True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(128)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # augment and batch
    dataset = dataset.map(read_and_preprocess_mnist)

    # note the batch diminsion is assumed in this syntax
    dataset = dataset.padded_batch(batch_size=batch_size, 
                    padded_shapes = ([28,28,1],[10]),
            drop_remainder=True)

    # assign static shape. Needed only for TPU training.
    # dataset = dataset.map(
    #     functools.partial(_set_shapes, batch_size)
    # )

    # prefetch data while training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset