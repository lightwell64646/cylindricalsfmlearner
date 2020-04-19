import tensorflow as tf
import os

'''
TODO: I don't understand the data format being read these constants are probably wrong
'''

IMG_HEIGHT = 1280
IMG_WIDTH = 2560
NUM_SOURCE = 3
SOURCE_SPACING = 1

# el-capitan has images in //data/jventu09/headcam/2018-10-03

def get_file_list(data_root, split):
    data_path = data_root+'/'+split
    frames = [os.path.join(data_path,f) for f in os.listdir(data_path)]
    frames = frames[:len(frames)-(len(frames)%NUM_SOURCE)]
    image_stacks = [[frames[i], frames[i+SOURCE_SPACING], frames[i+SOURCE_SPACING*2]] 
                for i in range(0,int(len(frames)/3), 3)]
    print(image_stacks)
    return image_stacks

def read_and_preprocess_panorama(image_stack):
    print(image_stack)
    src_tgt_seq = [tf.image.decode_image(
                    tf.io.read_file(image_stack[i]))
                for i in range(NUM_SOURCE)]
    src_tgt_seq = tf.concat(src_tgt_seq, axis = -1)
    return src_tgt_seq, src_tgt_seq

def get_panorama_datset(flags, is_training = True):
    dataset_dir = flags.dataset_dir
    batch_size = flags.batch_size
    dataset = tf.data.Dataset.from_tensor_slices(
        get_file_list(dataset_dir, 'train' if is_training else "eval"))
    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(128)

    # augment and batch
    dataset = dataset.map(read_and_preprocess_panorama)

    dataset = dataset.batch(batch_size=batch_size,
            drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset