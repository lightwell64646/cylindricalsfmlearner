from mnist_prune_distributed import mnist_prune_trainer_distributed
from mnist_prune import mnist_prune_trainer
from multi_run_pruning import cultivate_model
from absl import flags
from absl import app

import tensorflow as tf

# get command line inputs
flags.DEFINE_string("cultivation_report_path", "./", "the path to write the report on pruning success")
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("mask_path", "", "Path to mask image")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate of for adam")
flags.DEFINE_float("l2_weight_reg", 0.05, "Learning rate of for adam")
flags.DEFINE_float("smooth_weight", 0.2, "Weight for smoothness")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("num_scales", 4, "Number of scales in multi-scale loss")
flags.DEFINE_integer("num_source", 2, "Number of source images")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("do_wrap", True, "Enables horizontal wrapping")
flags.DEFINE_boolean("cylindrical", True, "Sets cylindrical projection")
FLAGS = flags.FLAGS

def main(argv):
    #I want to do distributed training so this is set for debug
    tf.debugging.set_log_device_placement(True)

    cultivate_model(mnist_prune_trainer_distributed, FLAGS)

if __name__ == "__main__":
    app.run(main)