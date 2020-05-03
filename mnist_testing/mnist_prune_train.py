import sys
sys.path.insert(0,'../prune_framework')

from prune_distributed import prune_trainer_distributed
from prune_single import prune_trainer
from multi_run_pruning import cultivate_model
from mnist_net import mnist_net
from mnist_data_loader import get_mnist_datset
from utils import get_loss_categorical

from absl import flags
from absl import app

import tensorflow as tf

# get command line inputs
flags.DEFINE_string("cultivation_report_path", "./reports/cultivationReport.csv", "the path to write the report on pruning success")
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("mask_path", "", "Path to mask image")

# So apparently keras can't handle path lengths of more than 170 characters in windows so ... yha. User be ware.
flags.DEFINE_string("checkpoint_dir", "./ckpts/", "Directory name to save the checkpoints")
flags.DEFINE_integer("max_checkpoints_to_keep", 3, "the number of checkpoints to store before deleting")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate of for adam")
flags.DEFINE_float("l2_weight_reg", 0.05, "Learning rate of for adam")
flags.DEFINE_float("smooth_weight", 0.2, "Weight for smoothness")
flags.DEFINE_integer("batch_size", 64, "The size of of a sample batch")
flags.DEFINE_integer("num_scales", 4, "Number of scales in multi-scale loss")
flags.DEFINE_integer("num_source", 2, "Number of source images")

# 1 so ignore effectively by default
flags.DEFINE_integer("target_parameter_count", 10000, "number of parameters in desired model")
flags.DEFINE_integer("max_prune_cycles", 5, "maximum number of prune cycles to run if target_parameter_count can not be met")
flags.DEFINE_integer("eval_steps", None, "number of batches to use for evaluation. (None means all in training set)")
flags.DEFINE_float("parameter_value_weighting", 0.002, "larger values favor smaller models when choosing which pruned model to propagate to next cycle. units are ((% acc)/(target_parameter_count parameters))")
flags.DEFINE_integer("prune_recovery_steps", 1000, "maximum number of training iterations to recover function after pruning network")
flags.DEFINE_integer("prune_recovery_log_count", 3, "number of prune recovery evaluations to perform durring recovery")
flags.DEFINE_integer("initial_training_steps", 10000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("do_wrap", True, "Enables horizontal wrapping")
flags.DEFINE_boolean("cylindrical", True, "Sets cylindrical projection")

#Don't touch this.
flags.DEFINE_boolean("do_accuracy", True, "whether or not to use accuracy as a metric.")
FLAGS = flags.FLAGS

def main(argv):
    #If I want to do distributed training set this for debug
    #tf.debugging.set_log_device_placement(True)

    cultivate_model(prune_trainer, mnist_net, get_mnist_datset, get_loss_categorical, FLAGS)

if __name__ == "__main__":
    app.run(main)