import sys
import os
sys.path.insert(0,os.path.split(os.path.abspath(__file__))[0] + '/../prune_framework')

from prune_distributed import prune_trainer_distributed
from prune_single import prune_trainer
from evaluate_saliency import evaluate_saliency
from mnist_net import mnist_net_prune_fin_shape
from mnist_data_loader import get_mnist_datset
from saliency_metrics import grad2_saliency, grad1_saliency, activity_saliency, active_grad_saliency, l2_saliency, l1_saliency
from utils import get_loss_categorical

from absl import flags
from absl import app

import copy

import tensorflow as tf

# get command line inputs
flags.DEFINE_string("saliency_report_path", "reports/saliencyReport", "the path to write the report on pruning success")
flags.DEFINE_string("run_name", "Repaired Pruning", "the path to write the report on pruning success")
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("mask_path", "", "Path to mask image")

# So apparently keras can't handle path lengths of more than 170 characters in windows so ... yha. User be ware.
flags.DEFINE_string("checkpoint_dir", "./ckpts/", "Directory name to save the checkpoints")
flags.DEFINE_integer("max_checkpoints_to_keep", 3, "the number of checkpoints to store before deleting")
flags.DEFINE_string("init_checkpoint_file", "grad1saliency_curvesaliency_base", "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate of for adam")
flags.DEFINE_float("l2_weight_reg", 0.05, "Learning rate of for adam")
flags.DEFINE_float("smooth_weight", 0.2, "Weight for smoothness")
flags.DEFINE_integer("batch_size", 64, "The size of of a sample batch")
flags.DEFINE_integer("num_scales", 4, "Number of scales in multi-scale loss")
flags.DEFINE_integer("num_source", 2, "Number of source images")

# 1 so ignore effectively by default
flags.DEFINE_integer("target_parameter_count", 10000, "number of parameters in desired model")
flags.DEFINE_integer("eval_steps", None, "number of batches to use for evaluation. (None means all in training set)")
flags.DEFINE_float("parameter_value_weighting", 0.002, "larger values favor smaller models when choosing which pruned model to propagate to next cycle. units are ((% acc)/(target_parameter_count parameters))")
flags.DEFINE_integer("num_prunes", 10, "number of pruning steps to run")
flags.DEFINE_float("prune_rate", 0.08, "percentage of neurons to kill in a step")
flags.DEFINE_integer("initial_steps", 50, "Maximum number of training iterations to start") # if zero pruning will fail. More will give a more accurate pruning.
flags.DEFINE_integer("repair_steps", 500, "Maximum number of training iterations to repair after prune") # if zero pruning will fail. More will give a more accurate pruning.


flags.DEFINE_integer("training_log_resolution", 7, "number of subdivisions of training to log")

flags.DEFINE_boolean("show_plots", False, "whether or not to use accuracy as a metric. Always used for MNIST supervised")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("do_wrap", True, "Enables horizontal wrapping")
flags.DEFINE_boolean("cylindrical", True, "Sets cylindrical projection")
flags.DEFINE_boolean("use_tpu", False, "whether to use tpu")
flags.DEFINE_boolean("do_accuracy", True, "whether or not to use accuracy as a metric. Always used for MNIST supervised")
FLAGS = flags.FLAGS


def log_training(net, steps, Flags):
    log = [net.eval(Flags.eval_steps)]
    for _ in range(Flags.training_log_resolution):
        net.train(steps/Flags.training_log_resolution)
        log.append(net.eval(Flags.eval_steps))
    return log

def main(argv):
    #If I want to do distributed training so this is set for debug
    #tf.debugging.set_log_device_placement(True)

    net = prune_trainer(FLAGS, mnist_net_prune_fin_shape, get_mnist_datset, get_loss_categorical, grad1_saliency)
    print(log_training(net, 10000, FLAGS))

if __name__ == "__main__":
    app.run(main)
