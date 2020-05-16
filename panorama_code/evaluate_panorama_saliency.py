import sys
import os
sys.path.insert(0,os.path.split(os.path.abspath(__file__))[0] + '/../prune_framework')

from prune_distributed import prune_trainer_distributed
from prune_single import prune_trainer
from evaluate_saliency import evaluate_saliency
from multi_run_pruning import cultivate_model

from compatible_depth_nets import depth_ego_net_compatible
from panoramaDataLoader import get_panorama_datset
from projective_loss import disparity_loss
from saliency_metrics import grad2_saliency, grad1_saliency, activity_saliency, l2_saliency, l1_saliency
from parse_intrinsics import parse_intrinsics

from absl import flags
from absl import app

import tensorflow as tf

# get command line inputs
flags.DEFINE_string("saliency_report_path", "reports/saliencyReport.csv", "the path to write the report on pruning success")
flags.DEFINE_string("dataset_dir", "../2018-10-03-subset", "Dataset directory")

# So apparently keras can't handle path lengths of more than 170 characters in windows so ... yha. User be ware.
flags.DEFINE_string("checkpoint_dir", "./ckpts/", "Directory name to save the checkpoints")
flags.DEFINE_integer("max_checkpoints_to_keep", 3, "the number of checkpoints to store before deleting")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")

flags.DEFINE_float("learning_rate", 0.0005, "Learning rate of for adam")
flags.DEFINE_float("l2_weight_reg", 0.05, "Learning rate of for adam")
flags.DEFINE_float("smooth_weight", 0.2, "Weight for smoothness")

flags.DEFINE_integer("batch_size", 2, "The size of of a sample batch")
flags.DEFINE_integer("num_scales", 4, "Number of scales in multi-scale loss")
flags.DEFINE_integer("num_source", 2, "Number of source images")

flags.DEFINE_string("intrinsics_file", "../data/headcam/intrinsics.txt", "path to file containing intrinsics for camera")

flags.DEFINE_integer("target_parameter_count", 10000, "number of parameters in desired model")
flags.DEFINE_integer("max_prune_cycles", 5, "maximum number of prune cycles to run if target_parameter_count can not be met")
flags.DEFINE_integer("eval_steps", None, "number of batches to use for evaluation. (None means all in training set)")
flags.DEFINE_float("parameter_value_weighting", 0.002, "larger values favor smaller models when choosing which pruned model to propagate to next cycle. units are ((% acc)/(target_parameter_count parameters))")
flags.DEFINE_integer("num_prunes", 10, "number of pruning steps to run")
flags.DEFINE_float("prune_rate", 0.01, "percentage of neurons to kill in a step")
flags.DEFINE_integer("initial_steps", 100, "Maximum number of training iterations to start") # if zero pruning will fail. More will give a more accurate pruning.
flags.DEFINE_integer("repair_steps", 10, "Maximum number of training iterations to repair after prune") # if zero pruning will fail. More will give a more accurate pruning.
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("do_wrap", True, "Enables horizontal wrapping")
flags.DEFINE_boolean("cylindrical", True, "Sets cylindrical projection")
flags.DEFINE_boolean("use_tpu", False, "whether to use tpu")

#do not set manually. Yes its jank just don't
flags.DEFINE_boolean("intrinsics", True, "data holder for use internally")
flags.DEFINE_boolean("do_accuracy", False, "whether or not to use accuracy as a metric. Meaningless for self supervised approaches")

FLAGS = flags.FLAGS

def main(argv):
    #If I want to do distributed training so this is set for debug
    #tf.debugging.set_log_device_placement(True)
    intrinsics = parse_intrinsics(FLAGS.intrinsics_file, FLAGS.num_scales)
    multi_intrinsics = tf.stack([intrinsics for _ in range(FLAGS.batch_size)], axis = 0)
    FLAGS.intrinsics = multi_intrinsics
    evaluate_saliency(prune_trainer, depth_ego_net_compatible, get_panorama_datset, 
            disparity_loss, grad1_saliency, FLAGS)

if __name__ == "__main__":
    app.run(main)