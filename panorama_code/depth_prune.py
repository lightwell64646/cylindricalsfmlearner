import os
import time
import tensorflow as tf

from panoramaDataLoader import get_panorama_datset
from compatible_depth_net import depth_net

import sys
sys.path.insert(0, "..")
from utils import projective_inverse_warp

class depth_prune_trainer(object):
    def __init__(self):
        # model variables
        self.depth_net = depth_net()
        self.metric_alpha = 0.9
        self.grad2 = [tf.zeros([l.units]) for l in self.depth_net.layers[:-1]]
        self.intrinsics = None

    def train(self, opt):
        loader = get_panorama_datset(opt.batch_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        logWriter = tf.summary.create_file_writer("./tmp/depthLogs.log")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.depth_net)
        manager = tf.train.CheckpointManager(
            checkpoint, directory="./tmp/depthModelCkpts", max_to_keep=5)
        with logWriter.as_default():
            step = 0
            for images in loader:
                # execute model
                with tf.GradientTape(persistent=True) as tape:
                    depths, poses = self.depth_net(images)
                    answer_loss = self.getDepthConsistencyLoss(images, depths, poses, self.intrinsics, opt)
                    regularization_loss = tf.reduce_sum(self.depth_net.losses)
                    loss = answer_loss + regularization_loss
                    grad1 = tape.gradient(loss, self.depth_net.last_outs)
                
                # train on loss
                grads = tape.gradient(loss, self.depth_net.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.depth_net.trainable_weights))

                # maintain pruning metrics
                g2 = tape.gradient(grad1, self.depth_net.last_outs)
                for i in range(len(g2)):
                    g2[i] = tf.reduce_mean(g2[i],[i for i in range(len(g2[i].shape) - 1)])
                    self.grad2[i] = self.grad2[i] * self.metric_alpha + g2[i] * (1 - self.metric_alpha)
                # Ensure that metrics are the right shape
                # print(self.grad2[0].shape,self.grad2[-1].shape)

                # maintain records.
                if (step % opt.summary_freq == 0):
                    tf.summary.experimental.set_step(step)
                    tf.summary.scalar("answer_loss", answer_loss)
                    tf.summary.scalar("regularization_loss", regularization_loss)
                    tf.summary.scalar("loss", loss)
                    print("training", answer_loss)
                    logWriter.flush()
                if ((step % opt.save_latest_freq) == 0): 
                    manager.save()
                step += 1
                if step >= opt.max_steps:
                    break

    def evaluate(self, opt):
        loader = get_panorama_datset(opt.batch_size, is_training=False)
        al_sum = rl_sum = test_cycles = 0
        for images in loader:
            depths, poses = self.depth_net(images)
            answer_loss = self.getDepthConsistencyLoss(images, depths, poses, self.intrinsics, opt)
            regularization_loss = tf.reduce_sum(self.depth_net.losses)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
        tf.summary.scalar("test answer_loss", al_sum / test_cycles)
        tf.summary.scalar("test regularization_loss", rl_sum / test_cycles)
        tf.summary.scalar("test loss", (rl_sum + al_sum) / test_cycles)
        print("test", (rl_sum + al_sum) / test_cycles)
        
    '''
    measure pose consistency of 
        images - [batch, sequence, scales, rows, colums, rgb]
        depths - [batch, sequence, scales, rows, columns, depth]
        poses - [batch, sequence, 6]
        intrinsics - [6]
    '''
    def getDepthConsistencyLoss(self, images, depths, poses, intrinsics, opt):
        depth_shape = depths.shape
        error = 0
        for i in range(1,depth_shape[1]):
            for j in range(depth_shape[2]):
                proj_image = projective_inverse_warp(
                    images[:,i,j],
                    depths[:,i,j],
                    poses[i],
                    intrinsics,
                    do_wrap=opt.do_wrap,
                    is_cylin=opt.cylindricalls
                )
                error += tf.reduce_mean(tf.abs(proj_image - images[:,0]))
        return error

    def prune(self):
        self.depth_net.prune(self.grad2)
        self.grad2 = [tf.zeros([l.units]) for l in self.depth_net.layers[:-1]]
        print("pruned", [g.shape for g in self.grad2])
        
