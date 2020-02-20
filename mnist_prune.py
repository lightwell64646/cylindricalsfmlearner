import os
import time
import tensorflow as tf
from mnistDataLoader import get_mnist_datset
from mnist_net import mnist_net

class mnist_prune_trainer(object):
    def __init__(self):
        # model variables
        self.mnist_net = mnist_net()
        self.metric_alpha = 0.9
        self.grad2 = [tf.zeros([l.units]) for l in self.mnist_net.layers[:-1]]

    def train(self, opt):
        loader = get_mnist_datset(opt.batch_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        logWriter = tf.summary.create_file_writer("./tmp/mnistLogs.log")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.mnist_net)
        manager = tf.train.CheckpointManager(
            checkpoint, directory="./tmp/mnistModelCkpts", max_to_keep=5)
        with logWriter.as_default():
            step = 0
            epoch_accuracy = tf.keras.metrics.Accuracy() 
            for image, labels in loader:
                # execute model
                with tf.GradientTape(persistent=True) as tape:
                    probabilities = self.mnist_net(image)
                    answer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels = labels, logits = probabilities))
                    regularization_loss = tf.reduce_sum(self.mnist_net.losses)
                    loss = answer_loss + regularization_loss
                    epoch_accuracy(labels, probabilities)
                    grad1 = tape.gradient(loss, self.mnist_net.last_outs)
                
                # train on loss
                grads = tape.gradient(loss, self.mnist_net.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.mnist_net.trainable_weights))

                # maintain pruning metrics
                g2 = tape.gradient(grad1, self.mnist_net.last_outs)
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
                    tf.summary.scalar("training accuracy", epoch_accuracy.result())
                    print("training", answer_loss)
                    logWriter.flush()
                if ((step % opt.save_latest_freq) == 0): 
                    manager.save()
                step += 1
                if step >= opt.max_steps:
                    break

    def evaluate(self, opt):
        loader = get_mnist_datset(opt.batch_size, is_training=False)
        test_accuracy = tf.keras.metrics.Accuracy()
        al_sum = rl_sum = test_cycles = 0
        for image, labels in loader:
            probabilities = self.mnist_net(image)
            answer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels = labels, logits = probabilities))
            regularization_loss = tf.reduce_sum(self.mnist_net.losses)
            test_accuracy(labels, probabilities)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
        tf.summary.scalar("test accuracy", test_accuracy.result())
        tf.summary.scalar("test answer_loss", al_sum / test_cycles)
        tf.summary.scalar("test regularization_loss", rl_sum / test_cycles)
        tf.summary.scalar("test loss", (rl_sum + al_sum) / test_cycles)
        print("test", (rl_sum + al_sum) / test_cycles)
        

    def prune(self):
        self.mnist_net.prune(self.grad2)
        self.grad2 = [tf.zeros([l.units]) for l in self.mnist_net.layers[:-1]]
        print("pruned", [g.shape for g in self.grad2])
        
