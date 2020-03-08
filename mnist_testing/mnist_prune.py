import os
import time
import tensorflow as tf
from mnistDataLoader import get_mnist_datset
from mnist_net import mnist_net

class mnist_prune_trainer(object):
    def __init__(self, opt):
        self.flags = opt
        self.optimizer = None
        self.epoch_accuracy = None

        self.mnist_net = mnist_net(opt)

        self.metric_alpha = 0.9
        self.grad2 = [tf.zeros([l.units]) for l in self.mnist_net.layers[:-1]]

    def getPruneMetric(self):
        return self.grad2

    def load_mnist_model(self, path):
        self.mnist_net = tf.keras.models.load_model(path)

    def get_loss(self, probabilities, labels):
        regularization_loss = tf.reduce_sum(self.mnist_net.losses)
        answer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels = labels, logits = probabilities))
        return answer_loss, regularization_loss

    def do_train(self, images, labels):
        with tf.GradientTape(persistent=True) as tape:
            probabilities = self.mnist_net(images)
            answer_loss, regularization_loss = self.get_loss(probabilities, labels)
            loss = answer_loss + regularization_loss
            self.epoch_accuracy.update_state(labels, probabilities)
            grad1 = tape.gradient(loss, self.mnist_net.last_outs)
        
        # train on loss
        grads = tape.gradient(loss, self.mnist_net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.mnist_net.trainable_weights))

        # maintain pruning metrics
        g2 = tape.gradient(grad1, self.mnist_net.last_outs)
        for i in range(len(g2)):
            g2[i] = tf.reduce_mean(g2[i],[i for i in range(len(g2[i].shape) - 1)])
            self.grad2[i] = self.grad2[i] * self.metric_alpha + g2[i] * (1 - self.metric_alpha)
        # Ensure that metrics are the right shape
        # print(self.grad2[0].shape,self.grad2[-1].shape)
        return answer_loss, regularization_loss

    def train(self):
        opt = self.flags
        if opt.init_checkpoint_file != None:
            self.load_mnist_model(opt.init_checkpoint_file)

        loader = get_mnist_datset(opt.batch_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()


        logWriter = tf.summary.create_file_writer("./tmp/mnistLogs.log")
        with logWriter.as_default():
            step = 0
            for images, labels in loader:
                # execute model
                if step == 0:
                    self.mnist_net._set_inputs(images)
                
                answer_loss, regularization_loss = self.do_train(images, labels)

                # maintain records.
                if (step % opt.summary_freq == 0):
                    acc = self.epoch_accuracy.result() 
                    self.epoch_accuracy.reset_states()
                    total_loss = answer_loss + regularization_loss

                    tf.summary.experimental.set_step(step)
                    tf.summary.scalar("answer_loss", tf.cast(answer_loss, tf.int64))
                    tf.summary.scalar("regularization_loss", regularization_loss)
                    tf.summary.scalar("loss", total_loss)
                    tf.summary.scalar("training accuracy", acc)
                    print("training", answer_loss, total_loss, acc)
                    logWriter.flush()
                if ((step % opt.save_latest_freq) == 0): 
                    self.mnist_net.save(opt.checkpoint_dir)
                step += 1
                print(step, opt.max_steps)
                if (step >= opt.max_steps):
                    break

    def save_explicit(self, path):
        self.mnist_net.save(path)

    def eval(self):
        opt = self.flags
        loader = get_mnist_datset(opt.batch_size, is_training=False)
        self.epoch_accuracy = tf.keras.metrics.Accuracy()
        al_sum = rl_sum = test_cycles = 0
        for image, labels in loader:
            probabilities = self.mnist_net(image)
            answer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels = labels, logits = probabilities))
            regularization_loss = tf.reduce_sum(self.mnist_net.losses)
            self.epoch_accuracy(labels, probabilities)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
        tf.summary.scalar("test accuracy", self.epoch_accuracy.result())
        tf.summary.scalar("test answer_loss", al_sum / test_cycles)
        tf.summary.scalar("test regularization_loss", rl_sum / test_cycles)
        tf.summary.scalar("test loss", (rl_sum + al_sum) / test_cycles)
        print("test", (rl_sum + al_sum) / test_cycles)
        

    def prune(self, kill_fraction = 0.1, save_path = None):
        if save_path == None:
            save_path = self.flags.checkpoint_dir + "prune/"
        self.mnist_net.prune(self.grad2, kill_fraction=kill_fraction)
        self.grad2 = [tf.zeros([l.units]) for l in self.mnist_net.layers[:-1]]
        checkpoint = tf.train.Checkpoint(model=self.mnist_net)
        checkpoint.save(save_path)
        print("pruned", [g.shape for g in self.grad2])
        
