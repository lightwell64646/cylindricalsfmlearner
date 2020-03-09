import os
import time
import tensorflow as tf
from mnistDataLoader import get_mnist_datset
from mnist_net import mnist_net, get_loss_categorical

'''
This class runs a custom training loop that keeps an exponential
running average of prune metrics and provides utility to apply those
metrics to the underlying network. 

It also handles model saving and tensofboard logging.

Note: strategy scope must be opened while initializing values in 
mnist_net and when calling strategy.experimental_run_v2
'''
class mnist_prune_trainer_distributed(object):
    def __init__(self, opt):
        self.flags = opt
        self.strategy = tf.distribute.MirroredStrategy()

        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)
            self.mnist_net = mnist_net(opt)

        self.data_loader = self.strategy.experimental_distribute_dataset(get_mnist_datset(opt.batch_size))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.mnist_net)
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.global_batch_size = opt.batch_size * self.strategy.num_replicas_in_sync
        self.first = True
        self.metric_alpha = 0.99
        self.grad2 = [tf.zeros([l.units]) for l in self.mnist_net.prunable_layers]

    def getPruneMetric(self):
        return self.grad2
    def setPruneMetric(self, override):
        self.grad2 = override

    # Looses all methods not on base Keras model class if we use keras.model.save
    # syntax is a bit messier with checkpoints but it works
    def load_mnist_model(self, path):
        with self.strategy.scope():
            print("restored", self.checkpoint.restore(tf.train.latest_checkpoint(path)))
            for images, labels in self.data_loader:
                self.mnist_net._set_inputs(images)
                self.first = False
                break

    # sorry this is messy distributed training has funky syntax
    # first we define a tf.function that will handle every train step
    # then we run through data in a loop running that train step
    def train(self):
        opt = self.flags
        if opt.init_checkpoint_file != None:
            self.load_mnist_model(opt.init_checkpoint_file)

        @tf.function
        def function_wrapped_training(images, labels):
            def do_train(images, labels):
                with tf.GradientTape(persistent=True) as tape:
                    probabilities = self.mnist_net(images)
                    answer_loss, regularization_loss = get_loss_categorical(self.mnist_net, probabilities, labels, self.global_batch_size)
                    loss = answer_loss + regularization_loss
                    self.epoch_accuracy(labels, probabilities)
                    grad1 = tape.gradient(loss, self.mnist_net.last_outs)
                
                # train on loss
                grads = tape.gradient(loss, self.mnist_net.trainable_weights)
                self.optimizer.apply_gradients(list(zip(grads, self.mnist_net.trainable_weights)))

                # get second derivative for pruning
                g2 = tape.gradient(grad1, self.mnist_net.last_outs)
                # Ensure that derivatives are the right shape
                # print(self.grad2[0].shape,self.grad2[-1].shape)

                return answer_loss, regularization_loss, g2

            #######               ######
            # End per node Declaration #
            #######               ######

            Aloss, Rloss, g2 = self.strategy.experimental_run_v2(
                do_train, 
                args=(images, labels)
            )

            #combine results from multiple GPUs if necicary
            if (self.strategy.num_replicas_in_sync != 1):
                Aloss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, Aloss, axis=0)
                Rloss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, Rloss, axis=0)
                g2 = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, g2, axis=0)
            return Aloss, Rloss, g2

        #######                 ######
        # End Train Step Declaration #
        #######                 ######

        checkpointManager = tf.train.CheckpointManager(self.checkpoint, opt.checkpoint_dir, opt.max_checkpoints_to_keep)
        logWriter = tf.summary.create_file_writer("./tmp/mnistLogs.log")
        with logWriter.as_default():
            step = 0
            with self.strategy.scope():
                for images, labels in self.data_loader:
                    # execute model
                    if self.first:
                        self.mnist_net._set_inputs(images)
                    
                    answer_loss, regularization_loss, g2 = function_wrapped_training(images, labels)
                    for i in range(len(g2)):
                        g2[i] = tf.reduce_mean(g2[i],[i for i in range(len(g2[i].shape) - 1)])
                        if (self.first):
                            self.grad2[i] = g2[i]
                        else:
                            self.grad2[i] = self.grad2[i] * self.metric_alpha + g2[i] * (1 - self.metric_alpha)

                    if self.first:
                        self.first = False

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
                        checkpointManager.save()
                    step += 1
                    if (step >= opt.max_steps):
                        break


    def save_explicit(self, path):
        if (not os.path.isdir(path)):
            os.mkdir(path)
        print([[w.shape for w in l.get_weights()] for l in self.mnist_net.layers])
        self.checkpoint.save(path)

    def eval(self):
        @tf.function
        #start train step declaration
        def function_wrapped_testing(images, labels):

            #start per node declaration
            def do_test(images, labels):
                probabilities = self.mnist_net(images)
                answer_loss, regularization_loss = get_loss_categorical(self.mnist_net, probabilities, labels, self.global_batch_size)
                self.epoch_accuracy(labels, probabilities)

                return answer_loss, regularization_loss

            #######               ######
            # End per node Declaration #
            #######               ######

            Aloss, Rloss = self.strategy.experimental_run_v2(
                do_test, 
                args=(images, labels)
            )

            #combine results from multiple GPUs if necicary
            if (self.strategy.num_replicas_in_sync != 1):
                Aloss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, Aloss, axis=0)
                Rloss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, Rloss, axis=0)
            return Aloss, Rloss

        #######                 ######
        # End Test Step Declaration  #
        #######                 ######

        opt = self.flags
        loader = get_mnist_datset(opt.batch_size, is_training=False)
        al_sum = rl_sum = test_cycles = 0
        for image, labels in loader:
            answer_loss, regularization_loss = function_wrapped_testing(image, labels)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
        acc = self.epoch_accuracy.result()
        self.epoch_accuracy.reset_states()
        tf.summary.scalar("test accuracy", acc)
        tf.summary.scalar("test answer_loss", al_sum / test_cycles)
        tf.summary.scalar("test regularization_loss", rl_sum / test_cycles)
        tf.summary.scalar("test loss", (rl_sum + al_sum) / test_cycles)
        print("test", (rl_sum + al_sum) / test_cycles)
    
        return acc
        

    def prune(self, kill_fraction = 0.1, save_path = None):
        with self.strategy.scope():
            self.mnist_net.prune(self.grad2, kill_fraction=kill_fraction)

        self.grad2 = [tf.zeros([l.units]) for l in self.mnist_net.prunable_layers]
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.mnist_net)
        if (save_path != None):
            self.save_explicit(save_path)
        print("pruned", [g.shape for g in self.grad2])
        
