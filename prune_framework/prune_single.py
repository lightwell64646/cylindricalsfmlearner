import os
import time
import tensorflow as tf
import numpy as np

'''
A single GPU version of prune_trainer_distributed

TODO: do object oriented stuff to reduce duplication
TODO: This code does not work? Testing
'''
class prune_trainer(object):
    def __init__(self, opt, net_generator, dataset_generator, loss_function, do_accuracy = True):
        self.flags = opt
        self.do_accuracy = do_accuracy
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)

        self.net_generator = net_generator
        self.dataset_generator = dataset_generator
        self.loss_function = loss_function

        self.data_loader = dataset_generator(opt)
        self.net = net_generator(opt)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)

        self.first = True
        self.metric_alpha = 0.99
        self.grad2 = [tf.zeros([l.units]) for l in self.net.saliency_tracked_layers]

    def clone_prune_state(self, other):
        self.grad2 = other.grad2
        for ls, lo in zip(self.net.saliency_tracked_layers, other.net.saliency_tracked_layers):
            ls.clone_prune_state(lo)

    def getPruneMetric(self):
        return self.grad2
    def setPruneMetric(self, override):
        self.grad2 = override

    def load_model(self, path):
        print("restored", path)
        self.checkpoint.restore(tf.train.latest_checkpoint(path))
        for images, labels in self.data_loader:
            self.net._set_inputs(images)
            break

    def do_train(self, images, labels):
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.net(images)
            answer_loss = tf.nn.compute_average_loss(
                self.loss_function(predictions, labels), self.flags.batch_size)
            regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
            loss = answer_loss + regularization_loss
            if self.do_accuracy:
                self.epoch_accuracy(labels, predictions)
            grad1 = tape.gradient(loss, self.net.last_outs)
        
        # train on loss
        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(list(zip(grads, self.net.trainable_weights)))

        # get second derivative for pruning
        g2 = []
        grads_per_neuron = tape.gradient(grad1, self.net.last_outs)
        for g in grads_per_neuron:
            collapse_dims = tf.range(len(g.shape._dims) - 1)
            g2.append(tf.reduce_mean(tf.abs(g), collapse_dims))
        # Ensure that derivatives are the right shape
        # print(self.grad2[0].shape,self.grad2[-1].shape)
        return answer_loss, regularization_loss, g2

    def train(self, max_steps = None, verbose = False):
        opt = self.flags

        checkpointManager = tf.train.CheckpointManager(self.checkpoint, opt.checkpoint_dir, opt.max_checkpoints_to_keep)
        logWriter = tf.summary.create_file_writer("./tmp/trainingLogs.log")
        with logWriter.as_default():
            self.manage_training(self.do_train, checkpointManager, logWriter, max_steps)
            

    def manage_training(self, train_step_func, checkpointManager, logWriter, max_steps):
        opt = self.flags
        step = 0
        for images, labels in self.data_loader:
            # execute model
            if self.first:
                try:
                    self.net._set_inputs(images)
                except:
                    print("tried to double set inputs. Oops")
            
            answer_loss, regularization_loss, g2 = train_step_func(images, labels)
            for i in range(len(g2)):
                if (self.first):
                    self.grad2[i] = g2[i]
                else:
                    self.grad2[i] = self.grad2[i] * self.metric_alpha + g2[i] * (1 - self.metric_alpha)

            if self.first:
                self.first = False

            # maintain records.
            if (step % opt.summary_freq == 0):
                total_loss = answer_loss + regularization_loss

                tf.summary.experimental.set_step(step)
                tf.summary.scalar("answer_loss", tf.cast(answer_loss, tf.int64))
                tf.summary.scalar("regularization_loss", regularization_loss)
                tf.summary.scalar("loss", total_loss)

                if self.do_accuracy:
                    acc = self.epoch_accuracy.result() 
                    self.epoch_accuracy.reset_states()
                    tf.summary.scalar("training accuracy", acc)
                    print("training", answer_loss, total_loss, acc)
                else:
                    print("training", answer_loss, total_loss)

                logWriter.flush()
            if ((step % opt.save_latest_freq) == 0):
                checkpointManager.save()
            step += 1
            if (step >= max_steps):
                break

    def save_explicit(self, path):
        if (not os.path.isdir(path)):
            os.mkdir(path)
        print("saving", [[w.shape for w in l.get_weights()] for l in self.net.layers])
        self.checkpoint.save(path)

    
    def do_test(self, images, labels):
        predictions = self.net(images)
        answer_loss = tf.nn.compute_average_loss(
            self.loss_function(predictions, labels), self.flags.batch_size)
        regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
        if self.do_accuracy:
            self.epoch_accuracy(labels, predictions)

        return answer_loss, regularization_loss

    def eval(self, eval_steps = None, verbose = False):

        opt = self.flags
        loader = self.dataset_generator(opt.batch_size, is_training=False)
        al_sum = rl_sum = test_cycles = 0
        for image, labels in loader:
            answer_loss, regularization_loss = self.do_test(image, labels)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
            if (eval_steps != None and eval_steps <= test_cycles):
                break

            
        average_anser_loss = al_sum / test_cycles
        average_regularization_loss = rl_sum / test_cycles
        tf.summary.scalar("test answer_loss", average_anser_loss)
        tf.summary.scalar("test regularization_loss", average_regularization_loss)
        average_anser_loss = float(average_anser_loss)
        average_regularization_loss = float(average_regularization_loss)
        if self.do_accuracy:
            acc = float(self.epoch_accuracy.result())
            self.epoch_accuracy.reset_states()
            tf.summary.scalar("test accuracy", acc)
            if verbose:
                print("test", average_anser_loss, average_regularization_loss, acc)
            return acc
        
        return average_anser_loss
        

    def prune(self, kill_fraction = 0.1, save_path = None, verbose = True):
        self.net.prune(self.grad2, kill_fraction=kill_fraction)
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)
        if (save_path != None):
            self.save_explicit(save_path)
            
        prune_breaks = [np.sort(metric)[int(metric.shape[0]*kill_fraction)] for metric in self.grad2]
        self.grad2 = [tf.zeros([l.units]) for l in self.net.saliency_tracked_layers]
        if verbose:
            print("pruned to", [g.shape for g in self.grad2])

        return prune_breaks
        