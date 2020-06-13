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
    def __init__(self, opt, net_generator, dataset_generator, loss_function, saliency_function):
        self.flags = opt
        self.do_accuracy = self.flags.do_accuracy
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)
        self.saliency_function = saliency_function

        self.net_generator = net_generator
        self.dataset_generator = dataset_generator
        self.loss_function = loss_function

        self.data_loader = dataset_generator(opt)
        self.net = net_generator(opt)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)

        self.first = True
        self.metric_alpha = 0.99
        self.saliency = [tf.zeros([l.units]) for l in self.net.saliency_tracked_layers]

    def clone_prune_state(self, other):
        self.saliency = other.saliency
        for ls, lo in zip(self.net.saliency_tracked_layers, other.net.saliency_tracked_layers):
            ls.clone_prune_state(lo)
    def reset_prune_state(self):
        self.saliency = [tf.zeros([l.units]) for l in self.net.saliency_tracked_layers]
        for ls in self.net.saliency_tracked_layers:
            ls.reset_prune_state()
    def getPruneMetric(self):
        return self.saliency
    def setPruneMetric(self, override):
        self.saliency = override

    def load_model(self, path):
        print("restored", path)
        self.net.load_weights(path + ".ckpt")
        #self.checkpoint.restore(path).assert_existing_objects_matched()
        print(len(self.net.layers[0].variables), end = ", ")
        for images, labels in self.data_loader:
            self.net._set_inputs(images)
            break
        print(len(self.net.layers[0].variables))

    def do_train(self, images, labels):
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.net(images)
            answer_loss = self.loss_function(predictions, labels, self.flags)
            regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
            loss = answer_loss + regularization_loss
            if self.do_accuracy:
                self.epoch_accuracy(labels, predictions)
            #grad1 = tape.gradient(loss, self.net.last_outs)
        
        # train on loss
        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(list(zip(grads, self.net.trainable_weights)))

        # get second derivative for pruning
        g2 = []
        #grads_per_neuron = tape.gradient(grad1, self.net.last_outs)
        grads_per_neuron = self.saliency_function(loss, tape, self.net)
        for g in grads_per_neuron:
            collapse_dims = tf.range(len(g.shape._dims) - 1)
            g2.append(tf.reduce_mean(tf.abs(g), collapse_dims))
        # Ensure that derivatives are the right shape
        # print(self.saliency[0].shape,self.saliency[-1].shape)
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
                    self.saliency[i] = g2[i]
                else:
                    self.saliency[i] = self.saliency[i] * self.metric_alpha + g2[i] * (1 - self.metric_alpha)

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
        print("saving", [[w.shape for w in l.get_weights()] for l in self.net.layers])
        print(path)
        #print(self.checkpoint.save(path))

        self.net.save_weights(path + ".ckpt")

    
    def do_test(self, images, labels):
        predictions = self.net(images)
        answer_loss = self.loss_function(predictions, labels, self.flags) / self.flags.batch_size
        regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
        if self.do_accuracy:
            self.epoch_accuracy(labels, predictions)

        return answer_loss, regularization_loss

    def eval(self, eval_steps = None, verbose = False):

        opt = self.flags
        loader = self.dataset_generator(opt, is_training=False)
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
        

    def prune(self, kill_fraction = 0.1, save_path = None, verbose = True, 
            kill_low = True, const_percent = False, wipe_saliency = False) :
        masks, prune_breaks = self.net.prune(self.saliency, kill_fraction = kill_fraction, kill_low = kill_low, const_percent = const_percent)

        #self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)
        if (save_path != None):
            self.save_explicit(save_path)
        
        if wipe_saliency:
            self.saliency = [tf.zeros([l.units]) for l in self.net.saliency_tracked_layers]
        else:
            self.saliency = [tf.gather(sal, mask, axis = 0) for sal, mask in zip(self.saliency, masks)]

        if verbose:
            print("pruned to", [g.shape for g in self.saliency])

        return prune_breaks
        
