import os
import time
import tensorflow as tf

'''
A single GPU version of prune_trainer_distributed

TODO: do object oriented stuff to reduce duplication
TODO: This code does not work? Testing
'''
class prune_trainer(object):
    def __init__(self, opt, net_generator, dataset_generator, loss_function):
        self.flags = opt
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)

        self.net_generator = net_generator
        self.dataset_generator = dataset_generator
        self.loss_function = loss_function

        self.data_loader = dataset_generator(opt.batch_size)
        self.net = net_generator(opt)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)

        self.first = True
        self.metric_alpha = 0.99
        self.grad2 = [tf.zeros([l.units]) for l in self.net.prunable_layers]

    def getPruneMetric(self):
        return self.grad2
    def setPruneMetric(self, ovefride):
        self.grad2 = ovefride

    def load_model(self, path):
        print("restored", self.checkpoint.restore(tf.train.latest_checkpoint(path)))
        for images, labels in self.data_loader:
            self.net._set_inputs(images)
            self.first = False
            break

    def do_train(self, images, labels):
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.net(images)
            answer_loss = tf.nn.compute_average_loss(
                self.loss_function(predictions, labels), self.flags.batch_size)
            regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
            loss = answer_loss + regularization_loss
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

    def train(self, max_steps = None):
        opt = self.flags
        if opt.init_checkpoint_file != None:
            self.load_model(opt.init_checkpoint_file)
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()


        checkpointManager = tf.train.CheckpointManager(self.checkpoint, opt.checkpoint_dir, opt.max_checkpoints_to_keep)
        logWriter = tf.summary.create_file_writer("./tmp/trainingLogs.log")
        with logWriter.as_default():
            step = 0
            for images, labels in self.data_loader:
                # execute model
                if self.first:
                    self.net._set_inputs(images)
                
                answer_loss, regularization_loss, g2 = self.do_train(images, labels)
                for i in range(len(g2)):
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
                if (step >= max_steps):
                    break

    def save_explicit(self, path):
        if (not os.path.isdir(path)):
            os.mkdir(path)
        print([[w.shape for w in l.get_weights()] for l in self.net.layers])
        self.checkpoint.save(path)

    def eval(self, eval_steps = None):
        def do_test(images, labels):
            predictions = self.net(images)
            answer_loss = tf.nn.compute_average_loss(
                self.loss_function(predictions, labels), self.flags.batch_size)
            regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
            self.epoch_accuracy(labels, predictions)

            return answer_loss, regularization_loss

        opt = self.flags
        loader = self.dataset_generator(opt.batch_size, is_training=False)
        al_sum = rl_sum = test_cycles = 0
        for image, labels in loader:
            answer_loss, regularization_loss = do_test(image, labels)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
            if (eval_steps != None and eval_steps <= test_cycles):
                break
        acc = self.epoch_accuracy.result()
        self.epoch_accuracy.reset_states()
        tf.summary.scalar("test accuracy", acc)
        tf.summary.scalar("test answer_loss", al_sum / test_cycles)
        tf.summary.scalar("test regularization_loss", rl_sum / test_cycles)
        tf.summary.scalar("test loss", (rl_sum + al_sum) / test_cycles)
        print("test", (rl_sum + al_sum) / test_cycles, acc)
    
        return acc
        

    def prune(self, kill_fraction = 0.1, save_path = None):
        self.net.prune(self.grad2, kill_fraction=kill_fraction)

        self.grad2 = [tf.zeros([l.units]) for l in self.net.prunable_layers]
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)
        if (save_path != None):
            self.save_explicit(save_path)
        print("pruned", [g.shape for g in self.grad2])
        
