import os
import time
import tensorflow as tf

'''
This class runs a custom training loop that keeps an exponential
running average of prune metrics and provides utility to apply those
metrics to the underlying network. 

It also handles model saving and tensofboard logging.

Note: strategy scope must be opened while initializing values in 
net and when calling strategy.experimental_run_v2
'''
class prune_trainer_distributed(object):
    def __init__(self, opt, net_generator, dataset_generator, loss_function, do_accuracy = True):
        self.flags = opt
        self.strategy = tf.distribute.MirroredStrategy()
        self.do_accuracy = do_accuracy

        self.net_generator = net_generator
        self.dataset_generator = dataset_generator
        self.loss_function = loss_function

        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)
            self.net = net_generator(opt)

        self.data_loader = self.strategy.experimental_distribute_dataset(dataset_generator(opt.batch_size))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.global_batch_size = opt.batch_size * self.strategy.num_replicas_in_sync
        self.first = True
        self.metric_alpha = 0.99
        self.grad2 = [tf.zeros([l.units]) for l in self.net.prunable_layers]

    def getPruneMetric(self):
        return self.grad2
    def setPruneMetric(self, override):
        self.grad2 = override

    # Looses all methods not on base Keras model class if we use keras.model.save
    # syntax is a bit messier with checkpoints but it works
    def load_model(self, path):
        with self.strategy.scope():
            print("restored", self.checkpoint.restore(tf.train.latest_checkpoint(path)))
            for images, labels in self.data_loader:
                self.net._set_inputs(images)
                self.first = False
                break

    # sorry this is messy distributed training has funky syntax
    # first we define a tf.function that will handle every train step
    # then we run through data in a loop running that train step
    def train(self, max_steps = None):
        opt = self.flags
        if opt.init_checkpoint_file != None:
            self.load_model(opt.init_checkpoint_file)

        if max_steps == None:
            max_steps = opt.max_steps

        @tf.function
        def function_wrapped_training(images, labels):
            def do_train(images, labels):
                with tf.GradientTape(persistent=True) as tape:
                    predictions = self.net(images)
                    answer_loss = tf.nn.compute_average_loss(
                        self.loss_function(predictions, labels), self.global_batch_size)
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
                print(grad1)
                grads_per_neuron = tape.gradient(grad1, self.net.last_outs)
                for g in grads_per_neuron:
                    collapse_dims = tf.range(len(g.shape._dims) - 1)
                    g2.append(tf.reduce_mean(tf.abs(g), collapse_dims))
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
        logWriter = tf.summary.create_file_writer("./tmp/trainingLogs.log")
        with logWriter.as_default():
            step = 0
            with self.strategy.scope():
                for images, labels in self.data_loader:
                    # execute model
                    if self.first:
                        self.net._set_inputs(images)
                    
                    answer_loss, regularization_loss, g2 = function_wrapped_training(images, labels)
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
        print([[w.shape for w in l.get_weights()] for l in self.net.layers])
        self.checkpoint.save(path)

    def eval(self, eval_steps = None):
        @tf.function
        #start train step declaration
        def function_wrapped_testing(images, labels):

            #start per node declaration
            def do_test(images, labels):
                predictions = self.net(images)
                answer_loss = tf.nn.compute_average_loss(
                    self.loss_function(predictions, labels), self.global_batch_size)
                regularization_loss = tf.nn.scale_regularization_loss(self.net.losses)
                if self.do_accuracy:
                    self.epoch_accuracy(labels, predictions)

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
        loader = self.dataset_generator(opt.batch_size, is_training=False)
        al_sum = rl_sum = test_cycles = 0
        for image, labels in loader:
            answer_loss, regularization_loss = function_wrapped_testing(image, labels)
            al_sum += answer_loss
            rl_sum += regularization_loss
            test_cycles += 1
            if (eval_steps != None and eval_steps <= test_cycles):
                break

            
        average_anser_loss = al_sum / test_cycles
        tf.summary.scalar("test answer_loss", average_anser_loss)
        tf.summary.scalar("test regularization_loss", rl_sum / test_cycles)
        if self.do_accuracy:
            acc = self.epoch_accuracy.result()
            self.epoch_accuracy.reset_states()
            tf.summary.scalar("test accuracy", acc)
            print("test", (rl_sum + al_sum) / test_cycles, acc)
            return acc
        
        return average_anser_loss
        

    def prune(self, kill_fraction = 0.1, save_path = None):
        with self.strategy.scope():
            self.net.prune(self.grad2, kill_fraction=kill_fraction)

        self.grad2 = [tf.zeros([l.units]) for l in self.net.prunable_layers]
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)
        if (save_path != None):
            self.save_explicit(save_path)
        print("pruned", [g.shape for g in self.grad2])
        
