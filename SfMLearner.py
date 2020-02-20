import os
import time
import math
import numpy as np
import tensorflow as tf
import tf_cylindrical as cylin
import cv2
from data_loader import DataLoader
from nets import *
from utils import *

class SfMLearner(object):
    def __init__(self):
        # model variables
        self.depth_net = disp_net()
        self.pose_net = pose_exp_net()

        # internal state variables
        self.tgt_image = None
        self.mask_ph = None
        self.mask_image = None
        self.pred_poses = None
        self.pred_depth = None
        self.steps_per_epoch = None

        self.total_loss = None
        self.pixel_loss = None
        self.exp_loss = None
        self.smooth_loss = None
        self.tgt_image_all = None
        self.src_image_stack_all = None
        self.proj_image_stack_all = None
        self.proj_error_stack_all = None
        self.exp_mask_stack_all = None
        self.mask_stack_all = None

        self.grads_and_vars = None
        self.train_op = None
        self.global_step = None
        self.incr_global_step = None

    def prune_graph(self):
        layers = self.depth_net.layers + self.prune_net.layers
        neurons = [np.mean(dl.output, axis = (0,1,2)) for dl in layers]
        second_grads = tf.hessians(self.total_loss, neurons)

        last_mask = None
        for dl, grad2 in zip(layers, second_grads):
            last_mask = dl.prune(grad2, last_mask)
        self.generate_train_ops()


    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            self.tgt_image, self.src_image_stack, self.intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(self.tgt_image)
            if opt.mask_path=='':
                opt.mask_path=None
            if opt.mask_path:
                self.mask_ph = tf.placeholder('float32',shape=self.tgt_image.shape[1:3])
                self.mask_image = tf.expand_dims(self.mask_ph,axis=0)
                self.mask_image = tf.expand_dims(self.mask_image,axis=-1)
            self.src_image_stack = self.preprocess_image(self.src_image_stack)

        with tf.name_scope("depth_prediction"):
            self.pred_disp = self.depth_net(self.tgt_image,
                                        is_training=True,
                                        do_wrap=opt.do_wrap)
            self.pred_depth = [1./d for d in self.pred_disp]
            self.pred_depth = [normalize_by_mean(d) for d in self.pred_depth]
            self.pred_disp = [normalize_by_mean(d) for d in self.pred_disp]

        with tf.name_scope("pose_and_explainability_prediction"):
            self.pred_poses = self.pose_net(tgt_image,
                             src_image_stack,
                             do_exp=(opt.explain_reg_weight > 0),
                             is_training=True,
                             do_wrap=opt.do_wrap)

        # Collect tensors that are useful later (e.g. tf summary)
        if opt.mask_path:
            self.mask_ph = mask_ph
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        
        self.generate_train_ops()

    def generate_train_ops(self):
        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            mask_stack_all = []
            # for each prediction scale
            for s in range(opt.num_scales):
                # Scale the source and target images for computing loss at the
                # according scale.
                curr_tgt_image = tf.image.resize_area(self.tgt_image,
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_src_image_stack = tf.image.resize_area(self.src_image_stack,
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                if opt.mask_path:
                    curr_mask = tf.image.resize_area(self.mask_image,
                        [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                # smoothness losses
                smooth_loss += opt.smooth_weight / (2**s) * self.depth_smoothness(
                    self.pred_disp[s], self.curr_tgt_image)

                # for each source image before current/target frame
                for i in range(opt.num_source):
                    # Inverse warp the Source image to the Target image frame using predicted Pose
                    mask = opt.mask_path and curr_mask
                    curr_proj_image = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)],
                        tf.squeeze(pred_depth[s], axis=3),
                        self.pred_poses[:,i,:],
                        self.intrinsics[:,s,:,:],
                        do_wrap=opt.do_wrap,
                        is_cylin=opt.cylindrical,
                        mask=mask)

                    # get projection/pixel error
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    if opt.mask_path:
                        curr_proj_error = curr_proj_error * curr_mask
                    # Photo-consistency loss weighted by explainability
                    if opt.explain_reg_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error)
                    

                    # dissabled in current version 2-10-2020
                    # get cross entropy error
                    # Cross-entropy loss as regularization for the
                    # explainability prediction
                    if opt.explain_reg_weight > 0:
                        ref_exp_mask = self.get_reference_explain_mask(s)
                        curr_exp_logits = tf.slice(pred_exp_logits[s],
                                                   [0, 0, 0, i*2],
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp_logits,
                                                      ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)


                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                        if opt.mask_path:
                            mask_stack = curr_mask
                    else:
                        proj_image_stack = tf.concat([proj_image_stack,
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack,
                                                      curr_proj_error], axis=3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack,
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)

                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.mask_path:
                    mask_stack_all.append(mask_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)



            total_loss = pixel_loss + smooth_loss + exp_loss
            self.total_loss = total_loss
            self.pixel_loss = pixel_loss
            self.exp_loss = exp_loss
            self.smooth_loss = smooth_loss
            self.tgt_image_all = tgt_image_all
            self.src_image_stack_all = src_image_stack_all
            self.proj_image_stack_all = proj_image_stack_all
            self.proj_error_stack_all = proj_error_stack_all
            self.exp_mask_stack_all = exp_mask_stack_all
            self.mask_stack_all = mask_stack_all

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            self.grads_and_vars = optim.compute_gradients(total_loss,
                                                          var_list=train_vars)
            self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (opt.batch_size,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
        opt = self.opt
        def gradient(pred):
            if opt.do_wrap:
                wrapped = cylin.wrap(pred, wrapping=2)  # one column per side
                wrapped_D_dx = wrapped[:, :, 1:, :] - wrapped[:, :, :-1, :]
                D_dx = cylin.unwrap(wrapped_D_dx, wrapping=2)
            else:
                D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    # from https://github.com/tensorflow/models/blob/master/research/vid2depth/model.py#L243 
    def depth_smoothness(self, depth, img):
       """Computes image-aware depth smoothness loss."""
       opt = self.opt
       def gradient(pred):
           if opt.do_wrap:
               wrapped = cylin.wrap(pred, wrapping=2)  # one column per side
               wrapped_D_dx = wrapped[:, :, 1:, :] - wrapped[:, :, :-1, :]
               D_dx = cylin.unwrap(wrapped_D_dx, wrapping=2)
           else:
               D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
           D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
           return D_dx, D_dy
       depth_dx, depth_dy = gradient(depth)
       image_dx, image_dy = gradient(img)
       weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
       weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
       smoothness_x = depth_dx * weights_x
       smoothness_y = depth_dy * weights_y
       return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        if opt.explain_reg_weight > 0:
            tf.summary.scalar("exp_loss", self.exp_loss)
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i),
                    self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
            if opt.mask_path:
              tf.summary.image('scale%d_mask' % s, self.mask_stack_all[s])
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        if opt.mask_path:
            mask = cv2.imread(opt.mask_path, flags=cv2.IMREAD_GRAYSCALE).astype('float32')/255.
        opt.num_source = opt.seq_length - 1 
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)



            start_time = time.time()
            for step in range(1, opt.max_steps):
                # populate essential operations
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                # add summary statistics
                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                # run train
                if opt.mask_path:
                    results = sess.run(fetches,feed_dict={self.mask_ph:mask})
                else:
                    results = sess.run(fetches)
                gs = results["global_step"]

                # generate summary
                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq,
                                results["loss"]))
                    start_time = time.time()

                #save on frequency
                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                #save on epoch
                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            self.depth_net = disp_net()
            pred_disp = self.depth_net(input_mc)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self, do_wrap=True):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
            self.img_height, self.img_width * self.seq_length, 3],
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _, pose_layers = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False, do_wrap=do_wrap)
            self.inputs = input_uint8
            self.pred_poses = pred_poses
            self.pose_layers = pose_layers

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1,
                        do_wrap=True):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph(do_wrap=do_wrap)

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
