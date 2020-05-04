import sys
sys.path.insert(0, "..")

from SFMLutils import projective_inverse_warp

import tensorflow as tf

#intrinsics must be of shape [batch, ]
def disparity_loss(pred, src_tgt_stack, opt):
    depths, poses = pred
    src = src_tgt_stack[:,:-1]
    depths = [1/d for d in depths]
    target_im = src_tgt_stack[:,-1]
    total_disparity = 0
    for s in range(opt.num_scales):
        for i in range(opt.num_source):
            scaled_src = src[:,i]
            target_im_scaled = target_im
            if (s != 0):
                scaled_src = tf.image.resize(scaled_src, depths[s][:,i].shape[1:3])
                target_im_scaled = tf.image.resize(target_im_scaled, depths[s][:,i].shape[1:3])
            curr_proj_image = projective_inverse_warp(
                scaled_src,
                tf.squeeze(depths[s][:,i], axis=-1),
                poses[:,i],
                opt.intrinsics[:,:,s],
                do_wrap=opt.do_wrap,
                is_cylin=opt.cylindrical)

            curr_proj_error = tf.abs(curr_proj_image - target_im_scaled)
            total_disparity += tf.reduce_mean(curr_proj_error)
    return total_disparity