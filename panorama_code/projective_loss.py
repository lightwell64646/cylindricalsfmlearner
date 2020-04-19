import sys
sys.path.insert(0, "..")

from SFMLutils import projective_inverse_warp

import tensorflow as tf

def disparity_loss(pred, src_tgt_stack, opt):
    depths, poses = pred
    src = src_tgt_stack[:-1]
    depths = [1/d for d in depths]
    target_im = src_tgt_stack[-1]
    total_disparity = 0
    for s in range(opt.num_scales):
        for i in range(opt.num_source):
            curr_proj_image = projective_inverse_warp(
                src[:,:,:,3*i:3*(i+1)],
                tf.squeeze(depths[s], axis=3),
                poses[:,i,:],
                opt.intrinsics[s,:,:],
                do_wrap=opt.do_wrap,
                is_cylin=opt.cylindrical)

            curr_proj_error = tf.abs(curr_proj_image - target_im)
            total_disparity += curr_proj_error
    return curr_proj_error