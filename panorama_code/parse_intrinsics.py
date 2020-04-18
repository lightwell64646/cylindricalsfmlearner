import tensorflow as tf

def get_multi_scale_intrinsics(intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

def make_intrinsics_matrix(fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

def parse_intrinsics(intrinsics_file, num_scales):
    rec_def = [1 for _ in range(9)]
    raw_cam_contents = open(intrinsics_file, 'rb').read()
    raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
    raw_cam_vec = tf.stack(raw_cam_vec)
    intrinsics = tf.reshape(raw_cam_vec, [3, 3])
    intrinsics = get_multi_scale_intrinsics(
            intrinsics, num_scales)