import tensorflow as tf

def get_multi_scale_intrinsics(intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[0]/(2 ** s)
            fy = intrinsics[1]/(2 ** s)
            cx = intrinsics[2]/(2 ** s)
            cy = intrinsics[3]/(2 ** s)
            intrinsics_mscale.append(
                make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

def make_intrinsics_matrix(fx, fy, cx, cy):
        # Assumes batch input
        r1 = tf.constant([float(fx), 0, float(cx)])
        r2 = tf.constant([0, float(fy), float(cy)])
        r3 = tf.constant([0.,0.,1.])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

def parse_intrinsics(intrinsics_file, num_scales):
    rec_def = [float(1) for _ in range(4)]
    raw_cam_contents = open(intrinsics_file, 'rb').read()
    raw_cam_vec = tf.io.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def,
                                    field_delim=' ')
    raw_cam_vec = tf.stack(raw_cam_vec)
    intrinsics = get_multi_scale_intrinsics(
            raw_cam_vec, num_scales)
    return intrinsics