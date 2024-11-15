import numpy as np
from ...utils import box_utils
import math
import numba


def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def transform_annotations_to_kitti_format_v1(annos, map_name_to_kitti=None, info_with_fakelidar=False, is_gt = True):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            # anno['name'][k] = map_name_to_kitti[anno['name'][k]] if anno['name'][k] != 'DontCare' else anno['name'][k]
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bev'] = np.zeros((len(anno['name']), 4))
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = gt_boxes_lidar[:, 0]
            anno['location'][:, 1] = gt_boxes_lidar[:, 1]
            anno['location'][:, 2] = gt_boxes_lidar[:, 2]

            if is_gt:
                dxdydz = gt_boxes_lidar[:, 3:6]
                anno['dimensions'] = dxdydz[:, [0, 1, 2]]  # lwh ==> lwh
                r_y = gt_boxes_lidar[:, 6]
                alpha = 90 - (r_y * 180) / np.pi
                anno['alpha'] = alpha
                anno['rotation_y'] = alpha * np.pi / 180
            else:
                dxdydz = gt_boxes_lidar[:, 3:6]
                anno['dimensions'] = dxdydz[:, [0, 1, 2]]  # lwh ==> lwh
                r_y = gt_boxes_lidar[:, 6]
                alpha = (r_y * 180) / np.pi
                anno['alpha'] = alpha
                anno['rotation_y'] = r_y

            # anno['bbox'] [y_min, z_min, y_max, z_max]
            # anno['bev'] [x_min, y_min, x_max, y_max]
            for i in range(len(anno['rotation_y'])):
                alpha_radian = anno['rotation_y'][i]
                l = anno['dimensions'][i, 0]
                w = anno['dimensions'][i, 1]
                h = anno['dimensions'][i, 2]
                dx = abs(math.cos(alpha_radian) * l) + abs(math.sin(alpha_radian) * w)
                dy = abs(math.sin(alpha_radian) * l) + abs(math.cos(alpha_radian) * w)
                dz = abs(h)
                anno['bbox'][i, 0] = anno['location'][i, 1] - dy / 2.0
                anno['bbox'][i, 2] = anno['location'][i, 1] + dy / 2.0
                anno['bbox'][i, 1] = anno['location'][i, 2] - dz / 2.0
                anno['bbox'][i, 3] = anno['location'][i, 2] + dz / 2.0

                anno['bev'][i, 0] = anno['location'][i, 0] - dx / 2.0
                anno['bev'][i, 2] = anno['location'][i, 0] + dx / 2.0
                anno['bev'][i, 1] = anno['location'][i, 1] - dy / 2.0
                anno['bev'][i, 3] = anno['location'][i, 1] + dy / 2.0

        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2