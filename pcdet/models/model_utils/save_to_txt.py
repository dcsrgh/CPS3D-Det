import os
import torch
import open3d as o3d
import numpy as np


def is_tensor(object):
    if isinstance(object,torch.Tensor):
        object = object.cpu().numpy()
    return object


def save_xyz_heat_value(root_path, frame_id, xyz, heat):
    assert len(xyz) == len(heat)
    save_file = os.path.join(root_path, frame_id + '.txt')
    new_xyz = np.empty((xyz.shape[0], xyz.shape[1] + 1))
    for i in range(len(xyz)):
        new_xyz[i] = np.append(xyz[i], heat[i], axis=None)
    new_xyz = np.asarray(new_xyz)
    np.savetxt(save_file, np.round(new_xyz, decimals=4))


def save_1(root_path, frame_id_list, pred_dicts):
    assert len(frame_id_list) == len(pred_dicts)
    for i in range(len(frame_id_list)):
        frame_id = is_tensor(frame_id_list[i])
        xyz = is_tensor(pred_dicts[i]['pred_boxes'])
        heat = is_tensor(pred_dicts[i]['pred_scores'])
        save_xyz_heat_value(root_path, frame_id, xyz, heat)

# 创建新的open3d对象
def create_new_o3d(pcd):
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pcd)
    return new_pcd

def compute_pc1_pc2_distance(pc1, pc2=None, distance_type='mahalanobis'):
    if not isinstance(pc1, np.arange):
        pc1 = pc1.cpu().numpy()
        pc1 = create_new_o3d(pc1)

    if pc2 is not None and not isinstance(pc2, np.arange):
        pc2 = pc2.cpu().numpy()
        pc2 = create_new_o3d(pc2)

    if distance_type is 'mahalanobis':
        distance = pc1.compute_mahalanobis_distance()
    if distance_type is 'neighbor':
        distance = pc1.compute_nearest_neighbor_distance()
    if distance_type is 'pointCloud':
        distance = pc1.compute_point_cloud_distance(pc2)
    
    pass
