import json
import os
import numpy as np
import shutil
from mmdet3d.core.visualizer import show_result

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval


file_name = 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590'
pwd_root = 'SA-BEV/'
version = 'v1.0-mini'
dataroot = pwd_root + 'data/nuscenes/'
result_path = pwd_root + 'work_dirs/test/results_eval/pts_bbox/results_nusc.json'
out_dir = pwd_root + 'work_dirs/test/output/'    # useless
save_path = pwd_root + 'work_dirs/pre_outs/'


def get_sample_token(file_name: str):
    with open(dataroot + version + '/sample_data.json', encoding='utf-8') as data_file:
        datas = json.load(data_file)
    sample_token = None
    for data in datas:
        if data['filename'].find(file_name) != -1:
            sample_token = data['sample_token']
            break
    return sample_token


# 根据obj文件名读取点云
def get_point_cloud(file_name: str):
    obj_file_path = pwd_root + 'work_dirs/show/' + file_name + '/' + file_name + '_points.obj'
    with open(obj_file_path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
    # points原本为列表，需要转变为矩阵，方便处理          
    points = np.array(points)
    return points


def draw_img():
    nusc = NuScenes('v1.0-mini', dataroot='data/nuscenes/', verbose=False)
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }
    nusc_eval = NuScenesEval(
        nusc,
        config=config_factory('detection_cvpr_2019'),
        result_path=result_path,
        eval_set=eval_set_map[version],
        output_dir=out_dir,
        verbose=False)

    # sample_token = '3e8750f331d7499e9b5123e9eb70f2e2'

    sample_token = get_sample_token(file_name)
    find_sample = nusc.get('sample', sample_token)

    # 需要提取的传感器
    sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    # 保存路径
    tar_path = save_path + sample_token
    # 在save_path下，创捡名为sample_token的文件夹
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    gt_boxes = nusc_eval.gt_boxes.boxes[sample_token]
    pred_boxes = nusc_eval.pred_boxes.boxes[sample_token]
    points = get_point_cloud(file_name)
    if len(gt_boxes) > 0:
        show_result(points, gt_boxes, pred_boxes, out_dir, file_name)
    else:
        print('no boxes in this scene')
    # for sensor in sensors:
    #     sensor = nusc.get('sample_data', find_sample['data'][sensor])
    #     per_img = dataroot + sensor['filename']
        
    #     shutil.copy(per_img, tar_path)

def test_draw():
    from nuscenes.nuscenes import NuScenes
    import json
    import os
    import shutil
    import torch
    from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
    from pyquaternion import Quaternion
    import numpy as np
    import torch.nn.functional as F
    import cv2
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

    nusc =NuScenes(version=version, dataroot=dataroot)
    # bevdepth_bev_root = 'bev_feat_bevdepth'
    sabev_bev_root = result_path
    # bevdepth_preds = json.load(open('results_bevdepth.json','rb'))['results']
    sabev_preds = json.load(open(sabev_bev_root,'rb'))['results']
    NameMapping = {
        'barrier':'movable_object.barrier' ,
        'bicycle': 'vehicle.bicycle',
        'bus': 'vehicle.bus.bendy',
        'bus': 'vehicle.bus.rigid',
        'car': 'vehicle.car',
        'construction_vehicle': 'vehicle.construction',
        'motorcycle': 'vehicle.motorcycle',
        'pedestrian': 'human.pedestrian.adult',
        'pedestrian': 'human.pedestrian.child',
        'pedestrian': 'human.pedestrian.construction_worker',
        'pedestrian': 'human.pedestrian.police_officer',
        'traffic_cone': 'movable_object.trafficcone',
        'trailer': 'vehicle.trailer',
        'truck': 'vehicle.truck'
    }

    sample_token="456ec36cb4a44ca78f36fbd90c0c34fa"
    pred=sabev_preds[sample_token]
    save_dir=save_path
    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    from PIL import Image
    from matplotlib import rcParams
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    tokens = nusc.get('sample', sample_token)['data']
    tokens = [tokens[key] for key in tokens.keys() if 'CAM' in key]
    for camera_token in tokens:
        # camera_token=nusc.get('sample', sample_token)['data']["CAM_FRONT"]
        lidar_token=nusc.get('sample', sample_token)['data']["LIDAR_TOP"]
        # data_path, boxes, camera_intrinsic = nusc.get_sample_data(camera_token)
        sd_record = nusc.get('sample_data', camera_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        # lidar_record = nusc.get('sample_data', lidar_token)
        # lidar_cs_record = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        data_path = nusc.get_sample_data_path(camera_token)
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
        boxes=[]
        for p in pred:
            if p['detection_score']<0.35: continue
            box = Box(p['translation'], p['size'], Quaternion(p['rotation']),
                        name=p['detection_name'])
            box.center[2]+=1
            # box.translate(-np.array(['translation']))
            # box.rotate(Quaternion(['rotation']).inverse)
            # box.translate(np.array(lidar_cs_record['translation']))
            # box.rotate(Quaternion(lidar_cs_record['rotation']))
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            # yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            # box.translate(-np.array(pose_record['translation']))
            # box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            if not box_in_image(box, cam_intrinsic, imsize):
                continue
            boxes.append(box)


        data = Image.open(data_path)
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(data)
        # Show boxes.
        for box in boxes:
            c = np.array(nusc.colormap[NameMapping[box.name]]) / 255.0
            box.render(ax, view=cam_intrinsic, normalize=True, colors=(c, c, c))
        # data_path, boxes, camera_intrinsic = nusc.get_sample_data(camera_token)
        # for box in boxes:
        #     c = np.array(nusc.colormap[box.name]) / 255.0
        #     box.render(ax, view=cam_intrinsic, normalize=True, colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)
        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(
            sd_record['channel'], labels_type=''))
        ax.set_aspect('equal')
        out_path = os.path.join(save_dir,'%s.png'%sd_record['channel'])
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

if __name__ == '__main__':
    test_draw()