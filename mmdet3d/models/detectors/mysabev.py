import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from .bevdet import BEVDepth4D
from .sabev import SABEV
from mmdet.models import DETECTORS
from torch.cuda.amp import autocast

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

@DETECTORS.register_module()
class MySABEV(SABEV):
    def __init__(
        self, 
        lss=False, 
        lc_fusion=False, 
        camera_stream=False,
        camera_depth_range=[4.0, 45.0, 1.0], 
        img_depth_loss_weight=1.0,  
        img_depth_loss_method='kld',
        grid=0.6, 
        num_views=4,
        se=False,
        final_dim=(900, 1600), 
        pc_range=[-50, -50, -5, 50, 50, 3], 
        downsample=4, 
        imc=256, 
        lic=384, 
        **kwargs
    ):
        super(MySABEV, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.lift = camera_stream
        self.se = se
        # if camera_stream:
        #     self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
        #     pc_range=pc_range, final_dim=final_dim, downsample=downsample)
        if lc_fusion:
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None, **kwargs):
        """Extract features from images and points."""
        img_bev_feats, img_preds = self.extract_img_feat(img, img_metas, **kwargs)
        img_bev_feat = img_bev_feats[0]
        pts_feats = self.extract_pts_feat(points, img_bev_feats, img_metas)

        if self.lift:
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]

        return dict(
            img_feats = img_bev_feats,
            pts_feats = pts_feats,
            img_preds = img_preds
        )
    
    def _draw_bev(self, pts_filename, points, corners, prefix):
        def cal(x, y):
            return int((x + 51.2) * 10), int((y + 51.2) * 10)
        
        import cv2
        import pathlib
        
        file_name = pts_filename.split('/')[-1].split('.')[0]
        dir = '/gpfsdata/home/huliang/bev/multi-SA-BEV/vis_dirs/points/' + file_name + '/'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        for point in points:
            x, y = cal(*point)
            cv2.circle(img, (x, y), radius=1, color=(255, 255, 255), thickness=-1)

        color=(0,0,255)
        for corner in corners:
            corner = np.array(corner[:, :2])
            corner = [(cal(x, y)) for x, y in corner]
            cv2.line(img, corner[0], corner[2], color, 1)
            cv2.line(img, corner[2], corner[6], color, 1)
            cv2.line(img, corner[6], corner[4], color, 1)
            cv2.line(img, corner[4], corner[0], color, 1)
        
        cv2.imwrite(dir + prefix + 'points_image.png', img)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        if self.use_bev_paste:
            from mmdet3d.core.points import LiDARPoints
            B = len(gt_bboxes_3d)
            paste_idx = []
            for i in range(B):
                for j in range(i, i + 1):
                    if j+1>=B: j-=B
                    paste_idx.append([i,j+1])
            
            gt_boxes_paste = []
            gt_labels_paste = []
            bda_mat_paste = []
            for i in range(len(paste_idx)):
                gt_boxes_tmp = []
                gt_labels_tmp = []
                for j in paste_idx[i]:
                    gt_boxes_tmp.append(gt_bboxes_3d[j])
                    gt_labels_tmp.append(gt_labels_3d[j])
                gt_boxes_tmp = torch.cat([tmp.tensor for tmp in gt_boxes_tmp], dim=0)
                gt_labels_tmp = torch.cat(gt_labels_tmp, dim=0)
                rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                gt_boxes_tmp, bda_rot = self.loader.bev_transform(gt_boxes_tmp.cpu(), rotate_bda, scale_bda, flip_dx, flip_dy)
                gt_boxes_tmp = gt_bboxes_3d[0].new_box(gt_boxes_tmp.cuda())
                bda_mat_paste.append(bda_rot.cuda())
                gt_boxes_paste.append(gt_boxes_tmp)
                gt_labels_paste.append(gt_labels_tmp)

                # bda to points
                # TODO 融合不同场景的points
                f_points = LiDARPoints(points[i], points_dim=points[i].shape[-1])
                f_points.rotate(bda_rot)
                points.append(f_points.tensor)

                # self._draw_bev(img_metas[i]['pts_filename'], f_points.bev, gt_boxes_tmp.corners, 'bda_ch_')
            gt_bboxes_3d = gt_boxes_paste
            gt_labels_3d = gt_labels_paste
            img_inputs.append(paste_idx)
            img_inputs.append(torch.stack(bda_mat_paste))

            
        
        # self._draw_bev('load_pts.png', points[0][:,:2], gt_bboxes_3d[0].corners)
        feature_dict = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        img_preds = feature_dict['img_preds']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            gt_depth = kwargs['gt_depth']
            gt_semantic = kwargs['gt_semantic']
            loss_depth, loss_semantic = \
                self.img_view_transformer.get_loss(img_preds, gt_depth, gt_semantic)
            losses_img = dict(loss_depth=loss_depth, loss_semantic=loss_semantic)
            losses.update(losses_img)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        pts_feats = feature_dict['pts_feats'] 
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list