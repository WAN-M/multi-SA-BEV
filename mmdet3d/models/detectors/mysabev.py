import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from .cam_stream_lss import LiftSplatShoot
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
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            pc_range=pc_range, final_dim=final_dim, downsample=downsample)
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
        pts_feats = self.extract_pts_feat(points, img_bev_feats, img_metas)

        if self.lift:
            # BN, C, H, W = img_feats[0].shape
            # batch_size = BN//self.num_views
            # img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            # rots = []
            # trans = []
            # for sample_idx in range(batch_size):
            #     rot_list = []
            #     trans_list = []
            #     for mat in img_metas[sample_idx]['lidar2img']:
            #         mat = torch.Tensor(mat).to(img_feats_view.device)
            #         rot_list.append(mat.inverse()[:3, :3])
            #         trans_list.append(mat.inverse()[:3, 3].view(-1))
            #     rot_list = torch.stack(rot_list, dim=0)
            #     trans_list = torch.stack(trans_list, dim=0)
            #     rots.append(rot_list)
            #     trans.append(trans_list)
            # rots = torch.stack(rots)
            # trans = torch.stack(trans)
            # lidar2img_rt = img_metas[sample_idx]['lidar2img']  #### extrinsic parameters for multi-view images
            
            # img_bev_feat, depth_dist = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas)
            # # print(img_bev_feat.shape, pts_feats[-1].shape)
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion:
                    img_bev_feat = img_bev_feats[0]
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
            gt_bboxes_3d = gt_boxes_paste
            gt_labels_3d = gt_labels_paste
            img_inputs.append(paste_idx)
            img_inputs.append(torch.stack(bda_mat_paste))
        
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