# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FasterRcnn Rcnn network."""

import numpy as np
import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore import context


class DenseNoTranpose(nn.Cell):
    """Dense method"""
    def __init__(self, input_channels, output_channels, weight_init):
        super(DenseNoTranpose, self).__init__()
        self.weight = Parameter(initializer(weight_init, [input_channels, output_channels], mstype.float32))
        self.bias = Parameter(initializer("zeros", [output_channels], mstype.float32))

        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"

    def construct(self, x):
        if self.device_type == "Ascend":
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            output = self.bias_add(self.matmul(x, weight), self.bias)
        else:
            output = self.bias_add(self.matmul(x, self.weight), self.bias)
        return output

class NormAwareEmbedding(nn.Cell):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            nn.init.normal_(proj[0].weight, std=0.01)
            nn.init.normal_(proj[1].weight, std=0.01)
            nn.init.constant_(proj[0].bias, 0)
            nn.init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = mindspore.ops.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp

class Rcnn(nn.Cell):
    def select_training_samples_gt(
        self,
        proposals, 
        targets,  
        is_source = True
    ):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = mindspore.ops.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, matched_gt_boxes

    def select_training_samples_da(self,
                                proposals, 
                                targets     
                                ):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        domain_labels = [t["domain_labels"] for t in targets]

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        for is_source, labels_per_image in zip(domain_labels, labels):
            labels_per_image[:] = 0
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        
        all_domains = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = mindspore.ops.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            # print(proposals[img_id].shape)

            domain_label = mindspore.ops.ones_like(labels[img_id], dtype=mindspore.dtype.uint8) if domain_labels[img_id].any() else mindspore.ops.zeros_like(labels[img_id], dtype=mindspore.dtype.uint8)
            all_domains.append(domain_label)

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, all_domains, matched_gt_boxes

    def extract_da(self,
                features,      
                proposals,     
                image_shapes,  
                targets=None,   
                ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if self.training:
            proposals, matched_idxs, labels, regression_targets, domain_labels_before,_ = self.select_training_samples_da(proposals, targets)
        

        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]
        )
        boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
        boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
        boxes, matched_idxs, labels, regression_targets, domain_labels,_ = self.select_training_samples_da(boxes, targets)
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features)
        box_embeddings, box_cls_scores = self.embedding_head(box_features)

        return box_embeddings, domain_labels, proposal_features["feat_res5"], domain_labels_before

    """
    Rcnn subnet.

    Args:
        config (dict) - Config.
        representation_size (int) - Channels of shared dense.
        batch_size (int) - Batchsize.
        num_classes (int) - Class number.
        target_means (list) - Means for encode function. Default: (.0, .0, .0, .0]).
        target_stds (list) - Stds for encode function. Default: (0.1, 0.1, 0.2, 0.2).

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        Rcnn(config=config, representation_size = 1024, batch_size=2, num_classes = 81, \
             target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    """
    def __init__(self,
                 config,
                 representation_size,
                 batch_size,
                 num_classes,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(Rcnn, self).__init__()
        cfg = config
        self.dtype = np.float32
        self.ms_type = mstype.float32
        self.rcnn_loss_cls_weight = Tensor(np.array(cfg.rcnn_loss_cls_weight).astype(self.dtype))
        self.rcnn_loss_reg_weight = Tensor(np.array(cfg.rcnn_loss_reg_weight).astype(self.dtype))
        self.rcnn_fc_out_channels = cfg.rcnn_fc_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_classes = num_classes
        self.in_channels = cfg.rcnn_in_channels
        self.train_batch_size = batch_size
        self.test_batch_size = cfg.test_batch_size
        self.box_roi_pool = mindspore.nn.MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        self.embedding_head = NormAwareEmbedding()
        self.reid_head = nn.ResNet50["layer4"]
        self.memory = None

        shape_0 = (self.rcnn_fc_out_channels, representation_size)
        weights_0 = initializer("XavierUniform", shape=shape_0[::-1], dtype=self.ms_type).to_tensor()
        shape_1 = (self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        weights_1 = initializer("XavierUniform", shape=shape_1[::-1], dtype=self.ms_type).to_tensor()
        self.shared_fc_0 = DenseNoTranpose(representation_size, self.rcnn_fc_out_channels, weights_0)
        self.shared_fc_1 = DenseNoTranpose(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels, weights_1)

        cls_weight = initializer('Normal', shape=[num_classes, self.rcnn_fc_out_channels][::-1],
                                 dtype=self.ms_type).to_tensor()
        reg_weight = initializer('Normal', shape=[num_classes * 4, self.rcnn_fc_out_channels][::-1],
                                 dtype=self.ms_type).to_tensor()
        self.cls_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes, cls_weight)
        self.reg_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes * 4, reg_weight)

        self.flatten = P.Flatten()
        self.relu = P.ReLU()
        self.logicaland = P.LogicalAnd()
        self.loss_cls = P.SoftmaxCrossEntropyWithLogits()
        self.loss_bbox = P.SmoothL1Loss(beta=1.0)
        self.reshape = P.Reshape()
        self.onehot = P.OneHot()
        self.greater = P.Greater()
        self.cast = P.Cast()
        self.sum_loss = P.ReduceSum()
        self.tile = P.Tile()
        self.expandims = P.ExpandDims()

        self.gather = P.GatherNd()
        self.argmax = P.ArgMaxWithValue(axis=1)

        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.value = Tensor(1.0, self.ms_type)

        self.num_bboxes = (cfg.num_expected_pos_stage2 + cfg.num_expected_neg_stage2) * batch_size

        rmv_first = np.ones((self.num_bboxes, self.num_classes))
        rmv_first[:, 0] = np.zeros((self.num_bboxes,))
        self.rmv_first_tensor = Tensor(rmv_first.astype(self.dtype))

        self.num_bboxes_test = cfg.rpn_max_num * cfg.test_batch_size

        range_max = np.arange(self.num_bboxes_test).astype(np.int32)
        self.range_max = Tensor(range_max)

    def construct(self, featuremap, bbox_targets, labels, mask):
        x = self.flatten(featuremap)

        x = self.relu(self.shared_fc_0(x))
        x = self.relu(self.shared_fc_1(x))

        x_cls = self.cls_scores(x)
        x_reg = self.reg_scores(x)

        if self.training:
            bbox_weights = self.cast(self.logicaland(self.greater(labels, 0), mask), mstype.int32) * labels
            labels = self.onehot(labels, self.num_classes, self.on_value, self.off_value)
            bbox_targets = self.tile(self.expandims(bbox_targets, 1), (1, self.num_classes, 1))

            loss, loss_cls, loss_reg, loss_print = self.loss(x_cls, x_reg, bbox_targets, bbox_weights, labels, mask)
            out = (loss, loss_cls, loss_reg, loss_print)
        else:
            out = (x_cls, (x_cls / self.value), x_reg, x_cls)

        return out

    def loss(self, cls_score, bbox_pred, bbox_targets, bbox_weights, labels, weights, is_source=False):
        """Loss method."""
        loss_print = ()
        loss_cls, _ = self.loss_cls(cls_score, labels)
        boxes, matched_idxs, box_pid_labels, box_reg_targets, matched_gt_boxes = self.select_training_samples_gt(boxes, bbox_targets)
        weights = self.cast(weights, self.ms_type)
        loss_cls = loss_cls * weights
        loss_cls = self.sum_loss(loss_cls, (0,)) / self.sum_loss(weights, (0,))

        bbox_weights = self.cast(self.onehot(bbox_weights, self.num_classes, self.on_value, self.off_value),
                                 self.ms_type)
        bbox_weights = bbox_weights * self.rmv_first_tensor

        pos_bbox_pred = self.reshape(bbox_pred, (self.num_bboxes, -1, 4))
        loss_reg = self.loss_bbox(pos_bbox_pred, bbox_targets)
        loss_reg = self.sum_loss(loss_reg, (2,))
        loss_reg = loss_reg * bbox_weights
        loss_reg = loss_reg / self.sum_loss(weights, (0,))
        loss_reg = self.sum_loss(loss_reg, (0, 1))

        loss = self.rcnn_loss_cls_weight * loss_cls + self.rcnn_loss_reg_weight * loss_reg
        loss_print += (loss_cls, loss_reg)
        box_features = self.box_roi_pool(bbox_weights, boxes)
        box_features = self.reid_head(box_features, is_source)
        box_embeddings, box_cls_scores = self.embedding_head(box_features)
        if is_source:
            loss_box_reid_s = self.memory(box_embeddings, box_pid_labels, is_source=True)
            loss.update(loss_box_reid_s=loss_box_reid_s)
        else:
            loss_box_reid_t = self.memory(box_embeddings, box_pid_labels, is_source=False)
            loss.update(loss_box_reid_t=loss_box_reid_t)

        return loss, loss_cls, loss_reg, loss_print




