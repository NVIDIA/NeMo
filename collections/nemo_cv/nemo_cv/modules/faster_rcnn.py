# Copyright (C) tkornuta, NVIDIA AI Applications Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

__author__ = "Tomasz Kornuta"

from torchvision.ops import misc as misc_nn_ops
from collections import OrderedDict

import torch
import torchvision.models.detection as detection
import torchvision.models.utils as utils

from torchvision.models.detection import _utils as det_utils
import torch.nn.functional as F

from ..utils.object_detection.rpn import AnchorGenerator, RPNHead, \
    RegionProposalNetwork
from ..utils.object_detection.roi_heads import RoIHeads

from torchvision.models.detection.transform import \
    GeneralizedRCNNTransform

from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.detection.faster_rcnn import TwoMLPHead, \
    FastRCNNPredictor

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from nemo.backends.pytorch.nm import TrainableNM

from nemo.core import NeuralType, AxisType, DeviceType,\
    BatchTag, ChannelTag, HeightTag, WidthTag, ListTag, BoundingBoxTag, \
    LogProbabilityTag

from ..utils.utils import pad_tensors_to_max


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
    'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(
            labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
            kp, proposals_per_image, discretization_size
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


class FasterRCNN(TrainableNM):
    """
        Wrapper class around the Faster R-CNN model.
    """

    @staticmethod
    def create_ports():
        input_ports = {
            # Batch of images.
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 3),
                                  2: AxisType(HeightTag),
                                  3: AxisType(WidthTag)}),
            # Batch of bounding boxes.
            "bounding_boxes": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(ListTag),
                                          2: AxisType(BoundingBoxTag)}),
            # Batch of targets.
            "targets": NeuralType({0: AxisType(BatchTag)}),
            # "Artificial" variable - tensor storing numbers of objects.
            "num_objects": NeuralType({0: AxisType(BatchTag)})
        }
        output_ports = {
            "predictions": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(LogProbabilityTag)
                                       })

        }
        return input_ports, output_ports

    def __init__(self, num_classes=91, progress=True, pretrained=False,
                 pretrained_backbone=True,
                 # transform parameters
                 min_size=800, max_size=1333,
                 # RPN parameters
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_score_thresh=0.05, box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25
                 ):
        """
        Creates the Faster R-CNN model.

        Args:
            num_classes: Number of output classes of the model.
            pretrained: use weights of model pretrained on COCO train2017.
        """

        super().__init__()

        # Create the model.
        if pretrained:
            # no need to download the backbone if pretrained is set
            pretrained_backbone = False

        # Create the ResNet+FPN "backbone".
        backbone = detection.backbone_utils.resnet_fpn_backbone(
            'resnet50', pretrained_backbone)

        # Create the other pieces of the model.
        out_channels = backbone.out_channels

        # if rpn_anchor_generator is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        # if rpn_head is None:
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[
                0]
        )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # SAMPLER USED IN CALCULATION OF RPN LOSS!
        # batch_size_per_image (int): number of anchors that are sampled during training of the RPN
        #    for computing the loss
        # positive_fraction (float): proportion of positive anchors in a mini-batch during training
        #    of the RPN
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            rpn_batch_size_per_image, rpn_positive_fraction
        )

        # if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2)

        # if box_head is None:
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        # if box_predictor is None:
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            None,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        # if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
        # if image_std is None:
        image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        # self.model = detection.FasterRCNN(backbone, num_classes)

        # super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

        # if pretrained:
        #    state_dict = utils.load_state_dict_from_url(
        #        model_urls['fasterrcnn_resnet50_fpn_coco'],
        #        progress=progress)
        #    self.model.load_state_dict(state_dict)

        # Get number of input features for the classifier.
        in_features = self.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one.
        self.roi_heads.box_predictor = \
            detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        self.to(self._device)

    ###########################################################################
    # PROCESSING RELATED METHODS.
    ###########################################################################

    def expand_masks(self, mask, padding):
        M = mask.shape[-1]
        scale = float(M + 2 * padding) / M
        padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
        return padded_mask, scale

    def resize_boxes(self, boxes, original_size, new_size):
        ratios = tuple(float(s) / float(s_orig)
                       for s, s_orig in zip(new_size, original_size))
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def paste_masks_in_image(self, masks, boxes, img_shape, padding=1):
        masks, scale = self.expand_masks(masks, padding=padding)
        boxes = self.expand_boxes(boxes, scale).to(dtype=torch.int64).tolist()
        # im_h, im_w = img_shape.tolist()
        im_h, im_w = img_shape
        res = [
            self.paste_mask_in_image(m[0], b, im_h, im_w)
            for m, b in zip(masks, boxes)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, im_h, im_w))
        return res

    def postprocess(self, predictions, image_shapes,
                    original_image_sizes):
        # if self.training:
        #    return result
        result = predictions
        for i, (pred, im_s, o_im_s) in enumerate(zip(predictions,
                                                     image_shapes,
                                                     original_image_sizes)):
            boxes = pred["boxes"]
            boxes = self.resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = self.paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = self.resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

    ###########################################################################
    # LOSS - region proposals.
    ###########################################################################

    def compute_loss(self, objectness, pred_bbox_deltas, labels,
                     regression_targets):
        """
        Arguments:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(
            torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(
            torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    ###########################################################################
    # FORWARD.
    ###########################################################################

    def forward(self, images, bounding_boxes, targets, num_objects):
        """
        Performs the forward step of the model.

        Args:
            images: Batch of images to be classified.
        """
        # print("Faster R-CNN forward:")
        # We need to put this in a tuple again, as OD "framework" assumes it :]

        # Unstack tensors with boxes and target, removing the "padded objects".
        bboxes_padded = torch.unbind(bounding_boxes, dim=0)
        targets_padded = torch.unbind(targets, dim=0)

        # Unpad bounding boxes.
        bboxes_unpadded = []
        for i in range(len(bounding_boxes)):
            bboxes_unpadded.append(bboxes_padded[i][0:num_objects[i], :])

        # Unpad targets.
        targets_unpadded = []
        for i in range(len(targets_padded)):
            targets_unpadded.append(targets_padded[i][0:num_objects[i]])

        targets_tuple = [{"boxes": b, "labels": t} for b, t
                         in zip(bboxes_unpadded, targets_unpadded)]

        # THE PROPPER forward pass.
        #######################################################################
        # loss_dict = self.model(images, targets_tuple)

        if self.training and targets_tuple is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]

        # Preprocess the images.
        images, targets_tuple = self.transform(images, targets_tuple)

        # Extract the features.
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        # Calculate the region proposals.
        proposals, anchors, objectness, pred_bbox_deltas = \
            self.rpn(images, features,
                     targets_tuple)

        # Empty!!! No detections in "training" mode.
        # print("Proposals for image 0: ", len(proposals[0]))

        # Calculate the regions.
        detections, class_logits, box_regression, labels, regression_targets = \
            self.roi_heads(
                features, proposals, images.image_sizes, targets_tuple)

        ######################################################################
        # ROI heads losses.
        ######################################################################

        detector_losses = {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            detector_losses = dict(loss_classifier=loss_classifier,
                                   loss_box_reg=loss_box_reg)

        if self.roi_heads.has_mask:
            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)

            detector_losses.update(loss_mask)

        if self.roi_heads.has_keypoint:
            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t["keypoints"] for t in targets]
                loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = dict(loss_keypoint=loss_keypoint)

            detector_losses.update(loss_keypoint)

        # Empty!!! No detections in "training" mode.
        # print(len(detections))
        # print(detections[0].keys()) # boxes, labels,scores
        # print(len(detections[0]["boxes"]))

        # Postprocess the images.
        detections = self.postprocess(
            detections, images.image_sizes, original_image_sizes)

        #######################################################################
        # RPN losses.
        proposal_losses = {}
        if self.training:
            labels, matched_gt_boxes = self.rpn.assign_targets_to_anchors(
                anchors, targets_tuple)

            regression_targets = self.rpn.box_coder.encode(
                matched_gt_boxes, anchors)

            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)

            proposal_losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        loss_dict = {}
        loss_dict.update(detector_losses)
        loss_dict.update(proposal_losses)

        # if self.training:
        #    return losses

        # Return.
        #######################################################################

        # Sum losses.
        losses = sum(loss for loss in loss_dict.values())

        print("Loss = ", losses.item())

        return losses
