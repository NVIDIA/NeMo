# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Copyright (C) IBM Corporation 2019
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

__author__ = "Tomasz Kornuta and Anh Tuan Nguyen"

# This file contains code artifacts adapted from the original implementation:
# https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/models/vision/image_encoder.py

# The object detection was adapted from the following detectron2 implementation:
# https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/detectron2_mscoco_proposal_maxnms.py
import os

from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.models as models

from torchvision.ops import nms

# for detection models
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.structures import Boxes, Instances, ImageList

from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import AxisKind, AxisType, ImageFeatureValue, ImageValue, LogitsType, NeuralType
from nemo.utils.configuration_error import ConfigurationError
from nemo.utils.configuration_parsing import get_value_from_dictionary

# Create the config store instance.
cs = ConfigStore.instance()

D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__)) # Root of detectron2


@dataclass
class ImageEncoderConfig:
	backbone_type: str = "resnet50"
	output_size: Optional[int] = None
	return_feature_maps: bool = False
	model_stage: int = 3
	pretrained: bool = False
	cls_threshold: float = 0.9
	nms_threshold: float = 0.7
	model_path: str = None
	config_file: str = None
	min_boxes: int = 36
	max_boxes: int = 36


# Register the config.
cs.store(
	group="nemo.collections.cv.modules",
	name="ImageEncoder",
	node=ObjectConf(target="nemo.collections.cv.modules.ImageEncoder", params=ImageEncoderConfig()),
)


class ImageEncoder(NeuralModule):
	"""
	Neural Module implementing a general-usage image encoder.
	It encapsulates several models from TorchVision (VGG16, ResNet152 and DensNet121, naming a few).
	It also supports object detector models from Detectron2 (Faster-RCNN and Mask-RCNN)
	Offers two operation modes and can return: image embeddings vs feature maps.
	"""

	def __init__(
		self,
		backbone_type: str,
		output_size: Optional[int] = None,
		return_feature_maps: bool = False,
		model_stage: int = 3,
		pretrained: bool = False,
		cls_threshold: float = 0.9,
		nms_threshold: float = 0.7,
		model_path: str = None,
		config_file: str = None,
		min_boxes: int = 36,
		max_boxes: int = 36
	):
		"""
		Initializes the ``ImageEncoder`` model, creates the required "backbone".
		Args:
			backbone_type: Type of backbone (Handled options: VGG16 | DenseNet121 | ResNet152 | ResNet50)
			output_size: Size of the output layer (Optional, Default: None)
			return_feature_maps: Return mode: image embeddings vs feature maps (Default: False)
			model_stage: extract features at some specific layer, currently support Resnet101 (Default: 3)
			pretrained: Loads pretrained model (Default: False)
			cls_threshold: Regions with class scores less than threshold will be discarded (Default: 0.9)
			nms_threshold: Threshold for performing non-maximum suppression (Default: 0.7)
			model_path: Model path for use with detectron2 in case pretrained is set to True (Default: None)
			config_file: Path to a yaml config file for use with detectron2 (Default: None)
		"""
		super().__init__()

		# Get operation modes.
		self._return_feature_maps = return_feature_maps
		self._model_state = model_stage
		self._pretrained = pretrained
		self._model_path = model_path
		self._cls_threshold = cls_threshold
		self._nms_threshold = nms_threshold
		self._model_path = model_path
		self._config_file = config_file
		self._min_boxes = min_boxes
		self._max_boxes = max_boxes

		# Get model type.
		self._backbone_type = get_value_from_dictionary(
			backbone_type, "vgg16 | densenet121 | resnet152 | resnet50 | resnet101 | object-detector".split(" | ")
		)

		# Get output size (optional - not in feature_maps).
		self._output_size = output_size

		if self._backbone_type == 'vgg16':
			# Get VGG16
			self._model = models.vgg16(pretrained=pretrained)

			if self._return_feature_maps:
				# Use only the "feature encoder".
				self._model = self._model.features

				# Remember the output feature map dims.
				self._feature_map_height = 7
				self._feature_map_width = 7
				self._feature_map_depth = 512

			else:
				# Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
				self._model.classifier._modules['6'] = torch.nn.Linear(4096, self._output_size)

		elif self._backbone_type == 'densenet121':
			# Get densenet121
			self._model = models.densenet121(pretrained=pretrained)

			if self._return_feature_maps:
				raise ConfigurationError("'densenet121' doesn't support 'return_feature_maps' mode (yet)")

			# Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
			self._model.classifier = torch.nn.Linear(1024, self._output_size)

		elif self._backbone_type == 'resnet152':
			# Get resnet152
			self._model = models.resnet152(pretrained=pretrained)

			if self._return_feature_maps:
				# Get all modules exluding last (avgpool) and (fc)
				modules = list(self._model.children())[:-2]
				self._model = torch.nn.Sequential(*modules)

				# Remember the output feature map dims.
				self._feature_map_height = 7
				self._feature_map_width = 7
				self._feature_map_depth = 2048

			else:
				# Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
				self._model.fc = torch.nn.Linear(2048, self._output_size)

		elif self._backbone_type == 'resnet50':
			# Get resnet50
			self._model = models.resnet50(pretrained=pretrained)

			if self._return_feature_maps:
				# Get all modules exluding last (avgpool) and (fc)
				modules = list(self._model.children())[:-2]
				self._model = torch.nn.Sequential(*modules)

				# Remember the output feature map dims.
				self._feature_map_height = 7
				self._feature_map_width = 7
				self._feature_map_depth = 2048

			else:
				# Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
				self._model.fc = torch.nn.Linear(2048, self._output_size)

		elif self._backbone_type == 'resnet101':
			# Get resnet101
			if self._return_feature_maps:
				# Extract features from some specific layer
				resnet101 = getattr(torchvision.models, self._backbone_type)(pretrained=True)
				layers = [
					resnet101.conv1,
					resnet101.bn1,  
					resnet101.relu,
					resnet101.maxpool,
				]
				for i in range(self._model_stage):
					name = 'layer%d' % (i + 1)
					layers.append(getattr(resnet101, name))
				self._model = torch.nn.Sequential(*layers)

				# Remember the output feature map dims.
				self._feature_map_height = 7
				self._feature_map_width = 7
				self._feature_map_depth = 2048

			else:
				# Use the whole model, but "reshape"/reinstantiate the last layer ("FC6").
				self._model.fc = torch.nn.Linear(2048, self._output_size)

		elif self._backbone_type == 'object-detector':
			# Only support faster-rcnn and mask-rcnn
			cfg = get_cfg()
			cfg.merge_from_file(os.path.join(D2_ROOT, self._config_file))

			# modify config paramaters
			cfg.MODEL.WEIGHTS = self._model_path
			cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._cls_threshold
			cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self._nms_threshold

			# Load model  
			self._model = DefaultPredictor(cfg)

	@property
	def input_types(self):
		"""
		Returns definitions of module input ports.
		"""
		return {
			"inputs": NeuralType(
				axes=(
					AxisType(kind=AxisKind.Batch),
					AxisType(kind=AxisKind.Channel, size=3),
					AxisType(kind=AxisKind.Height, size=224),
					AxisType(kind=AxisKind.Width, size=224),
				),
				elements_type=ImageValue(),
				# TODO: actually encoders pretrained on ImageNet require special image normalization.
				# Probably this should be a new image type.
			)
		}

	@property
	def output_types(self):
		"""
		Returns definitions of module output ports.
		"""
		# Return neural type.
		if self._backbone_type != "object-detector":
			if self._return_feature_maps:
				return {
					"outputs": NeuralType(
						axes=(
							AxisType(kind=AxisKind.Batch),
							AxisType(kind=AxisKind.Channel, size=self._feature_map_depth),
							AxisType(kind=AxisKind.Height, size=self._feature_map_height),
							AxisType(kind=AxisKind.Width, size=self._feature_map_width),
						),
						elements_type=ImageFeatureValue(),
					)
				}
			else:
				return {
					"outputs": NeuralType(
						axes=(AxisType(kind=AxisKind.Batch), AxisType(kind=AxisKind.Any, size=self._output_size),),
						elements_type=LogitsType(),
					)
				}
		# Return neural types for object-detector
		else:
			return {
				# Note: num_boxes is in the range of (self._min_boxes, self._max_boxes)

				# classes of diffent objects in a batch of images (B x num_boxes)
				"classes": NeuralType(
					axes=(AxisType(kind=AxisKind.Batch), AxisType(kind=AxisKind.Any),),
					elements_type=PredictionsType(),
				),
				# prediction score of different objects in a batch of images (B x num_boxes)
				"scores": NeuralType(
					axes=(AxisType(kind=AxisKind.Batch), AxisType(kind=AxisKind.Any),),
					elements_type=PredictionsType(),
				),
				# bounding box of different objects in a batch of images (B x num_boxes x 4)
				"bboxes": NeuralType(
					axes=(
						AxisType(kind=AxisKind.Batch), 
						AxisType(kind=AxisKind.Any),
						AxisType(kind=AxisKind.Any, size=4),
					),
					elements_type=PredictionsType(),
				),
				# pooled features of different objects in a batch of images (B x num_boxes x 2048)
				"features": NeuralType(
					axes=(
						AxisType(kind=AxisKind.Batch), 
						AxisType(kind=AxisKind.Any),
						AxisType(kind=AxisKind.Any, size=2048),
					),
					elements_type=ImageFeatureValue(),
				)
			}

	def fast_rcnn_inference_single_image(
		self, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
	):
		"""
		Extract Fast-RCNN features from single images
		"""
		scores = scores[:, :-1]
		num_bbox_reg_classes = boxes.shape[1] // 4
		# Convert to Boxes to use the `clip` function ...
		boxes = Boxes(boxes.reshape(-1, 4))
		boxes.clip(image_shape)
		boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

		# Select max scores
		max_scores, max_classes = scores.max(1)       # R x C --> R
		num_objs = boxes.size(0)
		boxes = boxes.view(-1, 4)
		idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
		max_boxes = boxes[idxs]     # Select max boxes according to the max scores.

		# Apply NMS
		keep = nms(max_boxes, max_scores, nms_thresh)
		if topk_per_image >= 0:
			keep = keep[:topk_per_image]
		boxes, scores = max_boxes[keep], max_scores[keep]

		result = Instances(image_shape)
		result.pred_boxes = Boxes(boxes)
		result.scores = scores
		result.pred_classes = max_classes[keep]

		return result, keep

	@typecheck()
	def forward(self, inputs):
		"""
		Main forward pass of the model.
		Args:
			inputs: expected stream containing images [BATCH_SIZE x IMAGE_DEPTH x IMAGE_HEIGHT x IMAGE WIDTH]
		Returns:
			outpus: added stream containing outputs [BATCH_SIZE x OUTPUT_SIZE]
				OR [BATCH_SIZE x OUTPUT_DEPTH x OUTPUT_HEIGHT x OUTPUT_WIDTH]

			# in case of using object detector model, we return a list of (classes, scores, bounding_box and features)
		"""
		# print("({}): input shape: {}, device: {}\n".format(self._backbone_type, inputs.shape, inputs.device))
		if self._backbone_type == 'object-detector':
			# convert input tensor into ImageList
			images = ImageList.from_tensors(inputs.unbind(), self._model.model.backbone.size_divisibility)

			# run backbone network to extract proposal of features
			features = self._model.model.backbone(images.tensor)
			proposals, _ = self._model.model.proposal_generator(images, features, None)

			# Run RoI head for each proposal (RoI Pooling + Res5)
			proposal_boxes = [x.proposal_boxes for x in proposals]
			features = [features[f] for f in self._model.model.roi_heads.in_features]

			box_features = self._model.model.roi_heads._shared_roi_transform(
				features, proposal_boxes
			)

			feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1
	   
			# Predict classes and boxes for each proposal.
			pred_class_logits, pred_proposal_deltas = self._model.model.roi_heads.box_predictor(feature_pooled)
			rcnn_outputs = FastRCNNOutputs(
				self._model.model.roi_heads.box2box_transform,
				pred_class_logits,
				pred_proposal_deltas,
				proposals,
				self._model.model.roi_heads.smooth_l1_beta,
			)

			# Fixed-number NMS
			instances_list, ids_list = [], []
			probs_list = rcnn_outputs.predict_probs()
			boxes_list = rcnn_outputs.predict_boxes()
			for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
				for nms_thresh in np.arange(0.3, 1.0, 0.1):
					instances, ids = fast_rcnn_inference_single_image(
						boxes, probs, image_size,
						score_thresh=self._cls_threshold,
						nms_thresh=self._nms_threshold, topk_per_image=self._max_boxes
					)
					if len(ids) >= self._min_boxes:
						break
				instances_list.append(instances)
				ids_list.append(ids)

				# Post processing for features
			features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
			roi_features_list = []
			for ids, features in zip(ids_list, features_list):
				roi_features_list.append(features[ids].detach())
	   
			# Post processing for bounding boxes (rescale to raw_image)
			raw_instances_list = []
			for instances, input_per_image, image_size in zip(
				instances_list, inputs, images.image_sizes
				):
					height = input_per_image.get("height", image_size[0])
					width = input_per_image.get("width", image_size[1])
					raw_instances = detector_postprocess(instances, height, width)
					raw_instances_list.append(raw_instances)

			# batch the features, num_boxes in the ranges of (self._min_boxes, self._max_boxes)
			pred_classes = torch.stack(instances.pred_classes for instances in raw_instances_list).type(torch.LongTensor) # (B x num_boxes)
			pred_scores = torch.stack(instances.scores for instances in raw_instances_list).type(torch.FloatTensor) # (B x num_boxes)
			bboxes = torch.stack(instances.pred_boxes.tensor for instances in raw_instances_list).type(torch.FloatTensor) # (B x num_boxes x 4)
			features = torch.stack(feature for feature in roi_features_list).type(torch.FloatTensor) # (B x num_boxes x 2048)

			# return features
			return (
				pred_classes,
				pred_scores,
				bboxes,
				features,
			)
		else:    
			outputs = self._model(inputs)

		# Add outputs to datadict.
		return outputs

	def save_to(self, save_path: str):
		"""
		Not implemented.
		Args:
			save_path (str): path to save serialization.
		"""
		pass

	@classmethod
	def restore_from(cls, restore_path: str):
		"""
		Not implemented.
		Args:
			restore_path (str): path to serialization
		"""
		pass
