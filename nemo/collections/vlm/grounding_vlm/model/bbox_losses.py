import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple

def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized intersection over union between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) containing N boxes represented as (x1, y1, x2, y2)
        boxes2: Tensor of shape (N, 4) containing N boxes represented as (x1, y1, x2, y2)
        
    Returns:
        giou: Tensor of shape (N,) containing the generalized IoU for each pair of boxes
    """
    # Get box areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Get intersection coordinates
    lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # left-top
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # right-bottom
    
    # Calculate intersection area
    wh = (rb - lt).clamp(min=0)  # width-height
    intersection = wh[..., 0] * wh[..., 1]
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    # Get enclosing box coordinates
    lt_c = torch.min(boxes1[..., :2], boxes2[..., :2])  # left-top
    rb_c = torch.max(boxes1[..., 2:], boxes2[..., 2:])  # right-bottom
    
    # Calculate enclosing box area
    wh_c = (rb_c - lt_c).clamp(min=0)  # width-height
    enclosing_area = wh_c[..., 0] * wh_c[..., 1]
    
    # Calculate GIoU
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou, iou


class HungarianMatchingLoss(nn.Module):
    """
    Implements Hungarian matching loss for object detection using GIoU.
    Assumes equal number of predictions and ground truth boxes.
    """
    def __init__(self, cost_giou: float = 1.0):
        """
        Args:
            cost_giou: Weight for GIoU cost in matching
            cost_bbox: Weight for L1 box coordinate cost in matching
        """
        super().__init__()
        self.cost_giou = cost_giou

    def compute_matching_cost_matrix(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cost matrix for matching predicted and ground truth boxes.
        
        Args:
            pred_boxes: Predicted boxes (batch_size, num_queries, 4)
            gt_boxes: Ground truth boxes (batch_size, num_queries, 4)
            valid_mask: Boolean mask for valid boxes (batch_size, num_queries)
            
        Returns:
            cost_matrix: Cost matrix of shape (batch_size, num_queries, num_queries)
            iou_matrix: IoU matrix of shape (batch_size, num_queries, num_queries)
        """
        batch_size, num_queries, _ = pred_boxes.shape
        
        # Compute costs for each pair in the batch
        cost_giou = []
        iou_matrix = []
        
        for b in range(batch_size):
            # Get boxes for current batch
            valid = torch.nonzero(valid_mask[b], as_tuple=False)
            num_valid_queries = valid.shape[0]
            if num_valid_queries == 0:
                iou_matrix.append(None)
                cost_giou.append(None)
                continue

            pred_b = pred_boxes[b, valid]  # (num_valid_queries, 4)
            gt_b = gt_boxes[b, valid]  # (num_valid_queries, 4)
            
            # Compute pairwise GIoU and IoU
            giou_b, iou_b = box_giou(pred_b.unsqueeze(1).expand(-1, num_valid_queries, -1),
                           gt_b.unsqueeze(0).expand(num_valid_queries, -1, -1))
            # Store IoU for return
            iou_matrix.append(iou_b)
            # Convert to cost (negative GIoU)
            cost_giou.append(-giou_b)
        
        return cost_giou, iou_matrix

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hungarian matching loss between predicted and ground truth boxes.
        
        Args:
            pred_boxes: Predicted boxes (batch_size, num_queries, 4)
            gt_boxes: Ground truth boxes (batch_size, num_queries, 4)
            valid_mask: Boolean mask for valid boxes (batch_size, num_queries)
            
        Returns:
            loss: Scalar loss value
            matched_ious: IoU values for matched pairs (batch_size, num_queries)
                         with -1 for invalid pairs
        """
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device
        
        # Compute cost matrix and IoU matrix
        cost_matrix, iou_matrix = self.compute_matching_cost_matrix(
            pred_boxes, gt_boxes, valid_mask
        )
        
        # Initialize total loss and IoU tracking
        total_loss = torch.tensor(0., device=device)
        total_valid_pairs = 0
        matched_ious = torch.full((batch_size, pred_boxes.shape[1]), -1., device=device)
        
        # Process each batch independently
        for b in range(batch_size):
            # Get valid boxes for this batch
            valid_count = valid_mask[b].sum()
            
            if valid_count == 0:
                continue
                
            # Get cost matrix for current batch
            cost_b = cost_matrix[b]
            
            # Use Hungarian algorithm to find optimal matching
            # Note: We use -cost because linear_sum_assignment minimizes cost
            matched_indices = linear_sum_assignment_with_inf(-cost_b.detach().cpu().numpy())
            matched_indices = torch.as_tensor(matched_indices, dtype=torch.long, device=device)
            
            # Compute loss and store IoUs for matched pairs
            pred_idx, gt_idx = matched_indices
            matched_cost = cost_b[pred_idx, gt_idx].sum()
            
            # Store IoUs for valid matched pairs
            # valid_pairs = valid_mask[b, pred_idx] & valid_mask[b, gt_idx]
            # matched_ious[b, pred_idx[valid_pairs]] = iou_matrix[b, pred_idx[valid_pairs], gt_idx[valid_pairs]]
            matched_ious[b, :valid_count, :valid_count] = iou_matrix[b]
            
            # Add to total loss (only for valid pairs)
            total_loss += matched_cost
            total_valid_pairs += valid_count.item()
        
        # Return average loss and matched IoUs
        avg_loss = total_loss / max(total_valid_pairs, 1)
        return avg_loss, matched_ious


def linear_sum_assignment_with_inf(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for scipy.optimize.linear_sum_assignment that handles infinite costs.
    """
    from scipy.optimize import linear_sum_assignment
    
    # Replace inf with a very large number
    cost_matrix = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=-1e5)
    
    return linear_sum_assignment(cost_matrix) 