import torch
from torch import nn
import torchvision.ops as ops
from src.model.components import ResNet101Backbone

class MaskRCNN(nn.Module):
    def __init__(self, num_classes, backbone_pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet101Backbone(pretrained=backbone_pretrained)
        self.rpn = ops.RPNHead(self.backbone.layer4.out_channels, 9) 
        self.roi_align = ops.RoIAlign(output_size=(7, 7), spatial_scale=1/32, sampling_ratio=2)
        self.mask_roi_align = ops.RoIAlign(output_size=(14, 14), spatial_scale=1/32, sampling_ratio=2) 
        self.mask_head = MaskHead(num_classes, in_channels=self.backbone.layer4.out_channels)
        self.box_predictor = nn.Linear(self.backbone.layer4.out_channels * 7 * 7, num_classes * 4)
        self.class_predictor = nn.Linear(self.backbone.layer4.out_channels * 7 * 7, num_classes)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        # Backbone
        features = self.backbone(images)
        features = features[0] # Use the last feature map

        # RPN
        rpn_output = self.rpn(features)
        proposals, proposal_losses = rpn_output[0], rpn_output[1]

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should not be None")
            # Assign ground truth boxes and labels to anchors
            anchors = self.rpn.anchor_generator(images, features)
            objectness_losses, rpn_box_losses = ops.rpn.compute_loss(
                proposals, anchors, targets
            )
            proposal_losses = objectness_losses + rpn_box_losses

            # Select positive and negative samples for ROI head
            sampled_proposals, matched_labels, matched_boxes = ops.roi_pool(
                proposals, targets, self.roi_align.output_size, self.roi_align.spatial_scale
            )
            if sampled_proposals is None:
                return {}, proposal_losses.mean()

            # ROI Align for box head
            roi_features = self.roi_align(features, sampled_proposals)
            roi_features = roi_features.flatten(start_dim=1)
            box_regression = self.box_predictor(roi_features)
            class_logits = self.class_predictor(roi_features)

            # ROI Align for mask head
            mask_roi_features = self.mask_roi_align(features, sampled_proposals)
            mask_logits = self.mask_head(mask_roi_features)

            # Calculate ROI head losses
            box_losses, mask_losses = ops.roi_heads.maskrcnn_loss(
                class_logits, box_regression, mask_logits, matched_labels, matched_boxes, targets
            )

            losses = {
                "loss_rpn": proposal_losses.mean(),
                "loss_classifier": box_losses.mean(),
                "loss_mask": mask_losses.mean(),
            }
            return losses, proposal_losses.mean() + box_losses.mean() + mask_losses.mean()
        else:
            # Inference mode
            roi_features = self.roi_align(features, proposals)
            roi_features = roi_features.flatten(start_dim=1)
            box_regression = self.box_predictor(roi_features)
            class_logits = self.class_predictor(roi_features)
            mask_roi_features = self.mask_roi_align(features, proposals)
            mask_logits = self.mask_head(mask_roi_features)

            detections = ops.roi_heads.multiclass_nms(
                proposals,
                class_logits,
                box_regression,
                score_threshold=0.05,
                nms_threshold=0.5,
                detections_per_img=100,
            )

            masks_probs = ops.roi_heads.maskrcnn_inference(mask_logits, detections, mask_threshold=0.5)

            result = []
            for i in range(len(detections)):
                result.append(
                    {
                        "boxes": detections[i][:, :4],
                        "labels": detections[i][:, 4].int(),
                        "scores": detections[i][:, 5],
                        "masks": masks_probs[i],
                    }
                )
            return result