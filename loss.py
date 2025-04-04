# modified from
# https://github.com/liyi14/STOW/blob/c1b0e9ef0f0d10ccd5573e09c222970c2b704af5/stow/modeling/criterion_stow.py#L113
import torch.nn as nn
import torch

from .model_util import (
    generalized_box_iou,
    nested_tensor_from_tensor_list,
)

# def cosine_distance(x1, x2):
#     return 0.5*(1-nn.functional.cosine_similarity(x1,x2, dim=-1),)
    
# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class TemporalSceneGraphGenerationLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        num_object_queries,
        num_classes,
        num_rel_labels,
        eos_coef,
        losses,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        model_training,
        focal_alpha,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
    ):
        """
        Create the criterion.

        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-6config.num_rel_labels269223"

        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_object_queries = num_object_queries
        self.num_classes = num_classes
        self.num_rel_labels = num_rel_labels
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.rel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.rel_sample_negatives = rel_sample_negatives
        self.rel_sample_nonmatching = rel_sample_nonmatching
        self.model_training = model_training
        self.focal_alpha = focal_alpha
        self.rel_sample_negatives_largest = rel_sample_negatives_largest
        self.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest
        self.nonmatching_cost = (
            -torch.log(torch.tensor(1e-8)) * matcher.class_cost
            + 4 * matcher.bbox_cost
            + 2 * matcher.giou_cost
            - torch.log(torch.tensor((1.0 / smoothing) - 1.0))
        )  # set minimum bipartite matching costs for nonmatched object queries
        self.connectivity_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def loss_labels(self, outputs, targets, indices, matching_costs, num_boxes):
        return self._loss_labels_focal(
            outputs, targets, indices, matching_costs, num_boxes
        )

    def _loss_labels_focal(
        self, outputs, targets, indices, matching_costs, num_boxes, log=True
    ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise ValueError("No logits were found in the outputs")

        source_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["class_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                source_logits.shape[0],
                source_logits.shape[1],
                source_logits.shape[2] + 1,
            ],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                source_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["class_labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    @torch.no_grad()
    def loss_uncertainty(self, outputs, targets, indices, matching_costs, num_boxes):
        nonzero_uncertainty_list = []
        for target, index, matching_cost in zip(targets, indices, matching_costs):
            nonzero_index = target["rel"][index[1], :, :][:, index[1], :].nonzero()
            uncertainty = matching_cost.sigmoid()
            nonzero_uncertainty_list.append(
                uncertainty[nonzero_index[:, 0]] * uncertainty[nonzero_index[:, 1]]
            )
        losses = {"uncertainty": torch.cat(nonzero_uncertainty_list).mean()}
        return losses

    def loss_boxes(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(src_boxes),
                center_to_corners_format(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        assert "pred_masks" in outputs, "No predicted masks found in outputs"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = nn.functional.interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_relations(self, outputs, targets, indices, matching_costs, num_boxes):
        losses = []
        connect_losses = []
        for i, ((src_index, target_index), target, matching_cost) in enumerate(
            zip(indices, targets, matching_costs)
        ):
            # Only calculate relation losses for matched objects (num_object_queries * num_object_queries -> num_obj * num_obj)
            full_index = torch.arange(self.num_object_queries)
            uniques, counts = torch.cat([full_index, src_index]).unique(
                return_counts=True
            )
            full_src_index = torch.cat([src_index, uniques[counts == 1]])
            full_target_index = torch.cat(
                [target_index, torch.arange(len(target_index), self.num_object_queries)]
            )
            full_matching_cost = torch.cat(
                [
                    matching_cost,
                    torch.full(
                        (self.num_object_queries - len(matching_cost),),
                        self.nonmatching_cost,
                        device=matching_cost.device,
                    ),
                ]
            )

            pred_rel = outputs["pred_rel"][i, full_src_index][
                :, full_src_index
            ]  # [num_obj_queries, num_obj_queries, config.num_rel_labels]
            target_rel = target["rel"][full_target_index][
                :, full_target_index
            ]  # [num_obj_queries, num_obj_queries, config.num_rel_labels]

            rel_index = torch.nonzero(target_rel)
            target_connect = torch.zeros(
                target_rel.shape[0], target_rel.shape[1], 1, device=target_rel.device
            )
            target_connect[rel_index[:, 0], rel_index[:, 1]] = 1
            pred_connectivity = outputs["pred_connectivity"][i, full_src_index][
                :, full_src_index
            ]
            loss = self.connectivity_loss(pred_connectivity, target_connect)
            connect_losses.append(loss)

            if self.model_training:
                loss = self._loss_relations(
                    pred_rel,
                    target_rel,
                    full_matching_cost,
                    self.rel_sample_negatives,
                    self.rel_sample_nonmatching,
                )
            else:
                loss = self._loss_relations(
                    pred_rel, target_rel, full_matching_cost, None, None
                )
            losses.append(loss)
        losses = {
            "loss_rel": torch.cat(losses).mean(),
            "loss_connectivity": torch.stack(connect_losses).mean(),
        }
        return losses

    def _loss_relations(
        self,
        pred_rel,
        target_rel,
        matching_cost,
        rel_sample_negatives,
        rel_sample_nonmatching,
    ):
        if (rel_sample_negatives is None) and (rel_sample_nonmatching is None):
            weight = 1.0 - matching_cost.sigmoid()
            weight = torch.outer(weight, weight)
            target_rel = target_rel * weight.unsqueeze(-1)
            loss = self.rel_loss(pred_rel, target_rel).mean(-1).reshape(-1)
        else:
            matched = matching_cost != self.nonmatching_cost
            num_target_objects = sum(matched)

            true_indices = target_rel[
                :num_target_objects, :num_target_objects, :
            ].nonzero()
            false_indices = (
                target_rel[:num_target_objects, :num_target_objects, :] != 1.0
            ).nonzero()
            nonmatching_indices = (
                torch.outer(matched, matched)
                .unsqueeze(-1)
                .repeat(1, 1, self.num_rel_labels)
                != True
            ).nonzero()

            num_target_relations = len(true_indices)
            if rel_sample_negatives is not None:
                if rel_sample_negatives == 0 or num_target_relations == 0:
                    sampled_idx = []
                else:
                    if self.rel_sample_negatives_largest:
                        false_sample_scores = pred_rel[
                            false_indices[:, 0],
                            false_indices[:, 1],
                            false_indices[:, 2],
                        ]
                        sampled_idx = torch.topk(
                            false_sample_scores,
                            min(
                                num_target_relations * rel_sample_negatives,
                                false_sample_scores.shape[0],
                            ),
                            largest=True,
                        )[1]
                    else:
                        sampled_idx = torch.tensor(
                            random.sample(
                                range(false_indices.size(0)),
                                min(
                                    num_target_relations * rel_sample_negatives,
                                    false_indices.size(0),
                                ),
                            ),
                            device=false_indices.device,
                        )
                false_indices = false_indices[sampled_idx]
            if rel_sample_nonmatching is not None:
                if rel_sample_nonmatching == 0 or num_target_relations == 0:
                    sampled_idx = []
                else:
                    if self.rel_sample_nonmatching_largest:
                        nonmatching_sample_scores = pred_rel[
                            nonmatching_indices[:, 0],
                            nonmatching_indices[:, 1],
                            nonmatching_indices[:, 2],
                        ]
                        sampled_idx = torch.topk(
                            nonmatching_sample_scores,
                            min(
                                num_target_relations * rel_sample_nonmatching,
                                nonmatching_indices.size(0),
                            ),
                            largest=True,
                        )[1]
                    else:
                        sampled_idx = torch.tensor(
                            random.sample(
                                range(nonmatching_indices.size(0)),
                                min(
                                    num_target_relations * rel_sample_nonmatching,
                                    nonmatching_indices.size(0),
                                ),
                            ),
                            device=nonmatching_indices.device,
                        )
                nonmatching_indices = nonmatching_indices[sampled_idx]

            relation_indices = torch.cat(
                [true_indices, false_indices, nonmatching_indices]
            )
            pred_rel = pred_rel[
                relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]
            ]
            target_rel = target_rel[
                relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]
            ]

            weight = 1.0 - matching_cost.sigmoid()
            weight = weight[relation_indices[:, 0]] * weight[relation_indices[:, 1]]
            target_rel = target_rel * weight
            loss = self.rel_loss(pred_rel, target_rel)
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, matching_costs, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "relations": self.loss_relations,
            "uncertainty": self.loss_uncertainty,
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, matching_costs, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "auxiliary_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, matching_costs = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, matching_costs, num_boxes
                )
            )

        if "pred_rels" in outputs:
            for pred_rel in outputs["pred_rels"]:
                outputs["pred_rel"] = pred_rel
                _loss_dict = self.loss_relations(
                    outputs, targets, indices, matching_costs, num_boxes
                )
                losses["loss_rel"] += _loss_dict["loss_rel"]

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):

                indices, matching_costs = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss in ["masks", "relations", "uncertainty"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(
                        loss,
                        auxiliary_outputs,
                        targets,
                        indices,
                        matching_costs,
                        num_boxes,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices, matching_costs = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ["masks", "relations", "uncertainty"]:
                    continue
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, matching_costs, num_boxes
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

class ContrastiveLoss():
    def __init__(self, margin=1.0):
        self.margin = margin
    
    def get_loss(self, output1, output2, matching_label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        #matching pairs
        loss_contrastive_matching  = torch.mean(matching_label * torch.pow(euclidean_distance, 2))
        #non-matching pairs
        loss_contrastive_nonmatching = torch.mean(torch.bitwise_not(matching_label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        #number_hard_negative -- from https://groups.csail.mit.edu/robotics-center/public_papers/Florence19.pdf
        # only scale the non-matching count by the amount which are truly hard negative, 
        # aka they are truly close when they shouldn't be (within margin distance).
        # if samples are actually far apart, they should be scaled less
        hard_negatives = torch.sum(torch.bitwise_and(torch.bitwise_not(matching_label),
                                                     (self.margin - euclidean_distance) > 0))
        
        loss_contrastive_nonmatching = 1/(max(1, hard_negatives))*loss_contrastive_nonmatching


        if torch.isnan(loss_contrastive_nonmatching) or torch.isnan(loss_contrastive_matching):
            print("no")
                                      
        return loss_contrastive_matching + loss_contrastive_nonmatching
    
def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes