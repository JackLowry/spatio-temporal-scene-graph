# Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

import argparse
import os
from glob import glob
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from torch.utils.data import DataLoader
import wandb

from data.isaac_detr import IsaacLabDetrDataset

# from data.open_image import OIDetection
# from data.visual_genome import VGDetection

from data.temporal_isaac_detr import TemporalIsaacLabDetrDataset
from egtr.lib.fpn import box_utils
from egtr.lib.pytorch_misc import argsort_desc, intersect_2d
from temporal_egtr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentor,
    TemporalDeformableDetrForObjectDetection,
)
from util.misc import use_deterministic_algorithms
from util.box_ops import box_cxcywh_to_xyxy
from omegaconf import DictConfig, OmegaConf
import hydra

from visualization import draw_graph, draw_sequence_boxes, draw_sequence_graphs

seed_everything(42, workers=True)

class LogPredictionSamplesCallback(Callback):
    def __init__(self, logger: WandbLogger):
        super().__init__()
        self.logger = logger
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 1 sample image predictions from the first batch
        if batch_idx == 0:

            # pl_module.model.sequence_matcher(node_latents.reshape(batch_size, seq_len, num_objects, -1), 
            #             edge_latents.reshape(batch_size, seq_len, num_edges, -1),
            #             node_network_mask.reshape(batch_size, seq_len, num_objects),
            #             edge_idx_to_node_idxs.reshape(batch_size, seq_len, num_edges, -1))
            
            outputs.logits = outputs.logits[batch_idx]
            outputs.pred_boxes = outputs.pred_boxes[batch_idx]
            orig_target_sizes = batch["orig_img"][batch_idx].shape[:-1]
            orig_target_sizes = torch.Tensor(list(orig_target_sizes[1:])).to(outputs.logits.device).repeat(orig_target_sizes[batch_idx], 1)
            processed_outs = pl_module.feature_extractor.post_process(
                outputs, orig_target_sizes
            )

            

            imgs = []
            seq_boxes = []
            seq_edge_matches =  []
            for seq_id in range(outputs.logits.shape[0]):
                boxes = processed_outs[seq_id]["boxes"]
                labels = processed_outs[seq_id]["labels"]
                boxes = boxes[labels == 0]

                img = batch["orig_img"].clone()
                img = img[batch_idx, seq_id]
                # img = (img - img.min())/img.max()
                img = img*255
                img = img.to(torch.uint8)

                imgs.append(img.cpu().numpy())
                seq_boxes.append(boxes.cpu().numpy())

                pred_logits = outputs["logits"][seq_id]
                obj_scores, pred_classes = torch.max(
                    pred_logits.softmax(-1), -1
                )
                sub_ob_scores = torch.outer(obj_scores, obj_scores)
                sub_ob_scores[
                    torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
                ] = 0.0  # prevent self-connection


                pred_rel = torch.clamp(outputs["pred_rel"][batch_idx, seq_id], 0.0, 1.0)
                pred_connectivity = torch.clamp(outputs["pred_connectivity"][batch_idx, seq_id], 0.0, 1.0)
                pred_rel = torch.mul(pred_rel, pred_connectivity)
            
                triplet_scores = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))
                pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                    :100, :
                ]  # [pred_rels, 3(s,o,p)]
                                
                rel_scores = (
                    pred_rel.cpu()
                    .clone()
                    .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
                )  # [pred_rels]
                score_thresholds = [0.1, 0.25, 0.5, 0.75]
                target_rels = batch["labels"][batch_idx][seq_id]["rel"]
                num_threshold_labels = []
                num_matching_labels = []
                for thresh in score_thresholds:
                    pred_rel_inds_thresh = pred_rel_inds[rel_scores > thresh]
                    if len(target_rels) == 0 or len(pred_rel_inds_thresh) == 0:
                        num_matches = 0
                    else:
                        num_matches = intersect_2d(pred_rel_inds_thresh, target_rels.cpu().numpy()).any(1).sum()

                    num_threshold_labels.append(pred_rel_inds_thresh.shape[0])
                    num_matching_labels.append(num_matches)
                seq_edge_matches.append((num_threshold_labels, num_matching_labels, score_thresholds))

            img = draw_sequence_graphs(imgs, seq_boxes, seq_edge_matches)

            # Option 1: log images with `WandbLogger.log_image`
            self.logger.log_metrics({"pred_boxes": img})

def collate_fn(batch, feature_extractor):

    batch_processed = {}
    batch_processed["pixel_values"] = []
    batch_processed["pixel_mask"] = []
    batch_processed["labels"] = []
    batch_processed["orig_img"] = []
    pixel_values = []
    orig_img = []

    seq_len = len(batch[0][0][0])
    batch_size = len(batch)


    for sample in batch:
        batch_processed["labels"].append(sample[1])
        batch_processed["orig_img"].append(torch.stack(sample[0][1]))
        for seq_id in range(seq_len):
            pixel_values.append(sample[0][0][seq_id])
        
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )        

    batch_processed["orig_img"] = torch.stack(batch_processed["orig_img"])

    batch_processed["pixel_values"] = encoding["pixel_values"].reshape(batch_size, seq_len, *encoding["pixel_values"].shape[1:])
    batch_processed["pixel_mask"] = encoding["pixel_mask"].reshape(batch_size, seq_len, *encoding["pixel_mask"].shape[1:])

    return batch_processed


class TemporalDetr(pl.LightningModule):
    def __init__(
        self,
        main_trained,
        id2label,
        architecture,
        feature_extractor,
        num_rel_labels,
        fg_matrix,
        args
    ):
        super().__init__()
        # replace COCO classification head with custom head
        config = DeformableDetrConfig.from_pretrained(architecture)
        for k, v in args.items():
            setattr(config, k, v)
        config.num_rel_labels = num_rel_labels
        config.num_labels = max(id2label.keys()) + 1
        config.output_attention_states = False
        self.model = TemporalDeformableDetrForObjectDetection(config=config, fg_matrix=fg_matrix)
        self.model.model.backbone.load_state_dict(
            torch.load(f"{config.backbone_dirpath}/{config.backbone}.pt")
        )

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = config.lr
        self.lr_backbone = config.lr_backbone
        self.weight_decay = config.weight_decay
        self.feature_extractor = feature_extractor
        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        return outputs


    def common_step(self, batch, batch_idx, ret_outputs=False):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        if ret_outputs:
            return outputs
        del outputs
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "training_loss": loss.item(),
        }
        log_dict.update({f"training_{k}": v.item() for k, v in loss_dict.items()})
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, ret_outputs=True)
        loss = outputs.loss        
        outputs.loss_dict["loss"] = loss

        return outputs

    def validation_epoch_end(self, outputs):

        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        for k in outputs[0].loss_dict.keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x.loss_dict[k] for x in outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # get the inputs
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [
            {k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in labels], dim=0
        )
        results = self.feature_extractor.post_process(
            outputs, orig_target_sizes
        )  # convert outputs of model to COCO api
        res = {
            target["image_id"].item(): output for target, output in zip(labels, results)
        }


    # def test_epoch_end(self, outputs):


    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params)) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in diff_lr_params) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return self.train_dataloader_obj

    def val_dataloader(self):
        return self.val_dataloader_obj


config_name = "default.yaml"
@hydra.main(version_base=None, config_path="conf/temporal_detr", config_name=config_name)
def main(config: DictConfig) -> None:
    # torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Path
    args = hydra.utils.instantiate(config)

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.img_size, max_size=args.max_size
    )
    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentor.from_pretrained(
            args.architecture, size=args.img_size, max_size=args.max_size
        )
    )

    dataset = TemporalIsaacLabDetrDataset(
        root_dir=args.data_path,
        feature_extractor=feature_extractor_train,
    )

    # if config.debug:
    #     test_dataset_size = 5990
    #     data_len = len(dataset) - test_dataset_size
    #     train_size = int(data_len*.9) 
    #     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, data_len - train_size, test_dataset_size])
    # else:
    data_len = len(dataset)
    train_size = int(data_len*.9)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, data_len - train_size])

    fg_matrix = TemporalIsaacLabDetrDataset.get_statistics(train_dataset)

    id2label = {0: "Object", 1: "No Object"}
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor_train),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Logger setting
    save_dir = (
        f"{args.output_path}/pretrained_detr__{args.architecture.replace('/', '__')}"
    )
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr__{args.lr_backbone}_{args.lr}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    

    # Trainer setting
    logger = WandbLogger(save_dir=save_dir, name=name, version=version,
                         project="sggen-temporal-detr",
                         )

    ckpt_path = None

    # Module
    module = TemporalDetr(
        main_trained="",
        id2label=id2label,
        architecture=args.architecture,
        feature_extractor=feature_extractor,
        num_rel_labels=dataset.num_relationship_labels,
        fg_matrix=fg_matrix,
        args=args,
    )

    module.train_dataloader_obj = train_dataloader
    module.val_dataloader_obj = val_dataloader

    # Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        filename="{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
        save_on_train_epoch_end = False,
        every_n_epochs=1

    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=args.patience, verbose=True, mode="min"
    )
    
    log_pred_callback = LogPredictionSamplesCallback(logger)

    # Train
    trainer = None
    if not args.skip_train:
        # Main training

        print(logger.save_dir)
        print(ckpt_path)
        # quit()

            # Training
        trainer = Trainer(
            precision=args.precision,
            logger=logger,
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            gradient_clip_val=args.gradient_clip_val,
            strategy=DDPStrategy(find_unused_parameters=True),
            callbacks=[checkpoint_callback, early_stop_callback, log_pred_callback],
            accumulate_grad_batches=args.accumulate,
            log_every_n_steps=10,
        )
        use_deterministic_algorithms()
        if trainer.is_global_zero:
            print("### Main training")
        trainer.fit(module, ckpt_path=ckpt_path)

        try:
            os.chmod(logger.save_dir, 0o0777)
        except PermissionError as e:
            print(e)

        # load best model & save best model as pytorch_model.bin
        ckpt_path = checkpoint_callback.best_model_path
        print(ckpt_path)

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)
        if trainer.is_global_zero:
            module.model.save_pretrained(logger.save_dir)

        # if trainer is not None:
        #     torch.distributed.destroy_process_group()
        #     try:
        #         os.chmod(logger.save_dir, 0o0777)
        #     except PermissionError as e:
        #         print(e)

    # # Evaluation
    if args.eval_when_train_end and (trainer is None or trainer.is_global_zero):
        # if args.skip_train and args.finetune:
        #     logger = TensorBoardLogger(
        #         save_dir, name=f"{name}__finetune", version=version
        #     )

        # Load best model
        ckpt_path = checkpoint_callback.best_model_path
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)

        # Eval
        trainer = Trainer(
            precision=args.precision, logger=logger, gpus=1, max_epochs=-1
        )

        test_dataloader = val_dataloader
        if trainer.is_global_zero:
            print("### Evaluation")
        trainer.test(module, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()