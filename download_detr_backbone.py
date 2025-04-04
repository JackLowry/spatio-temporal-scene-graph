import torch
from transformers import PretrainedConfig
from transformers.models.detr.modeling_detr import DetrTimmConvEncoder, DetrConvModel, build_position_encoding

BACKBONE_DIRPATH = "/home/jack/research/scene_graph/spatio_temporal_sg/backbone"

config = PretrainedConfig.from_pretrained("facebook/detr-resnet-50")
backbone = DetrTimmConvEncoder(config.backbone, config.dilation)
position_embeddings = build_position_encoding(config)
backbone = DetrConvModel(backbone, position_embeddings)
torch.save(backbone.state_dict(), f"{BACKBONE_DIRPATH}/resnet50.pt")