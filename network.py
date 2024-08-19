import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.ops as ops

class BiLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, use_gpu, batch_size, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2*self.num_layers, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2*self.num_layers, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2*self.num_layers, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2*self.num_layers, self.batch_size, self.hidden_dim)))
        
    def forward(self, x):
        
        y, self.hidden = self.lstm(x, None)

        return y

class RGBFeatureExtractor(nn.Module):

    def __init__(self):
        super(RGBFeatureExtractor, self).__init__()

        self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    def forward(self, x):
        features = self.extractor.get_intermediate_layers(x, 1)[0]
        return features.reshape(x.shape[0], 16, 16, -1)
    
class LatentDecoderHead(nn.Module):

    def __init__(self, latent_size, layer_dimensions):
        super(LatentDecoderHead, self).__init__()

        self.head = ops.MLP(latent_size, layer_dimensions)

    def forward(self, x):

        return self.head(x)

class SceneGraphGenerator(nn.Module):

    def __init__(self, num_nodes):
        super(SceneGraphGenerator, self).__init__()

        self.extractor = RGBFeatureExtractor()

        self.num_nodes = num_nodes

        self.num_edges = num_nodes*(num_nodes-1)

        self.use_gpu = True
        self.batch_size = 16
        self.embedding_dim = 768

        self.node_encoder = BiLSTM(25*self.embedding_dim, 512, self.use_gpu, self.batch_size)

        self.edge_encoder = BiLSTM(49*self.embedding_dim, 512, self.use_gpu, self.batch_size)

        self.node_latent_downscaler = ops.MLP(1024, [256, 64])



    def forward(self, image, object_bounding_boxes, union_bounding_boxes):

        image_features = self.extractor(image).permute((0, 3, 1, 2))

        batch_size = image.shape[0]
        num_objects = object_bounding_boxes[0].shape[0]
        num_edges = union_bounding_boxes[0].shape[0]


        scale_factor = image_features.shape[-1] / float(image.shape[-1])

        is_all_zeros = torch.all(object_bounding_boxes.flatten(0, 1) == 0, dim=1)

        object_bounding_boxes = torch.split(object_bounding_boxes, 1)

        object_bounding_boxes = [o.squeeze() for o in object_bounding_boxes]
        object_features = ops.roi_align(image_features, object_bounding_boxes, output_size=(5,5), spatial_scale=scale_factor)
        object_features = object_features.reshape((batch_size, num_objects, -1))

        objects_features_flattened = object_features.flatten(0, 1)
        objects_features_flattened[is_all_zeros] = 0
        object_features = objects_features_flattened.reshape(object_features.shape)

        union_bounding_boxes = torch.split(union_bounding_boxes, 1)
        union_bounding_boxes = [o.squeeze() for o in union_bounding_boxes]
        edge_features = ops.roi_align(image_features, union_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor)
        edge_features = edge_features.reshape(batch_size, num_edges, -1)

        node_latents = self.node_encoder(object_features)
        
        # node_latents_downscaled = self.node_latent_downscaler(node_latents)

        edge_latents = self.edge_encoder(edge_features)

        return node_latents, edge_latents
    
class SceneGraphExtractor(nn.Module):

        def __init__(self):
            super(SceneGraphExtractor, self).__init__()

            self.visible_extractor = LatentDecoderHead(1024, [256, 64, 2])

            self.relative_position_extractor = LatentDecoderHead(1024, [256, 64, 3])

        def forward(self, node_latents, edge_latents):
            
            visible = self.visible_extractor(node_latents)
            relative_position = self.relative_position_extractor(edge_latents)

            return visible, relative_position
        
class FullSceneGraphModel(nn.Module):

        def __init__(self, num_nodes):
            super(FullSceneGraphModel, self).__init__()

            self.generator = SceneGraphGenerator(num_nodes)
            self.extractor = SceneGraphExtractor()

        # def reset_model(self):


        def forward(self, image, object_bounding_boxes, union_bounding_boxes):

            node_latents, edge_latents = self.generator(image, object_bounding_boxes, union_bounding_boxes)

            visible, relative_position = self.extractor(node_latents, edge_latents)

            return visible, relative_position


