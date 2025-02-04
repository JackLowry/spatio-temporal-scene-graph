import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class BiLSTM_Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, use_gpu, batch_size, num_lstm_layers, dropout):
        super(BiLSTM_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout)
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
    
class CNN_Encoder(nn.Module):

    def __init__(self, input_channels, channels, kernel_sizes, dropout=0, activation_fn=nn.ReLU):
        super(CNN_Encoder, self).__init__()


        layers = []
        last_channel_size = input_channels

        #create CNN
        for layer_idx in range(len(channels)):
            # if dropout != 0:
            #     layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv2d(last_channel_size, channels[layer_idx], kernel_sizes[layer_idx]))
            layers.append(activation_fn())

            layers.append(nn.MaxPool2d(2, 2))

            last_channel_size = channels[layer_idx]

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x).squeeze()

class RGBFeatureExtractor(nn.Module):

    def __init__(self, model_type):
        super(RGBFeatureExtractor, self).__init__()

        if model_type == "dino":
            self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.forward = self.dino_forward
            self.embedding_dim=768
        elif model_type == "resnet_fpn":
            self.extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            self.forward = self.resnet_fpn_forward
            self.embedding_dim = 3840
            self.outputs = []

            self.extractor.layer1.register_forward_hook(self.hook)
            self.extractor.layer2.register_forward_hook(self.hook)
            self.extractor.layer3.register_forward_hook(self.hook)
            self.extractor.layer4.register_forward_hook(self.hook)
            self.extractor.fc = nn.Identity()

            # self.fpn = 

    def hook(self, module, input, output):
        self.outputs.append(output)        
    
    def dino_forward(self, x):
        
        features = self.extractor.get_intermediate_layers(x, 1)[0]
        
        return features.reshape(x.shape[0], x.shape[-2]//14,  x.shape[-1]//14, -1).permute((0, 3, 1, 2))
    
    def resnet_fpn_forward(self, x):
        self.outputs = []
        self.extractor(x)
        return self.outputs

class MLP(nn.Module):

    def __init__(self, out_size, layer_dimensions, dropout, activation_fn=nn.ReLU, normalization_layer=nn.BatchNorm1d):
        super(MLP, self).__init__()

        layers = []
        last_layer_dim = out_size

        #create MLP
        for layer_dim in layer_dimensions[:-1]:
            if dropout != 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(last_layer_dim, layer_dim))

            layers.append(ops.Permute([0, 2, 1]))
            if normalization_layer:
                layers.append(normalization_layer(layer_dim))
            layers.append(ops.Permute([0, 2, 1]))

            layers.append(activation_fn())
            last_layer_dim = layer_dim
        layers.append(nn.Linear(last_layer_dim, layer_dimensions[-1]))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class SceneGraphGenerator(nn.Module):

    def __init__(self, num_nodes, num_lstm_layers, latent_size, dropout, encoder_model, feature_extractor):
        super(SceneGraphGenerator, self).__init__()

        self.feature_extractor = feature_extractor

        # if feature_extractor == "dino":
        self.extractor = RGBFeatureExtractor(feature_extractor)
        self.embedding_dim = self.extractor.embedding_dim
        # else:

        self.num_nodes = num_nodes

        self.num_edges = num_nodes*(num_nodes-1)

        self.use_gpu = True
        self.batch_size = 16
        self.encoder_model = encoder_model

        if self.encoder_model == 'lstm':
            self.node_encoder = BiLSTM_Encoder(49*self.embedding_dim, latent_size//2, self.use_gpu, self.batch_size, num_lstm_layers, dropout)
            self.edge_encoder = BiLSTM_Encoder(49*self.embedding_dim, latent_size//2, self.use_gpu, self.batch_size, num_lstm_layers, dropout)
        elif self.encoder_model == "cnn":
            self.node_encoder = CNN_Encoder(self.embedding_dim, [1024,latent_size], [2,2])
            self.edge_encoder = CNN_Encoder(self.embedding_dim, [1024,latent_size], [2,2])
        # self.node_latent_downscaler = ops.MLP(128, [256, 64], dropout=0.2)



    def forward(self, image, object_bounding_boxes, union_bounding_boxes):

        image_features = self.extractor(image)

        batch_size = image.shape[0]
        num_objects = object_bounding_boxes[0].shape[0]
        num_edges = union_bounding_boxes[0].shape[0]



        is_all_zeros = torch.all(object_bounding_boxes.flatten(0, 1) == 0, dim=1)

        object_bounding_boxes = torch.split(object_bounding_boxes, 1)

        object_bounding_boxes = [o.squeeze() for o in object_bounding_boxes]

        # objects_features_flattened = object_features.flatten(0, 1)
        # objects_features_flattened[is_all_zeros] = 0
        # object_features = objects_features_flattened.reshape(object_features.shape)

        union_bounding_boxes = torch.split(union_bounding_boxes, 1)
        union_bounding_boxes = [o.squeeze() for o in union_bounding_boxes]

        if self.feature_extractor=="dino":
            scale_factor = image_features.shape[-1] / float(image.shape[-1])
            object_features = ops.roi_align(image_features, object_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor)
            edge_features = ops.roi_align(image_features, union_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor)
        if self.feature_extractor=="resnet_fpn":

            object_features = []
            edge_features = []

            for i in range(len(image_features)):
                feature_map = image_features[i]
                scale_factor = feature_map.shape[-1] / float(image.shape[-1])
                object_features.append(ops.roi_align(feature_map, object_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor))
                edge_features.append(ops.roi_align(feature_map, union_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor))
            
            object_features = torch.concat(object_features,dim=1)
            # object_features = object_features.reshape(object_features.shape[0], object_features.shape[1], -1)
            edge_features = torch.concat(edge_features,dim=1)
            # edge_features = edge_features.reshape(edge_features.shape[0], edge_features.shape[1], -1, )
            


        if self.encoder_model == 'lstm':
            object_features = object_features.reshape((batch_size, num_objects, -1))
            edge_features = edge_features.reshape(batch_size, num_edges, -1)

        node_latents = self.node_encoder(object_features)
        
        # node_latents_downscaled = self.node_latent_downscaler(node_latents)

        edge_latents = self.edge_encoder(edge_features)

        if self.encoder_model == "cnn":
            node_latents = node_latents.reshape((batch_size, num_objects, -1))
            edge_latents = edge_latents.reshape(batch_size, num_edges, -1)

        return node_latents, edge_latents

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class SceneGraphExtractor(nn.Module):

        def __init__(self):
            super(SceneGraphExtractor, self).__init__()

            self.visible_extractor = MLP(512, [256, 64, 2])

            self.relative_position_extractor = MLP(512, [256, 64, 3])

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
        
class PretrainSceneGraphModel(nn.Module):

        def __init__(self, num_nodes, num_node_labels, num_edge_labels, num_lstm_layers=1, latent_size=512, dropout=0.5, encoder_model="lstm", feature_extractor="dino"):
            super(PretrainSceneGraphModel, self).__init__()

            self.generator = SceneGraphGenerator(num_nodes, num_lstm_layers, latent_size, dropout, encoder_model, feature_extractor)
            
            self.node_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_node_labels], dropout)
            self.edge_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_edge_labels], dropout)

            self.relative_position_extractor = MLP(latent_size, [latent_size//2, latent_size//4, 2], dropout)


        # def reset_model(self):


        def forward(self, image, object_bounding_boxes, union_bounding_boxes):

            node_latents, edge_latents = self.generator(image, object_bounding_boxes, union_bounding_boxes)

            node_labels = self.node_label_extractor(node_latents)
            edge_labels = self.edge_label_extractor(edge_latents)

            relative_position = self.relative_position_extractor(edge_latents)

            return (node_labels, edge_labels, relative_position)

class StowTrainSceneGraphModel(nn.Module):

        def __init__(self, num_nodes, num_node_labels, num_edge_labels, num_lstm_layers=1, latent_size=512, dropout=0.5, encoder_model="lstm", feature_extractor="dino"):
            super(StowTrainSceneGraphModel, self).__init__()

            self.generator = SceneGraphGenerator(num_nodes, num_lstm_layers, latent_size, dropout, encoder_model, feature_extractor)
            
            self.node_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_node_labels], dropout)
            self.edge_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_edge_labels], dropout)

            self.relative_position_extractor = MLP(latent_size, [latent_size//2, latent_size//4, 3], dropout)


        # def reset_model(self):


        def forward(self, image, object_bounding_boxes, union_bounding_boxes):

            node_latents, edge_latents = self.generator(image, object_bounding_boxes, union_bounding_boxes)

            node_labels = self.node_label_extractor(node_latents)
            edge_labels = self.edge_label_extractor(edge_latents)

            relative_position = self.relative_position_extractor(edge_latents)

            return (node_labels, edge_labels, relative_position)

class UpdateSceneGraphModel(nn.Module):
        def __init__(self, num_nodes, num_node_labels, num_edge_labels, num_lstm_layers=1, latent_size=512, dropout=0.5, encoder_model="lstm", feature_extractor="dino"):
            super(StowTrainSceneGraphModel, self).__init__()

            self.latent_encoder = SceneGraphGenerator(num_nodes, num_lstm_layers, latent_size, dropout, encoder_model, feature_extractor)
            
            self.graph_


            self.node_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_node_labels], dropout)
            self.edge_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_edge_labels], dropout)

            self.relative_position_extractor = MLP(latent_size, [latent_size//2, latent_size//4, 3], dropout)


        # def reset_model(self):


        def forward(self, observations):

            latents = []

            for o in observations:
                (image, object_bounding_boxes, union_bounding_boxes, node_latents, edge_latents) = o
                latents.append(self.generator(image, object_bounding_boxes, union_bounding_boxes))

            node_latents, edge_latents = self.generator(image, object_bounding_boxes, union_bounding_boxes)

            node_labels = self.node_label_extractor(node_latents)
            edge_labels = self.edge_label_extractor(edge_latents)

            relative_position = self.relative_position_extractor(edge_latents)

            return (node_labels, edge_labels, relative_position)
