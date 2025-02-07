import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision

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
    
class  CNN_Encoder(nn.Module):

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

    def __init__(self, num_nodes, network_args):
        super(SceneGraphGenerator, self).__init__()

        self.feature_extractor_type = network_args['feature_extractor']

        # if feature_extractor == "dino":
        self.extractor = RGBFeatureExtractor(self.feature_extractor_type)
        self.embedding_dim = self.extractor.embedding_dim
        # else:

        self.num_nodes = num_nodes

        self.num_edges = num_nodes*(num_nodes-1)

        self.use_gpu = True
        self.encoder_model = network_args['encoder_model']
        self.latent_size = network_args['latent_size']

        if self.encoder_model == 'lstm':
            #kwargs:
            # num_lstm_layers: number of lstm layers in the encoder
            # dropout: amount of dropout to use in the intermediate layersz
            self.node_encoder = BiLSTM_Encoder(49*self.embedding_dim, self.latent_size//2, self.use_gpu, self.batch_size, 
                                               network_args['num_lstm_layers'], network_args['dropout'])
            self.edge_encoder = BiLSTM_Encoder(49*self.embedding_dim, self.latent_size//2, self.use_gpu, self.batch_size, 
                                               network_args['num_lstm_layers'], network_args['dropout'])
        elif self.encoder_model == "cnn":
            self.node_encoder = CNN_Encoder(self.embedding_dim, [1024,self.latent_size], [2,2])
            self.edge_encoder = CNN_Encoder(self.embedding_dim, [1024,self.latent_size], [2,2])
        elif self.encoder_model == 'iterative-message-passing':
            #kwargs:
            # iterations: number of message passing iterations
            self.encoder = IterativeMessagePoolingPassingLayer(num_nodes, self.embedding_dim, self.latent_size, self.latent_size, network_args['iterations'])
        
        # self.node_latent_downscaler = ops.MLP(128, [256, 64], dropout=0.2)



    def forward(self, image, object_bounding_boxes, union_bounding_boxes, edge_idx_to_node_idxs):

        image_features = self.extractor(image)

        batch_size = image.shape[0]
        num_objects = object_bounding_boxes[0].shape[0]
        num_edges = union_bounding_boxes[0].shape[0]

        # is_all_zeros = torch.all(object_bounding_boxes.flatten(0, 1) == 0, dim=1)

        object_bounding_boxes = torch.split(object_bounding_boxes, 1)

        object_bounding_boxes = [o.squeeze() for o in object_bounding_boxes]

        # objects_features_flattened = object_features.flatten(0, 1)
        # objects_features_flattened[is_all_zeros] = 0
        # object_features = objects_features_flattened.reshape(object_features.shape)

        union_bounding_boxes = torch.split(union_bounding_boxes, 1)
        union_bounding_boxes = [o.squeeze() for o in union_bounding_boxes]

        if self.feature_extractor_type=="dino":
            scale_factor = image_features.shape[-1] / float(image.shape[-1])
            object_features = ops.roi_align(image_features, object_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor)
            edge_features = ops.roi_align(image_features, union_bounding_boxes, output_size=(7,7), spatial_scale=scale_factor)
        if self.feature_extractor_type=="resnet_fpn":

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
            
        if self.encoder_model == 'iterative-message-passing':
            node_latents, edge_latents = self.encoder(object_features, edge_features, edge_idx_to_node_idxs)
        else:
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

#implement message pooling & passing from Scene Graph Generation by Iterative Message Passing (Xu et al)
class IterativeMessagePoolingPassingLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, node_latent_dim, edge_latent_dim, iterations):
        super(IterativeMessagePoolingPassingLayer, self).__init__()

        self.node_gru = nn.GRUCell(node_latent_dim, node_latent_dim)
        self.edge_gru = nn.GRUCell(edge_latent_dim, edge_latent_dim)

        # each pooling layer takes in all node/edge features, computes an adaptive pooling factor based on all adjacent edges/nodes
        self.node_pool_factor = nn.Sequential(nn.Linear(node_latent_dim + edge_latent_dim, 1, bias=False), nn.Sigmoid())
        self.edge_pool_factor_subject = nn.Sequential(nn.Linear(node_latent_dim + edge_latent_dim, 1, bias=False), nn.Sigmoid())
        self.edge_pool_factor_object = nn.Sequential(nn.Linear(node_latent_dim + edge_latent_dim, 1, bias=False), nn.Sigmoid())

        self.node_latent_dim = node_latent_dim
        self.edge_latent_dim = edge_latent_dim
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.num_edges = num_nodes*(num_nodes-1)
        self.iterations=iterations

        self.node_cnn= CNN_Encoder(embedding_dim, [1024,self.node_latent_dim], [2,2])
        self.edge_cnn = CNN_Encoder(embedding_dim, [1024,self.edge_latent_dim], [2,2])


    def forward(self, node_latents, edge_latents, edge_idx_to_node_idxs):

        node_latents = self.node_cnn(node_latents)
        edge_latents = self.edge_cnn(edge_latents)

        node_hidden = self.node_gru(node_latents)
        edge_hidden = self.edge_gru(edge_latents)

        edge_subject_idxs = edge_idx_to_node_idxs[:, :, 1].to(torch.long)
        edge_object_idxs = edge_idx_to_node_idxs[:, :, 2].to(torch.long)

        node_hiddens = [node_hidden.clone()]
        edge_hiddens = [edge_hidden.clone()]

        batch_size = node_latents.shape[0]//self.num_nodes

        #mapping of which nodes are connected to which edges
        node_edge_mat = torch.zeros(batch_size, self.num_nodes, self.num_edges).to(node_latents.device)
        batch_idxs = torch.arange(batch_size).repeat_interleave(2*self.num_edges).to(torch.long).to(node_latents.device)
        node_idxs = edge_idx_to_node_idxs[:, :, 1:].reshape(-1).to(torch.long).to(node_latents.device)
        edge_idxs = torch.arange(self.num_edges).repeat_interleave(2).tile((batch_size)).to(node_latents.device)
        
        node_edge_mat[batch_idxs, node_idxs, edge_idxs] = 1.0

        for i in range(self.iterations):
            node_hidden = node_hiddens[-1].clone()
            edge_hidden = edge_hiddens[-1].clone()
            node_hidden = node_hidden.reshape(batch_size, -1, self.node_latent_dim)
            edge_hidden = edge_hidden.reshape(batch_size, -1, self.edge_latent_dim)

            nodes_repeated_subject = torch.gather(
                node_hidden,
                index=edge_subject_idxs.unsqueeze(-1).tile((1,1,self.node_latent_dim)),
                dim=1
            )

            nodes_repeated_object = torch.gather(
                node_hidden,
                index=edge_object_idxs.unsqueeze(-1).tile((1,1,self.node_latent_dim)),
                dim=1
            )
            #Each row in this tensor is [the subject node latent, edge latent] for all edges
            node_message = self.node_pool_factor(torch.cat((
                nodes_repeated_subject, #we need to repeat these nodes to match the number of edges. 
                edge_hidden,
            ), dim=-1)) * edge_hidden
            node_message = node_edge_mat @ node_message

            edge_message_subject = self.edge_pool_factor_subject(torch.cat((
                nodes_repeated_subject, #we need to repeat these nodes to match the number of edges. 
                edge_hidden
            ), dim=-1)) * nodes_repeated_subject

            edge_message_object = self.edge_pool_factor_object(torch.cat((
                nodes_repeated_object, #we need to repeat these nodes to match the number of edges. 
                edge_hidden
            ), dim=-1)) * nodes_repeated_object

            edge_message = edge_message_subject + edge_message_object

            node_message = node_message.reshape(-1, self.node_latent_dim)
            edge_message = edge_message.reshape(-1, self.edge_latent_dim)

            node_hiddens.append(self.node_gru(node_message, node_hiddens[i]))
            edge_hiddens.append(self.edge_gru(edge_message, edge_hiddens[i]))

        
        
        return (node_hidden, edge_hidden)

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

        #num_lstm_layers=1, latent_size=512, dropout=0.5, encoder_model="lstm", feature_extractor="dino"
        def __init__(self, batch_size, num_nodes, num_node_labels, num_edge_labels, network_args):
            super(StowTrainSceneGraphModel, self).__init__()

            dropout = network_args['dropout']
            latent_size = network_args['latent_size']

            self.generator = SceneGraphGenerator(num_nodes, network_args)
            
            self.node_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_node_labels], dropout)
            self.edge_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_edge_labels], dropout)

            self.relative_position_extractor = MLP(latent_size, [latent_size//2, latent_size//4, 3], dropout)


        # def reset_model(self):


        def forward(self, image, object_bounding_boxes, union_bounding_boxes, edge_idx_to_node_idxs):

            node_latents, edge_latents = self.generator(image, object_bounding_boxes, union_bounding_boxes, edge_idx_to_node_idxs)

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
