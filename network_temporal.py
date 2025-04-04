import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision

from network import MLP, BiLSTM_Encoder, CNN_Encoder, IterativeMessagePoolingPassingLayer, RGBFeatureExtractor, SceneGraphGenerator
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from scipy.optimize import linear_sum_assignment

class TemporalSceneGraphModel(nn.Module):
        #num_lstm_layers=1, latent_size=512, dropout=0.5, encoder_model="lstm", feature_extractor="dino"
        def __init__(self, batch_size, num_nodes, num_node_labels, num_edge_labels, sequence_length, network_args):
            super(TemporalSceneGraphModel, self).__init__()

            dropout = network_args['dropout']
            latent_size = network_args['latent_size']
            self.generator = TemporalSceneGraphGenerator(num_nodes, sequence_length, network_args)
            
            if network_args['use_batch_norm']:
                batch_norm = nn.BatchNorm1d
            else:
                batch_norm = None

            self.node_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_node_labels], dropout, normalization_layer=batch_norm)
            self.edge_label_extractor = MLP(latent_size, [latent_size//2, latent_size//4, num_edge_labels], dropout, normalization_layer=batch_norm)

            self.relative_position_extractor = MLP(latent_size, [latent_size//2, latent_size//4, 3], dropout)

            self.node_box_head = DeformableDetrMLPPredictionHead(
                input_dim=latent_size,
                hidden_dim=latent_size,
                output_dim=4,
                num_layers=3
            )


        # def reset_model(self):


        def forward(self, image, object_bounding_boxes, union_bounding_boxes, edge_idx_to_node_idxs,
                    node_network_mask, edge_network_mask):

            batch_size = image.shape[0]
            seq_len = image.shape[1]
            num_objects = object_bounding_boxes.shape[2]
            num_edges = union_bounding_boxes.shape[2]

            node_latents, edge_latents = self.generator(image, object_bounding_boxes, union_bounding_boxes, edge_idx_to_node_idxs,
                                                        node_network_mask, edge_network_mask)

            node_latents = node_latents.reshape(batch_size, seq_len, num_objects, -1)

            node_boxes = self.node_box_head(node_latents)
            
            # node_labels = self.node_label_extractor(node_latents)
            # edge_labels = self.edge_label_extractor(edge_latents)
            # relative_position = self.relative_position_extractor(edge_latents)

            # edge_labels = edge_labels.reshape(batch_size, seq_len, num_edges, -1)
            # relative_position = relative_position.reshape(batch_size, seq_len, num_edges, -1)

            return node_latents, node_boxes

class TemporalSceneGraphGenerator(nn.Module):

    def __init__(self, num_nodes, sequence_length, network_args):
        super(TemporalSceneGraphGenerator, self).__init__()

        self.feature_extractor_type = network_args['feature_extractor']

        self.extractor = RGBFeatureExtractor(self.feature_extractor_type)
        self.embedding_dim = self.extractor.embedding_dim

        self.num_nodes = num_nodes
        self.num_edges = num_nodes*(num_nodes-1)
        self.sequence_length = sequence_length

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
            self.encoder = TemporalIterativeMessagePoolingPassingLayer(num_nodes, self.embedding_dim, self.latent_size, self.latent_size, network_args['iterations'],
                                                                       sequence_length, network_args["multi_frame_attn"])
        
        self.matcher = SequenceObjectMatcher(network_args["matching_threshold"], self.latent_size)
        # self.node_latent_downscaler = ops.MLP(128, [256, 64], dropout=0.2)



    def forward(self, image, object_bounding_boxes, union_bounding_boxes, edge_idx_to_node_idxs,
                node_network_mask, edge_network_mask):

        # flatten sequence dim in to batch for bbox processing
        batch_size = image.shape[0]
        seq_len = image.shape[1]
        num_objects = object_bounding_boxes.shape[2]
        num_edges = union_bounding_boxes.shape[2]
        image = image.flatten(0, 1)
        object_bounding_boxes = object_bounding_boxes.flatten(0, 1)
        union_bounding_boxes = union_bounding_boxes.flatten(0, 1)
        edge_idx_to_node_idxs = edge_idx_to_node_idxs.flatten(0, 1)
        node_network_mask = node_network_mask.flatten(0, 1)
        edge_network_mask = edge_network_mask.flatten(0, 1)

        image_features = self.extractor(image)
        
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
            node_latents, edge_latents = self.encoder(object_features, edge_features, edge_idx_to_node_idxs,
                                                      node_network_mask, edge_network_mask)


            self.matcher(node_latents.reshape(batch_size, seq_len, num_objects, -1), 
                        edge_latents.reshape(batch_size, seq_len, num_edges, -1),
                        node_network_mask.reshape(batch_size, seq_len, num_objects),
                        edge_idx_to_node_idxs.reshape(batch_size, seq_len, num_edges, -1))
        # else:
        #     if self.encoder_model == 'lstm':
        #         object_features = object_features.reshape((batch_size, num_objects, -1))
        #         edge_features = edge_features.reshape(batch_size, num_edges, -1)

        #     node_latents = self.node_encoder(object_features)
            
        #     # node_latents_downscaled = self.node_latent_downscaler(node_latents)

        #     edge_latents = self.edge_encoder(edge_features)

        #     if self.encoder_model == "cnn":
        #         node_latents = node_latents.reshape((batch_size, num_objects, -1))
        #         edge_latents = edge_latents.reshape(batch_size, num_edges, -1)
        
        return node_latents, edge_latents
    

#implement message pooling & passing from Scene Graph Generation by Iterative Message Passing (Xu et al)
class TemporalIterativeMessagePoolingPassingLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, node_latent_dim, edge_latent_dim, iterations,
                 sequence_length, use_multi_frame_attn):
        super(TemporalIterativeMessagePoolingPassingLayer, self).__init__()

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

        self.sequence_length = sequence_length

        self.node_cnn= CNN_Encoder(embedding_dim, [1024,self.node_latent_dim], [2,2])
        self.edge_cnn = CNN_Encoder(embedding_dim, [1024,self.edge_latent_dim], [2,2])

        self.node_attention_head = MultiFrameAttention(node_latent_dim, self.num_nodes, sequence_length)
        self.edge_attention_head = MultiFrameAttention(edge_latent_dim, self.num_edges, sequence_length)
        self.use_multi_frame_attn = use_multi_frame_attn


    def forward(self, node_latents, edge_latents, edge_idx_to_node_idxs,
                node_network_mask, edge_network_mask):

        node_latents = self.node_cnn(node_latents)
        edge_latents = self.edge_cnn(edge_latents)

        node_hidden = self.node_gru(node_latents)
        edge_hidden = self.edge_gru(edge_latents)

        edge_subject_idxs = edge_idx_to_node_idxs[..., 1].to(torch.long)
        edge_object_idxs = edge_idx_to_node_idxs[..., 2].to(torch.long)

        node_hiddens = [node_hidden.clone()]
        edge_hiddens = [edge_hidden.clone()]

        batch_size = node_latents.shape[0]//(self.num_nodes*self.sequence_length)


        #mapping of which nodes are connected to which edges
        node_edge_mat = torch.zeros(batch_size*self.sequence_length, self.num_nodes, self.num_edges).to(node_latents.device)
        batch_idxs = torch.arange(batch_size*self.sequence_length).repeat_interleave(2*self.num_edges).to(torch.long).to(node_latents.device)
        node_idxs = edge_idx_to_node_idxs[..., 1:].reshape(-1).to(torch.long).to(node_latents.device)
        edge_idxs = torch.arange(self.num_edges).repeat_interleave(2).tile((batch_size*self.sequence_length)).to(node_latents.device)
        
        node_edge_mat[batch_idxs, node_idxs, edge_idxs] = 1.0

        for i in range(self.iterations):
            node_hidden = node_hiddens[-1].clone()
            edge_hidden = edge_hiddens[-1].clone()
            node_hidden = node_hidden.reshape(batch_size*self.sequence_length, -1, self.node_latent_dim)
            edge_hidden = edge_hidden.reshape(batch_size*self.sequence_length, -1, self.edge_latent_dim)

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


            if self.use_multi_frame_attn:
                node_message = node_message.reshape(batch_size, self.sequence_length, self.num_nodes, self.node_latent_dim)
                edge_message = edge_message.reshape(batch_size, self.sequence_length, self.num_edges, self.edge_latent_dim)

                node_message = self.node_attention_head(node_message)
                edge_message = self.edge_attention_head(edge_message)

            node_message = node_message.reshape(-1, self.node_latent_dim)
            edge_message = edge_message.reshape(-1, self.edge_latent_dim)

            node_hiddens.append(self.node_gru(node_message, node_hiddens[i]))
            edge_hiddens.append(self.edge_gru(edge_message, edge_hiddens[i]))  
        
        return (node_hiddens[-1].reshape(batch_size*self.sequence_length, -1, self.node_latent_dim),
                edge_hiddens[-1].reshape(batch_size*self.sequence_length, -1, self.edge_latent_dim))
    
#implement message pooling & passing from Scene Graph Generation by Iterative Message Passing (Xu et al)
class MultiFrameAttention(nn.Module):
    def __init__(self, latent_dim, num_graph_elems, num_scenes):
        super(MultiFrameAttention, self).__init__()

        self.pos_encoding = Summer(PositionalEncoding2D(latent_dim))
        self.attention_layer = torch.nn.MultiheadAttention(latent_dim, 4, dropout=0.2, batch_first=True)

        #each graph should only attend with itself and past graphs.
        self.attn_mask = torch.triu(torch.ones((num_scenes, num_scenes)), diagonal=1).cuda()
        tmp_attn_mask = torch.zeros((num_scenes*num_graph_elems, num_scenes*num_graph_elems)).cuda()
        for i in range(num_scenes):
            for j in range(num_scenes):
                tmp_attn_mask[i*num_graph_elems:(i+1)*num_graph_elems, j*num_graph_elems:(j+1)*num_graph_elems] = \
                    self.attn_mask[i,j]
        self.attn_mask = tmp_attn_mask == 1.0


    # input is (batch_size, scene_sequence_length, num_graph elems, latent_size)
    def forward(self, x):

        x_with_positional_embedding = self.pos_encoding(x)

        x_flattened = x_with_positional_embedding.flatten(1, 2)

        attn_output, attn_output_weights = self.attention_layer(x_flattened,
                                                                x_flattened,
                                                                x_flattened, 
                                                                attn_mask=self.attn_mask)

        return attn_output
    
# a module that performs object tracking between frames, and updates the graph according to their similarity.
# also reorders 
class SequenceObjectMatcher(nn.Module):
    def __init__(self, matching_threshold, embedding_dim):
        super(SequenceObjectMatcher, self).__init__()

        self.matching_threshold = matching_threshold
        self.occluded_node_embedding = nn.Embedding(1, embedding_dim)
        self.occluded_edge_embedding = nn.Embedding(1, embedding_dim)
        self.empty_node_embedding = nn.Embedding(1, embedding_dim)
        self.empty_edge_embedding = nn.Embedding(1, embedding_dim)

    #unbatched
    def match_graph(self, prior_objects, full_nodes):

        if len(prior_objects) == 0:
            indices = []
            for idx, obj in enumerate(full_nodes):
                prior_objects.append([obj])
                indices.append(idx)
            return prior_objects, indices, indices
        
        # add a number of matches equal to the number of objects in the current frame, such that if there are no similar objects to match to, it can serve as
        # a signal to add the object as a new object
        similarity_matrix = torch.full((len(prior_objects) + full_nodes.shape[0], full_nodes.shape[0]), self.matching_threshold)
        
        #performs dot product between all nodes with attached object detections and prior detected objects
        for object_index, object_occurences in enumerate(prior_objects):
            similarity_matrix_across_prior_occurences = torch.einsum("nc,mc->nm", torch.stack(object_occurences), full_nodes)
            similarity_matrix[object_index] = similarity_matrix_across_prior_occurences.max(dim=0).values.cpu()

        # solve arrangement problem, find a matching between prior objects (+ threshold values) that maximizes the sum of similarity    
        prior_indices, node_indices = linear_sum_assignment(similarity_matrix.detach(), maximize=True)
        
        for prior_index, node_index in zip(prior_indices, node_indices):
            # object matched with one of the threshold objects, and thus should be added as a new object
            #ensure that if node indices are out of order, we add it to the correct spot in prior_objects
            for new_object_idx in range(prior_index - (len(prior_objects)-1)):
                prior_objects.append([])
            # object matched with a prior object it, append it to the list of older objects
            else:
                prior_objects[prior_index].append(full_nodes[node_index])

        return (prior_objects, prior_indices, node_indices)


    def forward(self, node_sequence, edge_sequence, network_mask, edge_idx_to_node):
        batch_size = node_sequence.shape[0]
        seq_len = node_sequence.shape[1]
        num_objects = node_sequence.shape[2]

        batch_node_indices = []
        batch_occluded_objects = []
        for batch_idx in range(batch_size):
            prior_objects = []

            batch_node_indices.append([])
            batch_occluded_objects.append([])

            for sequence_idx in range(seq_len):
                curr_nodes = node_sequence[batch_idx, sequence_idx]
                curr_network_mask = network_mask[batch_idx, sequence_idx]

                nonempty_nodes = curr_nodes[curr_network_mask]
                
                prior_objects, prior_indices, node_indices = self.match_graph(prior_objects, nonempty_nodes)
                
                if len(node_indices) < len(prior_objects):
                    for obj_idx in range(len(prior_objects)):
                        if obj_idx not in node_indices:
                            batch_occluded_objects[-1].append(obj_idx)

                batch_node_indices[-1].append(node_indices)

                

                node_sequence[batch_idx, sequence_idx, prior_indices] = nonempty_nodes[node_indices]
                if len(batch_occluded_objects[-1]) > 0:
                    node_sequence[batch_idx, sequence_idx, batch_occluded_objects[-1]] = self.occluded_node_embedding(torch.Tensor([0]).to(torch.long).cuda())
                if len(prior_objects) < num_objects:
                    empty_idxs = torch.arange(len(prior_objects), num_objects)
                    node_sequence[batch_idx, sequence_idx, empty_idxs] = self.empty_node_embedding(torch.Tensor([0]).to(torch.long).cuda())

                # edge_new = torch.zeros_like(edge_sequence[batch_idx, sequence_idx])

                # object_full_mask = edge_idx_to_node[batch_idx, sequence_idx, 1] in node_indices
                # object_occluded_mask = edge_idx_to_node[batch_idx, sequence_idx, 1] in batch_occluded_objects[-1]
                # object_empty_mask = edge_idx_to_node[batch_idx, sequence_idx, 1] > len(prior_objects)

                # subject_full_mask = edge_idx_to_node[batch_idx, sequence_idx, 2] in node_indices
                # subject_occluded_mask = edge_idx_to_node[batch_idx, sequence_idx, 2] in batch_occluded_objects[-1]
                # subject_empty_mask = edge_idx_to_node[batch_idx, sequence_idx, 2] > len(prior_objects)

                # edge_new
                
class DeformableDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.
    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return self.sigmoid(x)
