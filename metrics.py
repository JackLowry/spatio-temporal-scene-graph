import sklearn
import torch

def graph_f1_score(logits, gt, avg='macro'):
    precision = graph_precision(logits, gt, avg=avg)
    recall = graph_recall_at_k(logits, gt, k=1, avg=avg)

    f1_score =  (2*precision*recall)/(precision+recall)

    return f1_score

# logits: shape (B, num_nodes/edges, num_classes)
# gt: shape (B, num_nodes/edges)
def graph_recall_at_k(logits, gt, k, avg='macro'):
    top_k_preds = torch.topk(logits, k, dim=-1).indices  # Shape: (n_samples, k)
    
    label_in_top_k = (top_k_preds == gt.unsqueeze(-1)).any(dim=-1)  # Shape: (n_samples,)

    true_positive_in_class_mat = torch.nn.functional.one_hot(gt)
    #sum across all nodes/edges in graph
    num_occurences_per_class = torch.sum(true_positive_in_class_mat, dim=1).to(torch.float)
    num_occurences = num_occurences_per_class.sum(dim=-1)
    true_positive_in_class_mat[torch.bitwise_not(label_in_top_k)] = 0
    #sum across samples and all nodes/edges in samplesQ
    true_positives_per_class = torch.sum(true_positive_in_class_mat, dim=1).to(torch.float)
    true_positives = true_positives_per_class.sum(dim=-1)
    
    if avg == 'macro':
        recall_at_k = true_positives/num_occurences
        recall_at_k = recall_at_k.mean()
    if avg == 'micro':
        recall_at_k = true_positives.sum()/num_occurences.sum()
    return recall_at_k

def graph_precision(logits, gt, avg='macro'):
    pred_labels = torch.argmax(logits, dim=-1)

    false_positives_per_class_mat = torch.nn.functional.one_hot(pred_labels)
    false_positives_per_class_mat[pred_labels == gt] = 0
    #sum across samples and all nodes/edges in samples
    false_positives_per_class = torch.sum(false_positives_per_class_mat, dim=1).to(torch.float)
    false_positives  = false_positives_per_class.sum(dim=-1)

    true_positive_in_class_mat = torch.nn.functional.one_hot(gt)
    true_positive_in_class_mat[pred_labels != gt] = 0
    #sum across samples and all nodes/edges in samples
    true_positives_per_class = torch.sum(true_positive_in_class_mat, dim=1).to(torch.float)
    true_positives = true_positives_per_class.sum(dim=-1)

    if avg == 'macro':
        precision = true_positives/(false_positives+true_positives)   
        precision = precision.mean()
    if avg == 'micro':
        true_positives = true_positives.sum()
        false_positives = false_positives.sum()
        precision = true_positives/(false_positives+true_positives)   
    return precision
# def confusion_matrix(logits, gt, labels=None):
#     pred_labels = torch.argmax(logits, dim=-1)
#     return sklearn.metrics.confusion_matrix(gt, pred_labels, labels)
