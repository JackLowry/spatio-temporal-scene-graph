import sklearn
import torch

def f1_score(logits, gt):
    pred_labels = torch.argmax(logits, dim=-1)

    return sklearn.metrics.f1_score(
        gt,
        pred_labels,
        average='macro'
    )

def recall_at_k(logits, gt, k):
    

    top_k_preds = torch.topk(logits, k, dim=1).indices  # Shape: (n_samples, k)
    
    label_in_top_k = (top_k_preds == gt.unsqueeze(1)).any(dim=1)  # Shape: (n_samples,)

    true_positive_in_class_mat = torch.nn.functional.one_hot(gt)
    num_occurences_per_class = torch.sum(true_positive_in_class_mat, dim=1)
    true_positive_in_class_mat[torch.bitwise_not(label_in_top_k)] = 0
    true_positives_per_class = torch.sum(true_positive_in_class_mat, dim=1)
    
    recall_at_k_per_class = num_occurences_per_class/true_positives_per_class
    
    # Compute mean Recall@K across all samples
    return recall_at_k_per_class.mean()

def confusion_matrix(logits, gt, labels=None):
    pred_labels = torch.argmax(logits, dim=-1)
    return sklearn.metrics.confusion_matrix(gt, pred_labels, labels)
