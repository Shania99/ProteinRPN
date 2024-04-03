"""
This module contains loss functions used for training.
"""

from calendar import c
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def custom_bce_loss_pruning(pred_list, node_list, data_y):
    """

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for each ego graph
        node_list (List[torch.Tensor]): List of nodes as part of ego graph for one batch
        data_y (torch.Tensor): Ground truth for entire batch
        values for each ego graph

    Returns:
        Tensor: Mean squared error loss for the entire batch
    """
    mse_losses = []
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        gt_tensor = gt_tensor.float()
        # print("node_tensor", node_tensor.shape)
        # print("data_y", data_y.shape)
        pred_tensor, gt_tensor = pred_tensor.to(gt_tensor.device), gt_tensor.to(
            pred_tensor.device
        )
        print("predicted labels in anchor", pred_tensor.squeeze())
        print("ground truth labels in anchor", gt_tensor)
        # print("Inside loss func")
        # print("printing shapes", pred_tensor.shape, gt_tensor.shape)
        # print("printing pred_tensor", pred_tensor)
        # print("printing gt_tensor", gt_tensor)
        gt_tensor = gt_tensor.unsqueeze(1)
        # pred_tensor = pred_tensor.squeeze(-1)
        mse = bce_loss(pred_tensor, gt_tensor, reduction="mean")
        mse_losses.append(mse)
    # Calculate the average BCE over all tensors in the batch
    batch_loss = torch.mean(torch.stack(mse_losses))
    return batch_loss


def compute_accuracy_pruning(pred_list, node_list, data_y):
    """

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for each ego graph
        node_list (List[torch.Tensor]): List of nodes as part of ego graph for one batch
        data_y (torch.Tensor): Ground truth for entire batch

    Returns:
        float: Accuracy for the entire batch
    """
    correct = 0
    total = 0
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        pred_tensor, gt_tensor = pred_tensor.to(gt_tensor.device), gt_tensor.to(
            pred_tensor.device
        )
        pred_tensor = pred_tensor > 0.5
        correct += (pred_tensor == gt_tensor).sum().item()
        total += pred_tensor.size(0)
    return correct / total


def compute_f1_score_pruning(pred_list, node_list, data_y):
    """
    Computes the F1 score for the entire batch.

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for each ego graph.
        node_list (List[torch.Tensor]): List of nodes as part of ego graph for one batch.
        data_y (torch.Tensor): Ground truth for entire batch.

    Returns:
        float: F1 score for the entire batch.
    """
    y_true = []
    y_pred = []
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        pred_tensor = pred_tensor > 0.5

        y_true.extend(gt_tensor.cpu().numpy().tolist())
        y_pred.extend(pred_tensor.cpu().numpy().tolist())

    # Calculate F1 score using true positives, false positives, and false negatives
    f1 = sklearn_f1_score(y_true, y_pred, average="binary")

    return f1


def compute_roc_auc_score_pruning(pred_list, node_list, data_y):
    """
    Computes the ROC AUC score for the entire batch.

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for each ego graph.
        node_list (List[torch.Tensor]): List of nodes as part of ego graph for one batch.
        data_y (torch.Tensor): Ground truth for entire batch.

    Returns:
        float: F1 score for the entire batch.
    """
    y_true = []
    y_pred = []
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        # pred_tensor = pred_tensor > 0.5
        y_true.extend(gt_tensor.cpu().numpy().tolist())
        y_pred.extend(pred_tensor.cpu().numpy().tolist())
    print("y_true", y_true)
    print("y_pred", y_pred)
    # Calculate F1 score using true positives, false positives, and false negatives
    roc = sklearn_roc_auc_score(y_true, y_pred, average=None)

    return roc


def compute_confusion_matrix(pred_list, node_list, data_y, func_proba, ego_gt_labels):
    """
    Computes the confusion matrix for the entire batch.

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for ONLY true positive anchors.
        node_list (List[torch.Tensor]): List of nodes as part of ego graphs that are true positive anchors.
        data_y (torch.Tensor): Ground truth for entire batch.
        func_proba (torch.Tensor): Functional probability for each ego graph/anchor.

    Returns:
        np.array, np.array: confusion matrix for pruning and anchor classification
    """
    y_true = []
    y_pred = []
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        pred_tensor = pred_tensor > 0.5
        y_true.extend(gt_tensor.cpu().numpy().tolist())
        y_pred.extend(pred_tensor.cpu().numpy().tolist())

    # Calculate F1 score using true positives, false positives, and false negatives
    confusion_matrix_pruning = sklearn_confusion_matrix(y_true, y_pred)

    func_pred = func_proba > 0.5
    func_true = ego_gt_labels.cpu().numpy()
    confusion_matrix_func = sklearn_confusion_matrix(func_true, func_pred)
    return confusion_matrix_pruning, confusion_matrix_func
