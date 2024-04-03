import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from torch.optim import Adam
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from itertools import compress
import esm

torch.manual_seed(2)
# from torch.optim.lr_scheduler import ExponentialLR

from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score


from torch_geometric.loader import DataLoader


# from torcheval.metrics import BinaryAUROC


from loss import (
    custom_bce_loss_pruning,
    compute_accuracy_pruning,
    compute_f1_score_pruning,
    compute_roc_auc_score_pruning,
    compute_confusion_matrix,
)

from GRPN import GraphRPN
from GRPN_processing import get_proposed_candidates_batches

esm6_model, esm6_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm6_model.eval()

data_list = torch.load("datasets/data_list.pt")

embed_dim = data_list[0].x.shape[1]
train_data_list = data_list[: int(0.8 * len(data_list))]
test_data_list = data_list[int(0.8 * len(data_list)) :]
train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphRPN(k=2, input_dim=embed_dim, hidden_dim=128, num_classes=1).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
# scheduler = ExponentialLR(optimizer, gamma=0.5)
# metric = BinaryAUROC(num_tasks=1)
##### TRAINING LOOP
print("Starting training loop", model)
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y/%H:%M:%S")
num_epochs = 100  # Replace with the desired number of epochs
writer = SummaryWriter(f"./GRPN_train_v{dt_string}")
# writer.add_graph(model, data_list[0])


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        print("Going through data")
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Assuming 'forward' returns two lists of variable-sized tensors
        pred_scores, pred_nodes, func_proba = model.forward(x, edge_index, batch)

        # Ground truth: a single list of variable-sized tensors
        gt_scores = data.y
        # print("data.y", data.y)
        ego_gt_labels = data.ego_label
        # print("pred_scores", len(pred_scores), len(pred_scores[0]))
        # print("pred_nodes", len(pred_nodes), len(pred_nodes[0]))
        # print("gt_scores", len(gt_scores), gt_scores)
        ego_gt_labels = ego_gt_labels.unsqueeze(1)

        pred_scores_pos = list(compress(pred_scores, ego_gt_labels))
        pred_nodes_pos = list(compress(pred_nodes, ego_gt_labels))
        gt_scores_pos = gt_scores
        pruning_loss = custom_bce_loss_pruning(
            pred_scores_pos,
            pred_nodes_pos,
            gt_scores_pos,
        )  # calculating pruning loss only for positive anchors

        ego_gt_labels = ego_gt_labels.float()
        # print("func_proba", func_proba.squeeze())
        # print("ego_gt_labels", ego_gt_labels.squeeze())
        functionality_loss = bce_loss(func_proba, ego_gt_labels)
        loss = (
            pruning_loss + functionality_loss
        )  ##### WRONG NEEDS TO DO PRUNING LOSS ONLY FOR FUNCTIONAL REGIONS (P*=1) [FIXED]
        # Backpropagation
        print("true functionality labels of each anchor node", ego_gt_labels.squeeze())
        print(
            "predicted functionality labels of each anchor node", func_proba.squeeze()
        )
        with torch.no_grad():
            acc = compute_accuracy_pruning(
                pred_scores_pos, pred_nodes_pos, gt_scores_pos
            )
            # print("pred_scores_pos", pred_scores_pos)
            # print("gt_scores_pos", gt_scores_pos)
            f1 = compute_f1_score_pruning(
                pred_scores_pos, pred_nodes_pos, gt_scores_pos
            )
            roc = compute_roc_auc_score_pruning(
                pred_scores_pos, pred_nodes_pos, gt_scores_pos
            )

            acc_func = sklearn_accuracy_score(
                ego_gt_labels.cpu().numpy(), func_proba.cpu().numpy() > 0.5
            )
            f1_func = sklearn_f1_score(
                ego_gt_labels.cpu().numpy(), func_proba.cpu().numpy() > 0.5
            )
            roc_func = sklearn_roc_auc_score(
                ego_gt_labels.cpu().numpy(), func_proba.cpu().numpy()
            )

            conf_matrix_pruning, conf_matrix_func = compute_confusion_matrix(
                pred_scores_pos, pred_nodes_pos, gt_scores, func_proba, ego_gt_labels
            )
            # print(f"Accuracy: {acc}")
            # print(f"F1 Score: {f1}")
            print(f"Pruning ROC AUC Score: {roc}")
            print(f"Functionality ROC AUC Score: {roc_func}")
            # print(f"Pruning F1 Score: {f1}")
            # print(f"Functionality F1 Score: {f1_func}")
            print(f"Confusion Matrix Pruning at threshold=0.5: \n{conf_matrix_pruning}")
            print(
                f"Confusion Matrix Functionality at threshold=0.5: \n{conf_matrix_func}"
            )

        # metric.update()
        loss.backward()
        optimizer.step()

        print(f"Batch Loss: {loss.item()}")

        writer.add_scalar(
            "training pruning loss", pruning_loss.item(), epoch * len(train_loader) + i
        )
        writer.add_scalar(
            "training functionality loss",
            functionality_loss.item(),
            epoch * len(train_loader) + i,
        )

        writer.add_scalar(
            "training pruning ROC AUC", roc, epoch * len(train_loader) + i
        )

        writer.add_scalar(
            "training functionality ROC AUC", roc_func, epoch * len(train_loader) + i
        )

    ##### TESTING LOOP
    model.eval()
    test_loss = 0
    test_acc = 0
    for i, test_data in enumerate(test_loader):
        print("Testing loop")
        test_x, test_edge_index, test_batch = (
            test_data.x,
            test_data.edge_index,
            test_data.batch,
        )
        # Assuming 'forward' returns two lists of variable-sized tensors
        test_pred_scores, test_pred_nodes, test_func_proba = model.forward(
            test_x, test_edge_index, test_batch
        )

        # Ground truth: a single list of variable-sized tensors
        test_gt_scores = test_data.y
        # print("data.y", data.y)
        test_ego_gt_labels = test_data.ego_label
        test_ego_gt_labels = test_ego_gt_labels.unsqueeze(1)

        test_pred_scores_pos = list(compress(test_pred_scores, test_ego_gt_labels))
        test_pred_nodes_pos = list(compress(test_pred_nodes, test_ego_gt_labels))
        with torch.no_grad():
            test_pruning_loss = custom_bce_loss_pruning(
                test_pred_scores_pos,
                test_pred_nodes_pos,
                test_gt_scores,
            )  # calculating pruning loss only for positive anchors

            test_ego_gt_labels = test_ego_gt_labels.float()
            # print("func_proba", func_proba.squeeze())
            # print("ego_gt_labels", ego_gt_labels.squeeze())
            test_functionality_loss = bce_loss(test_func_proba, test_ego_gt_labels)
            test_loss = (
                test_pruning_loss + test_functionality_loss
            )  ##### WRONG NEEDS TO DO PRUNING LOSS ONLY FOR FUNCTIONAL REGIONS (P*=1)
            # Backpropagation
            print(
                "true test functionality labels of each anchor node",
                test_ego_gt_labels.squeeze(),
            )
            print(
                "predicted test functionality labels of each anchor node",
                test_func_proba.squeeze(),
            )

            test_acc = compute_accuracy_pruning(
                test_pred_scores_pos, test_pred_nodes_pos, test_gt_scores
            )
            # print("pred_scores_pos", pred_scores_pos)
            # print("gt_scores_pos", gt_scores_pos)
            test_f1 = compute_f1_score_pruning(
                test_pred_scores_pos, test_pred_nodes_pos, test_gt_scores
            )
            test_roc = compute_roc_auc_score_pruning(
                test_pred_scores_pos, test_pred_nodes_pos, test_gt_scores
            )

            test_acc_func = sklearn_accuracy_score(
                test_ego_gt_labels.cpu().numpy(), test_func_proba.cpu().numpy() > 0.5
            )
            test_f1_func = sklearn_f1_score(
                test_ego_gt_labels.cpu().numpy(), test_func_proba.cpu().numpy() > 0.5
            )
            test_roc_func = sklearn_roc_auc_score(
                test_ego_gt_labels.cpu().numpy(), test_func_proba.cpu().numpy()
            )

            conf_matrix_pruning, conf_matrix_func = compute_confusion_matrix(
                test_pred_scores_pos,
                test_pred_nodes_pos,
                test_gt_scores,
                test_func_proba,
                test_ego_gt_labels,
            )
        # print(f"Accuracy: {acc}")
        # print(f"F1 Score: {f1}")
        print(f"Test Pruning ROC AUC Score: {test_roc}")
        print(f"Test Functionality ROC AUC Score: {test_roc_func}")
        # print(f"Pruning F1 Score: {f1}")
        # print(f"Functionality F1 Score: {f1_func}")
        print(
            f"Test Confusion Matrix Pruning at threshold=0.5: \n{conf_matrix_pruning}"
        )
        print(
            f"Test Confusion Matrix Functionality at threshold=0.5: \n{conf_matrix_func}"
        )

        print(f"Batch Loss: {loss.item()}")

        writer.add_scalar(
            "testing pruning loss",
            test_pruning_loss.item(),
            epoch * len(test_loader) + i,
        )
        writer.add_scalar(
            "testing functionality loss",
            test_functionality_loss.item(),
            epoch * len(test_loader) + i,
        )

        writer.add_scalar(
            "testing pruning ROC AUC", test_roc, epoch * len(test_loader) + i
        )

        writer.add_scalar(
            "testing functionality ROC AUC", test_roc_func, epoch * len(test_loader) + i
        )


test_graph = data_list[1:2]
test_loader2 = DataLoader(test_graph, batch_size=32)
batch = torch.tensor(
    [0 for _ in range(len(test_graph[0].x))] + [1 for _ in range(len(test_graph[1].x))]
)


# print("PREDICTED NODES FOR TEST GRAPH", pred_nodes)
for i, test_data in enumerate(test_loader2):
    print("Testing loop")
    test_x, test_edge_index, test_batch = (
        test_data.x,
        test_data.edge_index,
        test_data.batch,
    )
    # Assuming 'forward' returns two lists of variable-sized tensors
    test_pred_scores, test_pred_nodes, test_func_proba = model.forward(
        test_x, test_edge_index, test_batch
    )
    get_proposed_candidates_batches(
        pred_scores,
        pred_nodes,
        func_proba,
        batch,
        func_threshold=0,
        node_score_threshold=0,
        check_overlap=False,
    )
