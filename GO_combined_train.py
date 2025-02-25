## change functions, add dense to sparse

import sys
import numpy as np

sys.path.insert(0, "./HEAL")
sys.path.insert(0, "./modules")
sys.path.insert(0, "./utils")

import os
import warnings
import argparse
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import torch.nn as nn

from modules.heal_network import CL_protNET
from HEAL.nt_xent import NT_Xent
from modules.supcon import SupConLoss
from HEAL.utils import log
from HEAL.config import get_config
from modules.connected_component_loss import connected_components_loss

# from HEAL.evaluation_metrics import Metrics

from modules.GRPN import GraphRPN
from GO_data_preprocessing import GoTermDataset, collate_fn, load_GO_annot
from modules.functional_node_attention import FunctionalNodeAttentionGNN

from tqdm import tqdm
import pickle as pkl

warnings.filterwarnings("ignore")  ## need to change to output vector of 0 and 1


# import os
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="protein-structure-prediction-refactored",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 1e-4,
#     "architecture": "GRPN+HEAL",
#     "dataset": "GeneOntology",
#     "epochs": 100,
#     }
# )


PATH = "/om2/user/shania/"

go_annot_path_pdb = PATH + "datasets/HEAL_data/nrPDB-GO_2019.06.18_annot.tsv"
go_annot_path_sm = PATH + "datasets/HEAL_data/nrSwiss-Model-GO_annot.tsv"

graph_list_pdb_t = (
    PATH + "datasets/HEAL_data/processed/train_heal_graphs_pdb_ss_coords.pt"
)
graph_list_sm_t = (
    PATH + "datasets/HEAL_data/processed/AF2train_heal_graphs_pdb_ss_coords.pt"
)

graph_list_pdb_v = (
    PATH + "datasets/HEAL_data/processed/val_heal_graphs_pdb_ss_coords.pt"
)
graph_list_sm_v = (
    PATH + "datasets/HEAL_data/processed/AF2val_heal_graphs_pdb_ss_coords.pt"
)

graph_list_pdb_ts = (
    PATH + "datasets/HEAL_data/processed/test_heal_graphs_pdb_ss_coords.pt"
)
graph_list_sm_ts = (
    PATH + "datasets/HEAL_data/processed/AF2test_heal_graphs_pdb_ss_coords.pt"
)

pdb_id_t = PATH + "datasets/HEAL_data/processed/train_pdbch.pt"
af2_id_t = PATH + "datasets/HEAL_data/processed/AF2train_pdbch.pt"

pdb_id_v = PATH + "datasets/HEAL_data/processed/val_pdbch.pt"
af2_id_v = PATH + "datasets/HEAL_data/processed/AF2val_pdbch.pt"

pdb_id_ts = PATH + "datasets/HEAL_data/processed/test_pdbch.pt"
af2_id_ts = PATH + "datasets/HEAL_data/processed/AF2test_pdbch.pt"

pdb_id_list_t = [
    torch.load(pdb_id_t)["train_pdbch"],
    torch.load(af2_id_t)["train_pdbch"],
    torch.load(af2_id_ts)["test_pdbch"],
]
pdb_id_list_v = [torch.load(pdb_id_v)["val_pdbch"], torch.load(af2_id_v)["val_pdbch"]]
pdb_id_list_ts = torch.load(pdb_id_ts)["test_pdbch"]

_, goterms, gonames, _ = load_GO_annot(go_annot_path_pdb)


def check_attribute_exists(batch, attribute_name):
    """
    Check if the attribute exists in any of the data objects within the batch.

    Parameters:
    batch (Batch): A batch of PyG data objects.
    attribute_name (str): The name of the attribute to check for.

    Returns:
    bool: True if the attribute exists in any data object, False otherwise.
    """
    for data in batch.to_data_list():
        if hasattr(data, attribute_name):
            return True
    return False


def train(config, task, suffix):

    train_set = GoTermDataset(
        annot_path=[go_annot_path_pdb, go_annot_path_sm, go_annot_path_sm],
        graph_list_file=[graph_list_pdb_t, graph_list_sm_t, graph_list_sm_ts],
        # graph_list_file="/om/user/layne_h/project/protein_function/datasets/GeneOntology/test_graphs.pt",
        pdb_id_list=[pdb_id_t, af2_id_t, af2_id_ts],
        suffix=["train", "train", "test"],
        task=task,
    )

    valid_set = GoTermDataset(
        annot_path=[go_annot_path_pdb, go_annot_path_sm],
        graph_list_file=[graph_list_pdb_v, graph_list_sm_v],
        pdb_id_list=[pdb_id_v, af2_id_v],
        suffix=["val", "val"],
        task=task,
    )

    test_set = GoTermDataset(
        annot_path=go_annot_path_pdb,
        graph_list_file=graph_list_pdb_ts,
        pdb_id_list=pdb_id_list_ts,
        task=task,
    )

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    output_dim = valid_set.y_true.shape[-1]

    # print(train_set[0][0].x.size())
    esm_embed_dim = train_set[0][0].x.shape[1]
    grpn_hidden_dim = 256

    # init__(self, k, input_dim, hidden_dim, output_dim)
    grpn = GraphRPN(
        k=2,
        input_dim=esm_embed_dim,
        hidden_dim=grpn_hidden_dim,
    ).to(
        config.device
    )  # output: pred_scores, pred_nodes, func_proba

    func_attention = FunctionalNodeAttentionGNN(
        in_channels=esm_embed_dim, out_channels=esm_embed_dim
    ).to(
        config.device
    )  # input: node scores, x, edge_index, batch
    # output: x
    model = CL_protNET(
        out_dim=output_dim,
        esm_embed=True,
        pooling="MTP",
        pertub=config.contrast,
        polynormer=config.polynormer,
    ).to(config.device)

    # input: input dim, output dim, change data in forward to x, edge_index, batch

    # if config.pretrained:
    #     model_path = '/om2/user/shania/protein_func_new/HEAL/model/model_bpCLaf.pt'
    #     saved_state_dict = torch.load(model_path)

    #     # Create a new state_dict with the correct keys
    #     new_state_dict = {}
    #     for key, value in saved_state_dict.items():
    #         new_key = key.replace('weight', 'lin.weight')  # Adjust the key names
    #         new_state_dict[new_key] = value

    #     model.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.Adam(
        params=list(grpn.parameters())
        + list(func_attention.parameters())
        + list(model.parameters()),
        **config.optimizer,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler)
    bce_loss = torch.nn.BCELoss(reduce=False)

    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []
    test_loss = []
    test_aupr = []
    test_Fmax = []
    es = 0

    print(valid_set.y_true.shape)
    print(len(valid_set.graph_list))
    for ith_epoch in range(config.max_epochs):
        # scheduler.step()
        for idx_batch, batch in enumerate(train_loader):
            # if idx_batch>5:
            #     break
            # with torch.autograd.set_detect_anomaly(True):
            grpn.train()
            func_attention.train()
            model.train()
            # optimizer.zero_grad()
            if config.contrast:
                # print("config contrast enabled")
                data = batch[0].to(config.device)
                esm_embeddings, native_x, edge_index, batch_vec = (
                    data.x,
                    data.native_x,
                    data.edge_index,
                    data.batch,
                )

                x = torch.tensor(esm_embeddings).to(config.device)
                adj = to_dense_adj(edge_index, batch_vec)

                if not check_attribute_exists(data, "ss"):
                    x, adj, node_drop, func_prob = grpn(x, adj, batch, native_x)

                else:
                    x, adj, node_drop, func_prob = grpn(
                        x, adj, batch, native_x, ss_tensor=data.ss, coords=data.coords
                    )

                edge_index, edge_weight = dense_to_sparse(adj)

                x = func_attention(x, edge_index, edge_weight, func_prob, batch)

                y_pred, g_feat1, g_feat2 = model(x, native_x, edge_index, batch_vec)
                y_true = batch[1].to(config.device)

                _loss = bce_loss(y_pred, y_true)  # * pos_weights.to(config.device)
                _loss = _loss.mean()
                criterion1 = NT_Xent(g_feat1.shape[0], 0.1, 1)
                criterion2 = SupConLoss(g_feat2.shape[0], temperature=0.1, world_size=1)
                cc_loss = connected_components_loss(
                    node_drop.cpu(), edge_index.cpu(), batch_vec.cpu(), y_true.cpu()
                )
                cl_loss = 0.05 * criterion2(g_feat1, y_true) + 0.05 * criterion1(
                    g_feat1, g_feat2
                )

                loss = _loss + cl_loss + 0.01 * cc_loss
            else:
                data = batch[0].to(config.device)
                esm_embeddings, native_x, edge_index, batch_vec = (
                    data.x,
                    data.native_x,
                    data.edge_index,
                    data.batch,
                )
                x = torch.tensor(esm_embeddings).to(config.device)
                adj = to_dense_adj(edge_index, batch_vec)
                ##### CHANGE
                if not check_attribute_exists(data, "ss"):
                    x, adj, node_drop, func_prob = grpn(x, adj, batch, native_x)
                else:
                    # print("ss tensor present")
                    x, adj, node_drop, func_prob = grpn(
                        x, adj, batch, native_x, ss_tensor=data.ss, coords=data.coords
                    )

                edge_index, edge_weight = dense_to_sparse(adj)
                x = func_attention(x, edge_index, edge_weight, func_prob, batch)

                y_pred = model(x, native_x, edge_index, batch_vec)  #
                # y_pred = y_pred.reshape([-1,2])
                y_true = batch[1].to(config.device)  # .reshape([-1])
                # print("y_pred", y_pred)
                # print("y_true", y_true)
                loss = bce_loss(y_pred, y_true)  # * pos_weights.to(config.device)
                cc_loss = connected_components_loss(
                    node_drop.cpu(), edge_index.cpu(), batch_vec.cpu(), y_true.cpu()
                )
                loss = loss.mean() + 0.01 * cc_loss

            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)}")
            train_loss.append(loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_loss = 0
        grpn.eval()
        func_attention.eval()
        model.eval()
        y_pred_all = []
        n_nce_all = []
        y_true_all = valid_set.y_true.float().reshape(-1)
        print("y_true_all shape", y_true_all.shape)
        with torch.no_grad():
            for idx_batch, batch in tqdm(enumerate(val_loader)):
                # if idx_batch > 4:
                #     break
                if config.contrast:
                    # y_pred, _, _ = model(batch[0].to(config.device))
                    data = batch[0].to(config.device)
                    x, native_x, edge_index, batch_vec = (
                        data.x,
                        data.native_x,
                        data.edge_index,
                        data.batch,
                    )
                    adj = to_dense_adj(edge_index, batch_vec)

                    if not check_attribute_exists(data, "ss"):
                        x, adj, node_drop, func_prob = grpn(x, adj, batch, native_x)
                    else:
                        x, adj, node_drop, func_prob = grpn(
                            x,
                            adj,
                            batch,
                            native_x,
                            ss_tensor=data.ss,
                            coords=data.coords,
                        )

                    edge_index, edge_weight = dense_to_sparse(adj)
                    x = func_attention(x, edge_index, edge_weight, func_prob, batch)

                    y_pred, _, _ = model(x, native_x, edge_index, batch_vec)

                else:
                    data = batch[0].to(config.device)
                    esm_embeddings, native_x, edge_index, batch_vec = (
                        data.x,
                        data.native_x,
                        data.edge_index,
                        data.batch,
                    )
                    adj = to_dense_adj(edge_index, batch_vec)

                    if not check_attribute_exists(data, "ss"):
                        x, adj, node_drop, func_prob = grpn(x, adj, batch, native_x)
                    else:
                        x, adj, node_drop, func_prob = grpn(
                            x,
                            adj,
                            batch,
                            native_x,
                            ss_tensor=data.ss,
                            coords=data.coords,
                        )

                    edge_index, edge_weight = dense_to_sparse(adj)
                    x = func_attention(x, edge_index, edge_weight, func_prob, batch)

                    y_pred = model(x, native_x, edge_index, batch_vec).to(config.device)
                    # y_pred = model(batch[0].to(config.device))
                y_pred_all.append(y_pred)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            print("y_pred_all shape", y_pred_all.shape)
            eval_loss = bce_loss(y_pred_all, y_true_all[: len(y_pred_all)]).mean()

            aupr = metrics.average_precision_score(
                y_true_all.numpy()[: len(y_pred_all)],
                y_pred_all.numpy(),
                average="samples",
            )

            precision, recall, thresholds = metrics.precision_recall_curve(
                y_true_all.numpy(), y_pred_all.numpy()
            )
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores = np.divide(
                numerator, denom, out=np.zeros_like(denom), where=(denom != 0)
            )
            max_f1_val = np.max(f1_scores)

            val_aupr.append(aupr)
            val_Fmax.append(max_f1_val)
            log(
                f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)} ||| Fmax: {round(float(max_f1_val),3)}"
            )
            val_loss.append(eval_loss.numpy())

            # wandb.log({"aupr": aupr, "Fmax":max_f1_val, "eval_loss": val_loss[-1], "train_loss": train_loss[-1]})

        ########## TEST

        test_loss = 0
        grpn.eval()
        func_attention.eval()
        model.eval()
        y_pred_test = []
        n_nce_all = []
        y_true_test = test_set.y_true.float().reshape(-1)
        with torch.no_grad():
            for idx_batch, batch in tqdm(enumerate(test_loader)):
                # if idx_batch > 4:
                #     break
                if config.contrast:
                    # y_pred, _, _ = model(batch[0].to(config.device))
                    data = batch[0].to(config.device)
                    x, native_x, edge_index, batch_vec = (
                        data.x,
                        data.native_x,
                        data.edge_index,
                        data.batch,
                    )

                    adj = to_dense_adj(edge_index, batch_vec)

                    if not check_attribute_exists(data, "ss"):
                        x, adj, node_drop, func_prob = grpn(x, adj, batch, native_x)
                    else:
                        x, adj, node_drop, func_prob = grpn(
                            x,
                            adj,
                            batch,
                            native_x,
                            ss_tensor=data.ss,
                            coords=data.coords,
                        )

                    edge_index, edge_weight = dense_to_sparse(adj)
                    x = func_attention(x, edge_index, edge_weight, func_prob, batch)
                    y_pred, _, _ = model(x, native_x, edge_index, batch_vec)

                else:
                    data = batch[0].to(config.device)
                    esm_embeddings, native_x, edge_index, batch_vec = (
                        data.x,
                        data.native_x,
                        data.edge_index,
                        data.batch,
                    )

                    adj = to_dense_adj(edge_index, batch_vec)

                    if not check_attribute_exists(data, "ss"):
                        x, adj, node_drop, func_prob = grpn(x, adj, batch, native_x)
                    else:
                        # print("ss tensor present")
                        x, adj, node_drop, func_prob = grpn(
                            x,
                            adj,
                            batch,
                            native_x,
                            ss_tensor=data.ss,
                            coords=data.coords,
                        )

                    edge_index, edge_weight = dense_to_sparse(adj)
                    x = func_attention(x, edge_index, edge_weight, func_prob, batch)

                    y_pred = model(x, native_x, edge_index, batch_vec).to(config.device)
                y_pred_test.append(y_pred)

            y_pred_test = torch.cat(y_pred_test, dim=0).cpu().reshape(-1)

            # test_loss = bce_loss(y_pred_test, y_true_test).mean()

            aupr_test = metrics.average_precision_score(
                y_true_test.numpy(), y_pred_test.numpy(), average="samples"
            )

            precision, recall, thresholds = metrics.precision_recall_curve(
                y_true_test.numpy(), y_pred_test.numpy()
            )
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores_test = np.divide(
                numerator, denom, out=np.zeros_like(denom), where=(denom != 0)
            )
            max_f1_test = np.max(f1_scores_test)

            test_aupr.append(aupr_test)
            test_Fmax.append(max_f1_test)
            log(
                f"{ith_epoch} TEST_epoch ||| aupr: {round(float(aupr_test),3)} ||| Fmax: {round(float(max_f1_test),3)}"
            )
            # val_loss.append(test_loss.numpy())

            # wandb.log({"aupr_test": aupr_test, "Fmax":max_f1_val, "eval_loss": val_loss[-1], "train_loss": train_loss[-1]})

            if ith_epoch == 0:
                best_eval_fmax = max_f1_val
                # best_eval_loss = aupr
            if max_f1_val > best_eval_fmax:
                # best_eval_loss = aupr
                best_eval_fmax = max_f1_val
                es = 0
                torch.save(
                    grpn.state_dict(),
                    config.model_save_path + task + f"{suffix}_grpn.pt",
                )
                torch.save(
                    func_attention.state_dict(),
                    config.model_save_path + task + f"{suffix}_func_attn.pt",
                )
                torch.save(
                    model.state_dict(),
                    config.model_save_path + task + f"{suffix}_heal.pt",
                )

                ######## RESULTS DUMP
                result_name = f"output_dicts/grpn_output_dict_train_{task}_{suffix}.pkl"
                with open(result_name, "wb") as f:
                    op_dict = {
                        "Y_true": y_true_test.numpy(),
                        "Y_pred": y_pred_test.numpy(),
                        "goterms": goterms[task],
                        "gonames": gonames[task],
                        "ontology": task,
                    }
                    pkl.dump(op_dict, f)

            else:
                es += 1
                print("Counter {} of 5".format(es))

                # torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
            if es > 4:

                torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                        "val_fmax": val_Fmax,
                    },
                    config.loss_save_path + task + f"{suffix}.pt",
                )

                # break
    return max_f1_val, max_f1_test, op_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == "True" or v == "true":
        return True
    if v == "False" or v == "false":
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", type=str, default="bp", choices=["bp", "mf", "cc"], help=""
    )
    p.add_argument(
        "--suffix", type=str, default="test_bp", help="suffix to designate model"
    )
    p.add_argument("--device", type=str, default="cpu", help="")

    p.add_argument(
        "--contrast",
        default=False,
        type=str2bool,
        help="whether to do contrastive learning",
    )
    p.add_argument("--batch_size", type=int, default=48, help="")
    p.add_argument(
        "--polynormer",
        type=str2bool,
        default=False,
        help="whether to use polynormer instead of GMT",
    )
    p.add_argument(
        "--model_save_path",
        type=str,
        default="/om2/user/shania/protein_func_new/model_weights",
    )

    args = p.parse_args()
    config = get_config()
    config.optimizer["lr"] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if args.device != "":
        config.device = args.device
    print(args)

    config.contrast = args.contrast
    # config.pen_attn = args.pen_attn
    # config.pretrained = args.pretrained
    config.polynormer = args.polynormer
    config.model_save_path = args.model_save_path

    train(config, args.task, args.suffix)
