import sys

sys.path.insert(0, "./HEAL")

import os
import warnings
import argparse
from sklearn import metrics
import torch
from torch.utils.data import DataLoader

from HEAL.network import CL_protNET
from HEAL.nt_xent import NT_Xent
from HEAL.utils import log
from HEAL.config import get_config


from GRPN import GraphRPN
from GO_data_preprocessing import GoTermDataset, collate_fn
from functional_node_attention import FunctionalNodeAttentionGNN
from GRPN_processing import get_proposed_candidates_batches

warnings.filterwarnings("ignore")  ## need to change to output vector of 0 and 1
pdb_data_path = "../datasets/GeneOntology" ## add in old file too


train_path = (
    f"{pdb_data_path}/train/"
)
# train_pdb = [os.path.abspath(os.path.join(train_path, p)) for p in os.listdir(train_path)]
train_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(train_path) for f in files
]

val_path = (
    f"{pdb_data_path}/valid/"
)
val_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(val_path) for f in files
]

test_path = train_path = (
    f"{pdb_data_path}/test/"
)
# train_pdb = [os.path.abspath(os.path.join(train_path, p)) for p in os.listdir(train_path)]
test_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(test_path) for f in files
]

train_pdb = [i for i in train_pdb if os.path.isfile(i)]
val_pdb = [i for i in val_pdb if os.path.isfile(i)]
test_pdb = [i for i in test_pdb if os.path.isfile(i)]

go_annot_path = "/om2/user/shania/datasets/GeneOntology/nrPDB-GO_annot.tsv"
train_pdb_names = [path.split("/")[-1].split("_")[0] for path in train_pdb]
val_pdb_names = [path.split("/")[-1].split("_")[0] for path in val_pdb]
test_pdb_names = [path.split("/")[-1].split("_")[0] for path in test_pdb]


def train(config, task, suffix):

    # data_path = "/om2/group/kellislab/shared/struct2func/datasets/GeneOntology/"
    data_path = "/om/user/layne_h/project/protein_function/datasets/GeneOntology"
    # train_set = GoTermDataset("train", task, config.AF2model)
    train_set = GoTermDataset(
        annot_path=go_annot_path,
        graph_list_file=f"{data_path}/train_graphs.pt",
        pdb_id_list=train_pdb_names,
        task=task,
    )
    pos_weights = torch.tensor(train_set.pos_weights).float()
    # valid_set = GoTermDataset("val", task, config.AF2model)
    valid_set = GoTermDataset(
        annot_path=go_annot_path,
        graph_list_file=f"{data_path}/val_graphs.pt",
        pdb_id_list=val_pdb_names,
        task=task,
    )
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    output_dim = valid_set.y_true.shape[-1]

    # SEQUENCE OF STEPS:
    # SEQUENCE + CONTACT MAP -> ADD ESM EMBEDDINGS -> PROCESS GO TERM DATA (ASSOCIATE PROTEIN GRAPHS TO GO TERM LABELS)
    # PASS PROTEIN GRAPH THROUGH GRPN FORWARD (EDGE INDEX, EMBEDDINGS, BATCH) -> GET PREDICTIONS -> PROCESS TO GET CANDIDATES
    # PASS CANDIDATES AS FUNCTIONAL NODE ONE HOT VECTOR TO FUNCTIONAL NODE ATTENTION GNN -> GET PREDICTIONS
    # PASS THROUGH HEAL -> GET GO TERM PREDICTIONS -> LOSS

    esm_embed_dim = train_set[0][0].x.shape[1]
    grpn_hidden_dim = 256
    grpn_num_classes = 1
    grpn = GraphRPN(
        k=2, input_dim=esm_embed_dim, hidden_dim=256, num_classes=1
    ).to(config.device)  # output: pred_scores, pred_nodes, func_proba
    func_attention = FunctionalNodeAttentionGNN(
        in_channels=esm_embed_dim, out_channels=esm_embed_dim, device=config.device
    ).to(config.device)  # input: node scores, x, edge_index, batch
    # output: x
    model = CL_protNET(output_dim, True, config.pooling, config.contrast).to(
        config.device
    )  # input: input dim, output dim, change data in forward to x, edge_index, batch
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **config.optimizer,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler)
    bce_loss = torch.nn.BCELoss(reduce=False)

    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []
    es = 0
    y_true_all = valid_set.y_true.float().reshape(-1)

    for ith_epoch in range(config.max_epochs):
        # scheduler.step()
        for idx_batch, batch in enumerate(train_loader):
            # with torch.autograd.set_detect_anomaly(True):
            model.train()
            # optimizer.zero_grad()
            # print("batch[0]", batch[0])
            # print("batch[1]", batch[1])
            if config.contrast:
                y_pred, g_feat1, g_feat2 = model(batch[0].to(config.device))
                y_true = batch[1].to(config.device)
                _loss = bce_loss(y_pred, y_true)  # * pos_weights.to(config.device)
                _loss = _loss.mean()
                criterion = NT_Xent(g_feat1.shape[0], 0.1, 1)
                cl_loss = 0.05 * criterion(g_feat1, g_feat2)

                loss = _loss + cl_loss
            else:
                data = batch[0].to(config.device)
                esm_embeddings, native_x, edge_index, batch_vec = (
                    data.x,
                    data.native_x,
                    data.edge_index,
                    data.batch,
                )
                print(esm_embeddings.shape, native_x.shape, edge_index.shape, batch_vec.shape)
                print('Max index in edge_index:', edge_index.max().item())
                print('Number of nodes:', esm_embeddings.size(0))
                
                # Ensure no indices are greater than or equal to the number of nodes
                # assert edge_index.max() < esm_embeddings.size(0), "edge_index contains out-of-bounds indices"

                pred_node_scores, pred_nodes, func_proba = grpn(
                    esm_embeddings, edge_index, batch_vec
                )
                # node_functional_vector = get_proposed_candidates(
                #     pred_node_scores, pred_nodes, func_proba
                # )  # TODO: change definition

                node_functional_vector = get_proposed_candidates_batches(
                    pred_node_scores, pred_nodes, func_proba, batch_vec
                )
                x = func_attention(
                    esm_embeddings, edge_index, node_functional_vector, batch_vec
                )  # needs to handle batches of vectors and node func vector

                y_pred = model(x, native_x, edge_index, batch_vec)  #
                # y_pred = y_pred.reshape([-1,2])
                y_true = batch[1].to(config.device)  # .reshape([-1])
                # print("y_pred", y_pred)
                # print("y_true", y_true)
                loss = bce_loss(y_pred, y_true)  # * pos_weights.to(config.device)
                loss = loss.mean()
                # loss = mlsm_loss(y_pred, y_true)

            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)}")
            train_loss.append(loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_loss = 0
        model.eval()
        y_pred_all = []
        n_nce_all = []

        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                if config.contrast:
                    y_pred, _, _ = model(batch[0].to(config.device))
                else:
                    data = batch[0].to(config.device)
                    esm_embeddings, native_x, edge_index, batch_vec = (
                        data.x,
                        data.native_x,
                        data.edge_index,
                        data.batch,
                    )
                    pred_node_scores, pred_nodes, func_proba = grpn(
                        esm_embeddings, edge_index, batch_vec
                    )
                    node_functional_vector = get_proposed_candidates_batches(
                        pred_node_scores, pred_nodes, func_proba, batch_vec
                    )
                    x = func_attention(
                        esm_embeddings, edge_index, node_functional_vector, batch_vec
                    )  # needs to handle batches of vectors and node func vector

                    y_pred = model(x, native_x, edge_index, batch_vec).to(config.device)
                    # y_pred = model(batch[0].to(config.device))
                y_pred_all.append(y_pred)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)

            eval_loss = bce_loss(y_pred_all, y_true_all).mean()

            aupr = metrics.average_precision_score(
                y_true_all.numpy(), y_pred_all.numpy(), average="samples"
            )
            val_aupr.append(aupr)
            log(
                f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)}"
            )
            val_loss.append(eval_loss.numpy())
            if ith_epoch == 0:
                best_eval_loss = eval_loss
                # best_eval_loss = aupr
            if eval_loss < best_eval_loss:
                # best_eval_loss = aupr
                best_eval_loss = eval_loss
                es = 0
                torch.save(
                    model.state_dict(), config.model_save_path + task + f"{suffix}.pt"
                )
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
                    },
                    config.loss_save_path + task + f"{suffix}.pt",
                )

                break


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
    p.add_argument("--suffix", type=str, default="", help="")
    p.add_argument("--device", type=str, default="", help="")
    p.add_argument("--esmembed", default=False, type=str2bool, help="")
    p.add_argument(
        "--pooling",
        default="MTP",
        type=str,
        choices=["MTP", "GMP"],
        help="Multi-set transformer pooling or Global max pooling",
    )
    p.add_argument(
        "--contrast",
        default=False,
        type=str2bool,
        help="whether to do contrastive learning",
    )
    p.add_argument(
        "--AF2model",
        default=False,
        type=str2bool,
        help="whether to use AF2model for training",
    )
    p.add_argument("--batch_size", type=int, default=32, help="")

    args = p.parse_args()
    config = get_config()
    config.optimizer["lr"] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if args.device != "":
        config.device = args.device
    config.esmembed = args.esmembed
    print(args)
    config.pooling = args.pooling
    config.contrast = args.contrast
    config.AF2model = args.AF2model
    train(config, args.task, args.suffix)
