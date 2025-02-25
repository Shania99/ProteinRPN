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

from HEAL.network import CL_protNET
from HEAL.nt_xent import NT_Xent
from HEAL.utils import log
from HEAL.config import get_config
# from HEAL.evaluation_metrics import Metrics

from GO_data_preprocessing import GoTermDataset, collate_fn, load_GO_annot

from tqdm import tqdm
import pickle as pkl
warnings.filterwarnings("ignore")  ## need to change to output vector of 0 and 1

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



PATH = '/om2/user/shania/'

go_annot_path_pdb = PATH+'datasets/HEAL_data/nrPDB-GO_2019.06.18_annot.tsv'
go_annot_path_sm = PATH+'datasets/HEAL_data/nrSwiss-Model-GO_annot.tsv'

graph_list_pdb_t = PATH+'datasets/HEAL_data/processed/train_heal_graphs.pt'
graph_list_sm_t = PATH+'datasets/HEAL_data/processed/AF2train_heal_graphs.pt'

graph_list_pdb_v = PATH+'datasets/HEAL_data/processed/val_heal_graphs.pt'
graph_list_sm_v = PATH+'datasets/HEAL_data/processed/AF2val_heal_graphs.pt'

graph_list_pdb_ts = PATH+'datasets/HEAL_data/processed/test_heal_graphs.pt'
graph_list_sm_ts = PATH+'datasets/HEAL_data/processed/AF2test_heal_graphs.pt'

pdb_id_t = PATH+'datasets/HEAL_data/processed/train_pdbch.pt'
af2_id_t = PATH+'datasets/HEAL_data/processed/AF2train_pdbch.pt'

pdb_id_v = PATH+'datasets/HEAL_data/processed/val_pdbch.pt'
af2_id_v = PATH+'datasets/HEAL_data/processed/AF2val_pdbch.pt'

pdb_id_ts = PATH+'datasets/HEAL_data/processed/test_pdbch.pt'
af2_id_ts = PATH+'datasets/HEAL_data/processed/AF2test_pdbch.pt'

pdb_id_list_t = [torch.load(pdb_id_t)['train_pdbch'], torch.load(af2_id_t)['train_pdbch'], torch.load(af2_id_ts)['test_pdbch']]
pdb_id_list_v = [torch.load(pdb_id_v)['val_pdbch'], torch.load(af2_id_v)['val_pdbch']]
pdb_id_list_ts = torch.load(pdb_id_ts)['test_pdbch']

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

    # train_set = GoTermDataset("train", task, config.AF2model)
    train_set = GoTermDataset(
        annot_path=[go_annot_path_pdb, go_annot_path_sm, go_annot_path_sm],
        graph_list_file=[graph_list_pdb_t, graph_list_sm_t, graph_list_sm_ts],
        # graph_list_file="/om/user/layne_h/project/protein_function/datasets/GeneOntology/test_graphs.pt",
        pdb_id_list=[pdb_id_t, af2_id_t, af2_id_ts],
        suffix = ['train', 'train', 'test'],
        task=task,
    )
    # pos_weights = torch.tensor(train_set.pos_weights).float()
    # valid_set = GoTermDataset("val", task, config.AF2model)
    valid_set = GoTermDataset(
        annot_path=[go_annot_path_pdb, go_annot_path_sm],
        graph_list_file=[graph_list_pdb_v, graph_list_sm_v],
        pdb_id_list=[pdb_id_v, af2_id_v],
        suffix = ['val', 'val'],
        task=task,
    )
    
    test_set = GoTermDataset(
        annot_path = go_annot_path_pdb,
        graph_list_file=graph_list_pdb_ts,
        pdb_id_list=pdb_id_list_ts,
        task=task,
    )
    
    
    
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    
    output_dim = valid_set.y_true.shape[-1]


    # SEQUENCE OF STEPS:
    # SEQUENCE + CONTACT MAP -> ADD ESM EMBEDDINGS -> PROCESS GO TERM DATA (ASSOCIATE PROTEIN GRAPHS TO GO TERM LABELS)
    # PASS PROTEIN GRAPH THROUGH GRPN FORWARD (EDGE INDEX, EMBEDDINGS, BATCH) -> GET PREDICTIONS -> PROCESS TO GET CANDIDATES
    # PASS CANDIDATES AS FUNCTIONAL NODE ONE HOT VECTOR TO FUNCTIONAL NODE ATTENTION GNN -> GET PREDICTIONS
    # PASS THROUGH HEAL -> GET GO TERM PREDICTIONS -> LOSS

    # print(train_set[0][0].x)
    # print(train_set[0][0].x.size())
    esm_embed_dim = train_set[0][0].x.shape[1]
    grpn_hidden_dim = 256
    grpn_num_classes = 1
    
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
            model.train()
            # optimizer.zero_grad()
            # print("batch[0]", batch[0])
            # print("batch[1]", batch[1])
            if config.contrast:
                # print("config contrast enabled")
                data = batch[0].to(config.device)
                y_pred, g_feat1, g_feat2 = model(data)
                # print("inside contrast", y_pred.shape, g_feat1.shape, g_feat2.shape)
                y_true = batch[1].to(config.device)
                _loss = bce_loss(y_pred, y_true)  # * pos_weights.to(config.device)
                _loss = _loss.mean()
                criterion1 = NT_Xent(g_feat1.shape[0], 0.1, 1)
                cl_loss = 0.05 * criterion1(g_feat1, g_feat2)   
                # if cl_loss==0:
                #     cl_loss = 0.05 * criterion1(g_feat1, g_feat2)

                loss = _loss + cl_loss
            else:
                data = batch[0].to(config.device)

                y_pred = model(data)  #
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
        y_true_all = valid_set.y_true.float().reshape(-1)
        print("y_true_all shape", y_true_all.shape)
        with torch.no_grad():
            for idx_batch, batch in tqdm(enumerate(val_loader)):
                # if idx_batch > 4:
                #     break
                if config.contrast:
                    # y_pred, _, _ = model(batch[0].to(config.device))
                    data = batch[0].to(config.device)

                    y_pred, _, _ = model(data)
                    
                else:
                    data = batch[0].to(config.device)
                    y_pred = model(data)
                    # y_pred = model(batch[0].to(config.device))
                y_pred_all.append(y_pred)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            print("y_pred_all shape", y_pred_all.shape)
            eval_loss = bce_loss(y_pred_all, y_true_all[:len(y_pred_all)]).mean()

            aupr = metrics.average_precision_score(
                y_true_all.numpy()[:len(y_pred_all)], y_pred_all.numpy(), average="samples"
            )
            
            precision, recall, thresholds = metrics.precision_recall_curve(y_true_all.numpy(), y_pred_all.numpy())
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
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
                    y_pred, _, _ = model(data)
                else:
                    data = batch[0].to(config.device)
                    y_pred = model(data)
                    # y_pred = model(batch[0].to(config.device))
                y_pred_test.append(y_pred)

            y_pred_test = torch.cat(y_pred_test, dim=0).cpu().reshape(-1)

            # test_loss = bce_loss(y_pred_test, y_true_test).mean()

            aupr_test = metrics.average_precision_score(
                y_true_test.numpy(), y_pred_test.numpy(), average="samples"
            )
            
            precision, recall, thresholds = metrics.precision_recall_curve(y_true_test.numpy(), y_pred_test.numpy())
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores_test = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
            max_f1_test = np.max(f1_scores_test)
            
            test_aupr.append(aupr_test)
            test_Fmax.append(max_f1_test)
            log(
                f"{ith_epoch} TEST_epoch ||| aupr: {round(float(aupr_test),3)} ||| Fmax: {round(float(max_f1_test),3)}"
            )
            # val_loss.append(test_loss.numpy())
            
            # wandb.log({"aupr_test": aupr_test, "Fmax":max_f1_val, "eval_loss": val_loss[-1], "train_loss": train_loss[-1]})

            
            
            
            if ith_epoch == 0:
                best_eval_loss = eval_loss
                # best_eval_loss = aupr
            if eval_loss < best_eval_loss:
                # best_eval_loss = aupr
                best_eval_loss = eval_loss
                es = 0
                torch.save(
                    model.state_dict(), config.model_save_path + task + f"{suffix}_heal_baseline.pt"
                )
                
                ######## RESULTS DUMP
                result_name = f'output_dicts/heal_output_dict_train_{task}_{suffix}.pkl'
                with open(result_name, "wb") as f:
                    op_dict = {'Y_true': y_true_test.numpy(), 'Y_pred': y_pred_test.numpy(), 
                            'goterms': goterms[task], 'gonames': gonames[task], 'ontology':task}
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
    p.add_argument("--device", type=str, default="cpu", help="")
    p.add_argument("--esmembed", default=True, type=str2bool, help="")
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
    p.add_argument("--batch_size", type=int, default=48, help="")

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
    
    #### to call
    # cd /om2/user/shania/protein_func_new
    # python GO_combined_train.py --task bp --suffix grpn_heal_combined_ss_bp --device cuda --esmembed True --pooling MTP --contrast False --AF2model False --batch_size 48
