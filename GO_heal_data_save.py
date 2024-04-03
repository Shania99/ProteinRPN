import sys

sys.path.insert(0, "/Users/shaniamitra/Downloads/Struct2Func_v-old/HEAL")

import os
import torch
from my_utils import pmap_single
from GO_data_preprocessing import process_pdb
import warnings


warnings.filterwarnings("ignore")  ## need to change to output vector of 0 and 1

train_path = (
    "/Users/shaniamitra/Downloads/Struct2Func_v-old/datasets/GeneOntology/train/"
)
# train_pdb = [os.path.abspath(os.path.join(train_path, p)) for p in os.listdir(train_path)]
train_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(train_path) for f in files
]

val_path = "/Users/shaniamitra/Downloads/Struct2Func_v-old/datasets/GeneOntology/valid/"
val_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(val_path) for f in files
]

test_path = "/Users/shaniamitra/Downloads/Struct2Func_v-old/datasets/GeneOntology/test/"
# train_pdb = [os.path.abspath(os.path.join(train_path, p)) for p in os.listdir(train_path)]
test_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(test_path) for f in files
]

train_pdb = [i for i in train_pdb if os.path.isfile(i)]
val_pdb = [i for i in val_pdb if os.path.isfile(i)]
test_pdb = [i for i in test_pdb if os.path.isfile(i)]


train_graphs = pmap_single(process_pdb, train_pdb, n_jobs=None, verbose=1)
torch.save(train_graphs, "datasets/train_graphs_GO.pt")


val_graphs = pmap_single(process_pdb, val_pdb, n_jobs=None, verbose=1)
torch.save(val_graphs, "datasets/val_graphs_GO.pt")

test_graphs = pmap_single(process_pdb, test_pdb, n_jobs=None, verbose=1)
torch.save(test_graphs, "datasets/test_graphs_GO.pt")
