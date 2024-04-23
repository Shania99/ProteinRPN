import os
import warnings

import torch

from GO_data_preprocessing import process_pdb

data_dir = "/om2/user/shania/datasets/GeneOntology"
out_dir = "/om2/group/kellislab/shared/struct2func/datasets/GeneOntology"
# esm_path = "/om2/group/kellislab/PLM/model_weights/esm-1b/esm1b_t33_650M_UR50S.pt"
esm_path = "/om2/group/kellislab/PLM/model_weights/esm2_t33_650M_UR50D.pt" 
n_jobs = 4


warnings.simplefilter("ignore")
os.makedirs(out_dir, exist_ok=True)

train_path = os.path.join(data_dir, "train")
train_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(train_path) for f in files
]

val_path = os.path.join(data_dir, "valid")
val_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(val_path) for f in files
]

test_path = os.path.join(data_dir, "test")
test_pdb = [
    os.path.join(pth, f) for pth, dirs, files in os.walk(test_path) for f in files
]

train_pdb = [i for i in train_pdb if os.path.isfile(i)][:5]
val_pdb = [i for i in val_pdb if os.path.isfile(i)][:5]
test_pdb = [i for i in test_pdb if os.path.isfile(i)][:5]

def construct_and_save(pdb_paths, output_prefix, n_jobs=n_jobs, esm_path=esm_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs, bad_paths = process_pdb(pdb_paths, n_jobs=n_jobs, esm_path=esm_path, device=device)
    print(output_prefix)
    print("Number of graphs", len(graphs))
    print("First graph dimensions", graphs[0])
    torch.save(graphs, f"{output_prefix}_graphs.pt")

    with open(f"{output_prefix}_bad_paths.txt", "w") as fh:
        fh.writelines(bad_paths)


print("processing train data")
construct_and_save(train_pdb, os.path.join(out_dir, "train_esm2_t"))

print("processing validation data")
construct_and_save(val_pdb, os.path.join(out_dir, "val_esm2_t"))

print("processing test data")
construct_and_save(test_pdb, os.path.join(out_dir, "test_esm2_t"))