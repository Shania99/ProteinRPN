import pandas as pd
import torch
import networkx as nx
from tqdm import tqdm
from preprocess import (
    process_data,
    create_protein_graph,
    ego_graph_label,
    convert_pyg,
)

##### DATA PREPARATION
# filename = "PDBSite_results.txt"
filename = "pdbsite_old_all.txt"
data_list = []  # [Data(...), ..., Data(...)]
data_dict = process_data(filename=filename, encoding="utf-16")
df = pd.DataFrame(data_dict)
print(df.head()["y"])


for pdb_id in tqdm(df.pdb_id.unique()):
    protein_graph = create_protein_graph(PDB_ID=pdb_id, df=df, esm_embeddings=None)
    nx.draw(protein_graph, with_labels=True)
    # print(protein_graph.nodes(data=True))
    gt_anchor_membership = ego_graph_label(protein_graph)
    g = convert_pyg(protein_graph)
    data_list.append(g)


torch.save(data_list, "datasets/data_list.pt")
