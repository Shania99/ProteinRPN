import csv
import numpy as np
import torch
from Bio.PDB import PDBParser
import esm
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def load_GO_annot(filename):
    # Load GO annotations
    onts = ["mf", "bp", "cc"]
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode="r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [
                    goterms[onts[i]].index(goterm)
                    for goterm in prot_goterms[i].split(",")
                    if goterm != ""
                ]
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts


RES2ID = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "-": 20,
}


def aa2idx(seq):
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype="|S1").view(np.uint8)
    idx = np.array(list(seq), dtype="|S1").view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx


def protein_graph(sequence, edge_index, esm_embed):
    seq_code = aa2idx(sequence)
    seq_code = torch.IntTensor(seq_code)
    # add edge to pairs whose distances are more possible under 8.25
    # row, col = edge_index
    edge_index = torch.LongTensor(edge_index)
    # if AF_embed == None:
    #     data = Data(x=seq_code, edge_index=edge_index)
    # else:
    data = Data(x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code)
    return data


# Assuming the restype_1to3 and restype_3to1 dictionaries are defined earlier in the code
restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


restype_3to1 = {v: k for k, v in restype_1to3.items()}


def process_pdb(pdb_path, device="cpu"):
    parser = PDBParser()
    struct = parser.get_structure("x", pdb_path)
    model = struct[0]
    chain_id = list(model.child_dict.keys())[0]
    chain = model[chain_id]
    Ca_array = []
    sequence = ""
    seq_idx_list = list(chain.child_dict.keys())

    for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1] + 1):
        try:
            Ca_array.append(chain[(" ", idx, " ")]["CA"].get_coord())
            sequence += restype_3to1[chain[(" ", idx, " ")].get_resname()]
        except:
            Ca_array.append([np.nan, np.nan, np.nan])
            sequence += "X"

    Ca_array = np.array(Ca_array)
    resi_num = Ca_array.shape[0]
    if resi_num <= 1:
        return None
    G = np.dot(Ca_array, Ca_array.T)
    H = np.tile(np.diag(G), (resi_num, 1))
    dismap = (H + H.T - 2 * G) ** 0.5

    # device = f"cuda:{device_id}"
    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()

    batch_labels, batch_strs, batch_tokens = batch_converter([("tmp", sequence[:1022])])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = (
            results["representations"][33][0].cpu().numpy().astype(np.float16)
        )
        esm_embed = token_representations[1 : len(sequence) + 1]

    row, col = np.where(dismap <= 10)
    edge = [row, col]
    graph = protein_graph(
        sequence, edge, esm_embed
    )  # Assuming protein_graph is defined elsewhere

    return graph


def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()


class GoTermDataset(Dataset):

    def __init__(self, annot_path, graph_list_file, pdb_id_list, task="mf"):
        # task can be among ['bp','mf','cc']
        self.task = task

        prot2annot, goterms, gonames, counts = load_GO_annot(annot_path)
        goterms = goterms[self.task]
        gonames = gonames[self.task]
        self.pdb_id_list = pdb_id_list
        output_dim = len(goterms)
        class_sizes = counts[self.task]
        mean_class_size = np.mean(class_sizes)
        pos_weights = mean_class_size / class_sizes
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
        # pos_weights = np.concatenate([pos_weights.reshape((len(pos_weights), 1)), pos_weights.reshape((len(pos_weights), 1))], axis=-1)
        # give weight for the 0/1 classification
        # pos_weights = {i: {0: pos_weights[i, 0], 1: pos_weights[i, 1]} for i in range(output_dim)}

        self.pos_weights = torch.tensor(pos_weights).float()

        self.graph_list = torch.load(graph_list_file)

        # self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        self.y_true = np.stack(
            [prot2annot[pdb_c][self.task] for pdb_c in self.pdb_id_list]
        )
        self.y_true = torch.tensor(self.y_true)

    def __getitem__(self, idx):

        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)
