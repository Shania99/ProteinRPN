import csv
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
import esm

from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from tqdm import tqdm

from my_utils import pmap_single


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
    data = Data(x=esm_embed, edge_index=edge_index, native_x=seq_code)
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

def get_sequences_and_edges_single(pdb_path, pdb_parser=None):
    if pdb_parser is None:
        pdb_parser = PDBParser()
    try:
        struct = pdb_parser.get_structure("x", pdb_path)
    except ValueError as e:
        print(f"got error {e} for path {pdb_path}, returning None")
        return None, None
    except PDBConstructionException as e:
        print(f"got error {e} for path {pdb_path}, returning None")
        return None, None

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
    if len(seq_idx_list) >= 1000:
        print(len(seq_idx_list))
    Ca_array = np.array(Ca_array)
    resi_num = Ca_array.shape[0]
    if resi_num <= 1:
        return None, None
    G = np.dot(Ca_array, Ca_array.T)
    H = np.tile(np.diag(G), (resi_num, 1))
    dismap = (H + H.T - 2 * G) ** 0.5

    row, col = np.where(dismap <= 10)
    edge = [row, col]
    return sequence, edge


def process_pdb(pdb_paths, n_jobs=None, device="cpu", esm_path=None, batch_size=128):
    parser = PDBParser()
    seqs_and_edges = pmap_single(get_sequences_and_edges_single, pdb_paths, n_jobs=n_jobs, verbose=1, pdb_parser=parser)

    bad_paths  = []
    seqs_filt = []
    edges_filt = []
    for i, (seq, edge) in enumerate(seqs_and_edges):
        if seq is None:
            bad_paths.append(pdb_paths[i])
            continue
        seqs_filt.append(seq)
        edges_filt.append(edge)
    print(f"dropped {len(seqs_and_edges) - len(seqs_filt)} proteins with bad inputs")

    if esm_path is None:
        # esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_path)
    esm_model.eval()
    esm_model = esm_model.to(device)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)

    num_batches = (len(seqs_filt) + batch_size - 1) // batch_size

    embeddings = []
    for batch_num in tqdm(range(num_batches)):
        start = batch_num * batch_size
        end = min(len(seqs_filt), (batch_num+1)*batch_size)
        batch_seqs = seqs_filt[start:end]
        _, _, batch_tokens = batch_converter(
            [(f"seq_{i}", seq) for i, seq in enumerate(batch_seqs)],
        )
        print(batch_tokens.shape)
        with torch.no_grad():
            results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
            token_representations = (
                results["representations"][33].detach().cpu()
            )
        embeddings.append(token_representations)
    embeddings = torch.cat(embeddings)
    # embeddings = torch.cat(embeddings).numpy()

    graphs = []
    print(len(seqs_filt), len(edges_filt), len(embeddings))
    for i in range(len(seqs_filt)):
        # print(i)
        graphs.append(protein_graph(
            seqs_filt[i], edges_filt[i], embeddings[i][1: min(len(seqs_filt[i])+1, 1022)]
        ))

    return graphs, bad_paths

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