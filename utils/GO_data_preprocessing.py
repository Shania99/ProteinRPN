import csv
import numpy as np
import torch
from Bio import PDB
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




def get_ss_tensor(pdb_file, structure=None):
    if structure is None:
        parser = PDB.PDBParser()
        structure = parser.get_structure("protein", pdb_file)
    try:    
        dssp = PDB.DSSP(model=structure[0], in_file=pdb_file, dssp='mkdssp')
        
        ss_tensor = []
        for key in dssp.keys():
            ss = dssp[key][2]
            if ss == 'H':  # Alpha helix
                ss_tensor.append(1)
            elif ss == 'E' or ss == 'B':  # Beta sheet or beta bridge
                ss_tensor.append(1)
            else:  # Coil, turn, bend, etc.
                ss_tensor.append(0)
        ss_tensor = np.array(ss_tensor, dtype=np.int64)
    except:
        model = structure[0]
        chain_id = list(model.child_dict.keys())[0]
        chain = model[chain_id]
        res_num = len([_ for _ in chain.get_residues() if PDB.is_aa(_)])
        ss_tensor = np.zeros(res_num)
    return ss_tensor

def get_ss_tensor_chain(pdb_file, chain_id, structure=None):
    if structure is None:
        parser = PDB.PDBParser()
        structure = parser.get_structure("protein", pdb_file)
    try:    
        dssp = PDB.DSSP(model=structure[0], in_file=pdb_file, dssp='mkdssp')
        
        ss_tensor = []
        for key in dssp.keys():
            if key[0] == chain_id:
                ss = dssp[key][2]
                if ss == 'H':  # Alpha helix
                    ss_tensor.append(1)
                elif ss == 'E' or ss == 'B':  # Beta sheet or beta bridge
                    ss_tensor.append(1)
                else:  # Coil, turn, bend, etc.
                    ss_tensor.append(0)
        ss_tensor = np.array(ss_tensor, dtype=np.int64)
    except:
        model = structure[0]
        chain_id = list(model.child_dict.keys())[0]
        chain = model[chain_id]
        res_num = len([_ for _ in chain.get_residues() if PDB.is_aa(_)])
        ss_tensor = np.zeros(res_num)
    return ss_tensor





def aa2idx(seq):
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype="|S1").view(np.uint8)
    idx = np.array(list(seq), dtype="|S1").view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx


def protein_graph(sequence, edge_index, esm_embed, ss_tensor=None, Ca_array=None):
    seq_code = aa2idx(sequence)
    esm_embed = torch.tensor(esm_embed)
    # print("Inside protein_graph esm_embed.shape, sequence.shape", esm_embed.shape, seq_code.shape)
    # assert esm_embed.shape[0] == seq_code.shape[0]
    seq_code = torch.IntTensor(seq_code)
    # add edge to pairs whose distances are more possible under 8.25
    # row, col = edge_index
    edge_index = torch.LongTensor(edge_index)
    # print("inside protein graph ss_tensor Ca_array", ss_tensor, Ca_array)
    if ss_tensor is not None:
        ss_tensor = torch.IntTensor(ss_tensor)
    if Ca_array is not None:
        Ca_array = torch.LongTensor(Ca_array)
        
    # if AF_embed == None:
    #     data = Data(x=seq_code, edge_index=edge_index)
    # else:
    # print("esm_embedding.shape, seq len", esm_embed.shape, seq_code.shape)
    # if esm_embed.shape[0]>=1021:
    #     print("1022 case")
    #     print("Edge index max", edge_index.max())
    #     print("seqeunce length esm", esm_embed.shape[0])
    # else:
    #     print("-")
    # assert edge_index.max() < esm_embed.shape[0], "esm embed shape lower"
    
    if Ca_array is None:
        data = Data(x=esm_embed, edge_index=edge_index, native_x=seq_code)
    else:
        data = Data(x=esm_embed, edge_index=edge_index, native_x=seq_code, Ca_array=Ca_array, ss_tensor=ss_tensor)
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



def get_sequences_and_edges_single(pdb_path, pdb_parser=None, return_coords=False):
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
    # print("seq_idx_list", seq_idx_list)
    bad_idx_list = []
    seq_x_list = []
    for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1] + 1):
        try:
            flag = 0
            Ca_array.append(chain[(" ", idx, " ")]["CA"].get_coord())
            flag = 1
            sequence += restype_3to1[chain[(" ", idx, " ")].get_resname()]
        except:
            if not flag:
                bad_idx_list.append(idx)
            if flag:
                sequence += "X"
                seq_x_list.append(idx)
            Ca_array = Ca_array[:len(sequence)]     

    Ca_array = np.array(Ca_array)
    resi_num = Ca_array.shape[0]

    if resi_num <= 1:
        return None, None
    G = np.dot(Ca_array, Ca_array.T)
    H = np.tile(np.diag(G), (resi_num, 1))
    dismap = (H + H.T - 2 * G) ** 0.5

    row, col = np.where(dismap <= 10)
    edge = [row, col]
    if len(sequence)>1024:
        print("sequence length", len(sequence))
        
    assert np.array(edge).max() < len(sequence), f"edge_index contains out-of-bounds indices {pdb_path} edge max {np.array(edge).max()} seq len {len(sequence)} CA array{Ca_array.shape}"
    
    # print("bad_idx_list", bad_idx_list)
    if return_coords:
        ss_tensor = get_ss_tensor(pdb_path, structure=struct)
        try:
            if len(sequence)!=ss_tensor.shape[0]:
                ss_tensor_final = torch.zeros(len(sequence))
                # print("len(sequence), ss_tensor, seq_x_list",len(sequence), ss_tensor.shape[0], seq_x_list)
                mask = torch.ones(len(ss_tensor_final), dtype=torch.bool)
                mask[seq_x_list] = 0
                ss_tensor_final[mask] = torch.tensor(ss_tensor)     
            assert len(sequence)==ss_tensor.shape[0], f"SS Tensor {ss_tensor.shape} and Sequence do not have same shape {len(sequence)}, bad idx {len(bad_idx_list)}"
        except:
            ss_tensor = None
            print("Skipping SS tensor")
        
        return sequence, edge, Ca_array, ss_tensor
    
    return sequence, edge


def process_pdb(pdb_paths, n_jobs=None, device="cpu", esm_path=None, batch_size=128, return_coords=False):
    
    parser = PDBParser()
    # seqs_and_edges = pmap_single(get_sequences_and_edges_single, pdb_paths, n_jobs=n_jobs, verbose=1, pdb_parser=parser)
    seq_edge_coord_ss = pmap_single(get_sequences_and_edges_single, pdb_paths, n_jobs=n_jobs, verbose=1, pdb_parser=parser, return_coords=return_coords)
    
    bad_paths  = []
    seqs_filt = []
    edges_filt = []
    coords_filt = []
    ss_tens_filt = []
    for i, (seq, edge, Ca_array, ss_tensor) in enumerate(seq_edge_coord_ss):
        if seq is None:
            bad_paths.append(pdb_paths[i])
            continue
        seqs_filt.append(seq)
        edges_filt.append(edge)
        coords_filt.append(Ca_array)
        ss_tens_filt.append(ss_tensor)
        # print("using coords, ss_tensor")
    print(f"dropped {len(seq_edge_coord_ss) - len(seqs_filt)} proteins with bad inputs")

    if esm_path is None:
        # esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_path)
    esm_model.eval()
    esm_model = esm_model.to(device)
    batch_converter = alphabet.get_batch_converter()

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
            results = esm_model(batch_tokens.to(device), repr_layers=[12], return_contacts=False)
            token_representations = (
                results["representations"][12].detach().cpu()
            )
        embeddings.extend(token_representations.numpy())
    # embeddings = torch.cat(embeddings)
    # embeddings = torch.cat(embeddings).numpy()

    graphs = []
    print(len(seqs_filt), len(edges_filt), len(embeddings))

    for i in tqdm(range(len(seqs_filt))):
        # print(i)
        if return_coords:
            # assert len(seqs_filt[i])==ss_tens_filt[i].shape[0], f"SS Tensor {ss_tens_filt[i].shape} and Sequence do not have same shape {len(seqs_filt[i])}"
            graphs.append(protein_graph(
                seqs_filt[i], edges_filt[i], embeddings[i][1: len(seqs_filt[i])+1], Ca_array=coords_filt[i], ss_tensor=ss_tens_filt[i]
            ))
        else:
            graphs.append(protein_graph(
                seqs_filt[i], edges_filt[i], embeddings[i][1: len(seqs_filt[i])+1]))
    return graphs, bad_paths

def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()




class GoTermDataset(Dataset):

    def __init__(self, annot_path, graph_list_file, pdb_id_list, task="bp", suffix=[]):
        # task can be among ['bp','mf','cc']
        # if single, pdb_id_list is a list, if multiple datasets, pdb_id list is a lits of pt files
        self.task = task
        if not isinstance(annot_path, list):
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

            print("graph_file:", graph_list_file)
            self.graph_list = torch.load(graph_list_file)

            # self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
            self.y_true = np.stack(
                [prot2annot[pdb_c][self.task] for pdb_c in self.pdb_id_list]
            )
            self.y_true = torch.tensor(self.y_true)
        else:
            #### all three arguments are lists of equal length
            self.graph_list = []
            self.pdb_id_list = []
            self.y_true = []
            for i, file in enumerate(graph_list_file):
                graph_list_i = torch.load(file) # load graphs
                self.graph_list+=graph_list_i
                pdb_id_list_i = torch.load(pdb_id_list[i])[f'{suffix[i]}_pdbch']# load labels
                self.pdb_id_list += pdb_id_list_i
                prot2annot, goterms, gonames, counts = load_GO_annot(annot_path[i])
                goterms = goterms[self.task]
                gonames = gonames[self.task]
                y_true = np.stack(
                [prot2annot[pdb_c][self.task] for pdb_c in pdb_id_list_i]
                )
                self.y_true.append(y_true)
            self.y_true = np.concatenate(self.y_true)
            self.y_true = torch.tensor(self.y_true)
            # self.graph_list = torch.tensor(self.graph_list)
    def __getitem__(self, idx):

        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)