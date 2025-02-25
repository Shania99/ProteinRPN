import os
import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBParser
import warnings
from tqdm import tqdm
import argparse

option = 'test'


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
id2res = {v: k for k, v in RES2ID.items()}


# for id in test_graphs[1].native_x:
    # print(restype_1to3[id2res[id.item()]], end=' ')
    
def download_pdb(pdb_id, download_path):
    pdbl = PDB.PDBList()
    # Download the file in PDB format, which saves as .ent
    file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=download_path, file_format='pdb')

    # Rename the file from .ent to .pdb
    base_name = os.path.basename(file_path)
    new_name = base_name.replace('ent', 'pdb')
    os.rename(file_path, os.path.join(download_path, new_name))
    
    
###### SEQUENCE ALIGNMENT
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def find_insertions_deletions(X, Y):
    # Perform a global alignment
    alignments = pairwise2.align.globalxx(X, Y)

    # Choose the first alignment (usually the highest scoring)
    alignment = alignments[0]
    aligned_X, aligned_Y = alignment[0], alignment[1]

    # Determine insertions and deletions
    insertions = []
    deletions = []
    y_index = 0

    for i in range(len(aligned_X)):
        if aligned_X[i] == '-':
            # Insertion in Y
            insertions.append(y_index)
            y_index += 1
        elif aligned_Y[i] == '-':
            # Deletion from X
            deletions.append(i)
        else:
            y_index += 1

    return np.array(insertions), np.array(deletions)

########### FINAL CODE
#1. load graphs and and pdbchlist
def func(option, af2=''):
    test_list = torch.load(f'/om2/user/shania/datasets/HEAL_data/processed/{af2}{option}_pdbch.pt')[f'{option}_pdbch']#### list of pdb ids
    test_graphs = torch.load(f'/om2/user/shania/datasets/HEAL_data/processed/{af2}{option}_heal_graphs.pt')

    warnings.simplefilter("ignore")


    new_pdb_graph_list = []
    for i in tqdm(range(len(test_list))):
        #2. download pdb file
        print(i, test_list[i])
        pdb_id_chain = test_list[i].split('-')
        pdb_id, chain_id = pdb_id_chain[0].lower(), pdb_id_chain[1]
        #3. load that particular graph
        pdb_graph = test_graphs[i]
        
        try:
            ######### HEAL SEQUENCE #######
            
            heal_sequence_x = ''
            for id in test_graphs[i].native_x:
                # print(id2res[id.item()], end='')
                heal_sequence_x+=id2res[id.item()]
            
            
            download_path = f'/om2/user/shania/datasets/HEAL_data/pdb_files/{af2}{option}'
            download_pdb(pdb_id, download_path)
            
            #4. Get ss tensor
            pdb_path = f'/om2/user/shania/datasets/HEAL_data/pdb_files/{af2}{option}/pdb{pdb_id}.pdb'
            pdb_parser = PDBParser()
            struct = pdb_parser.get_structure("x", pdb_path)
            model = struct[0]
            dssp = PDB.DSSP(model=model, in_file=pdb_path, dssp='mkdssp')
            # print(dssp)

            for key in dssp.keys():
                if key[0] == chain_id:
                    last_pos = key[1][1]


            # print("len in heal, len in dssp", len(pdb_graph.native_x), last_pos)
            ss_tensor = np.zeros(last_pos)
            for key in dssp.keys():
                if key[0] == chain_id:
                    pos = key[1][1]-1
                    ss = dssp[key][2]
                    if ss == 'H':  # Alpha helix
                        ss_tensor[pos] = 1
                    elif ss == 'E' or ss == 'B':  # Beta sheet or beta bridge
                        ss_tensor[pos] = 1
                    else:  # Coil, turn, bend, etc.
                        # ss_tensor.append(0)
                        pass
            ss_tensor = np.array(ss_tensor, dtype=np.int64)
            ######### DSSP SEQUENCE #######
            dssp_sequence_y = ''
            dssp_ss_tensor = []
            for key in dssp.keys():
                if key[0] == chain_id:
                    dssp_sequence_y+=dssp[key][1]
                    ss = dssp[key][2]
                    if ss == 'H':
                        dssp_ss_tensor.append(1)
                    elif ss == 'E' or ss == 'B':
                        dssp_ss_tensor.append(1)
                    else:
                        dssp_ss_tensor.append(0)
            dssp_ss_tensor = np.array(dssp_ss_tensor)

            
            ######## COORDINATE SEQUENCE ########
            chain = model[chain_id]
            Ca_array = []
            coord_sequence = ""
            seq_idx_list = list(chain.child_dict.keys())
            bad_idx_list = []
            seq_x_list = []

            for idx in range(seq_idx_list[0][1], max(330, seq_idx_list[-1][1]) + 1):
                # print(idx)
                try:
                    # print("in try")
                    flag = 0
                    residue = chain[(" ", idx, " ")]
                    # Try to get the CA coordinate, if not available, use any other available atom
                    atom_found = False
                    if "CA" in residue:
                        Ca_array.append(residue["CA"].get_coord())
                        atom_found = True
                    else:
                        # Fallback to any available atom
                        for atom in residue:
                            Ca_array.append(atom.get_coord())
                            # print("found atom")
                            atom_found = True
                            break

                    if atom_found:
                        flag = 1
                        coord_sequence += restype_3to1[residue.get_resname()]
                    else:
                        print("NO")
                except KeyError:
                    # print("in e1")
                    if not flag:
                        
                        bad_idx_list.append(idx)   

            Ca_array = np.array(Ca_array)

            # print(len(heal_sequence_x), len(dssp_sequence_y), len(dssp_ss_tensor), len(coord_sequence), len(Ca_array))

            X = heal_sequence_x  ### heal sequence/ seqres sequence
            Y = dssp_sequence_y ### pdb coord sequence
            #### final array I want is similar to X, at the deletions in X locations coords are zero, and from coord array take all except the insertions in Y indices

            insertions, deletions = find_insertions_deletions(X, Y)
            # print("Insertions at indices in Y:", insertions)
            # print("Deletions at indices in X:", deletions)

            insertions_dssp, deletions_dssp = find_insertions_deletions(heal_sequence_x, dssp_sequence_y)
            insertions_coord, deletions_coord = find_insertions_deletions(heal_sequence_x, coord_sequence)

            ss_tensor_final = np.zeros(len(heal_sequence_x))
            Ca_array_final = np.zeros((len(heal_sequence_x), 3))

            # if dssp_ss_tensor is not None:
            try:
                ss_tensor_final[~np.isin(np.arange(len(ss_tensor_final)), deletions_dssp)] = dssp_ss_tensor[~np.isin(np.arange(len(dssp_ss_tensor)), insertions_dssp)]
            except:
                print(f"skipping sequence, {pdb_id}")
                ss_tensor_final = np.zeros(len(heal_sequence_x))
            try:
                Ca_array_final[~np.isin(np.arange(len(Ca_array_final)), deletions_coord)] = Ca_array[~np.isin(np.arange(len(Ca_array)), insertions_coord)]
            except:
                print(f"skipping coordinates of sequence, {pdb_id}")
                Ca_array_final = np.zeros((len(heal_sequence_x), 3))
        # print(len(heal_sequence_x), len(ss_tensor_final), len(Ca_array_final))
        except:
            print(f"skipping full protein, {pdb_id}")
            ss_tensor_final = np.zeros(len(heal_sequence_x))
            Ca_array_final = torch.zeros((len(heal_sequence_x), 3))
            
        pdb_graph.ss = torch.tensor(ss_tensor_final)
        pdb_graph.coords = torch.tensor(Ca_array_final)
        new_pdb_graph_list.append(pdb_graph)

    torch.save(new_pdb_graph_list, f'/om2/user/shania/datasets/HEAL_data/processed/{af2}{option}_heal_graphs_pdb_ss_coords.pt')
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--option", type=str, default="test", choices=["test", "val", "train"], help=""
    )
    p.add_argument(
        "--af2", type=str, default="", choices=["AF2", ""], help=""
    )

    args = p.parse_args()
    func(args.option, args.af2)
    
    #### to call
    # cd /om2/user/shania/protein_func_new
    # python GO_heal_train.py --task bp --suffix grpn_heal_esm2_bp --device cuda --esmembed True --pooling MTP --contrast False --AF2model False --batch_size 48

