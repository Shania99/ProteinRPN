"""
This module contains functions to preprocess the PDBSite dataset and 
create a graph representation of the protein structure in pyg
"""

import re
import warnings
import pandas as pd
import numpy as np
from Bio import PDB
import esm
import torch

# from Bio.PDB import PDBList
import Bio.PDB.PDBParser as PDBParser
import Bio.PDB.PDBList as PDBList
import networkx as nx
import torch_geometric
import torch_geometric.utils


warnings.simplefilter("ignore")


AA_NAME_MAP = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "TER": "*",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "XAA": "X",
}


def process(string):
    """Processes lines from the PDBSite file and removes the header

    Args:
        string (str): line from the PDBSite file
    Returns:
        List[str]: Splits entries using space and returns all but the header (first entry)
    """
    entries = string.split(" ")
    return entries[1:]


def process_data(filename="PDBSite_results.txt", encoding="utf-8"):
    """Extracts useful information from the PDBSite file
    Returns a dictionary with columns of dataframe as keys and lists of values as values
    each row of the resulting dataframe corresponds to a single annotation

    Args:
        filename (str, optional): Path of PDBSite entries file. Defaults to "PDBSite_results.txt".

    Returns:
        dict: Dictionary containing the extracted information for all pdb_site entries consisting of:
            - pdb_id: pdb_id of the protein
            - pdbsite_id: id of the annotation; a single protein can have multiple annotations
            - y: functional (1) or non-functional (0);
                since we extract functional residues from pdb site entries, label is 1
            - len_site: length of functional site
            - site_chains: chains to which functional residues belong
            - site_pos: positions of functional residues
            - site_res: residues of functional sites
            - env_pos: positions of residues in the environment
            - env_res: residues in the environment
            - node_labels: label for each functional node to be used while constructing graph,
                            format: "chain_position_residue"
    """
    f = open(filename, "r+", encoding=encoding)
    string = f.read()
    entries = string.split(
        "\nEND\n"
    )  # extracting each entry in the pdbsite file by splitting at "END" (single pdbsite id)
    data_dict = {
        "pdb_id": [],
        "pdbsite_id": [],
        "y": [],
        "len_site": [],
        "site_chains": [],
        "site_pos": [],
        "site_res": [],
        # "env_pos": [],
        # "env_res": [],
        "node_labels": [],
    }
    for entry in entries:
        pdb_id = re.findall(r"^PDBID.+$", entry, re.MULTILINE)
        pdbsite_id = re.findall(r"^ID.+$", entry, re.MULTILINE)
        positions = re.findall(r"^POS.+$", entry, re.MULTILINE)
        residues = re.findall(r"^RESNAME.+$", entry, re.MULTILINE)
        num_residues = re.findall(r"^NUMBER_OF_AA.+$", entry, re.MULTILINE)
        site_chains = re.findall(r"^SITE_CHAINS.+$", entry, re.MULTILINE)
        header = re.findall(r"^HEADER.+$", entry, re.MULTILINE)
        if all([pdb_id, pdbsite_id, positions, residues]):
            data_dict["pdb_id"].append(process(pdb_id[0])[0])  # pdb id of one protein
            data_dict["pdbsite_id"].append(
                process(pdbsite_id[0])[0]
            )  # pdbsite id of one annotation
            site_pos = process(
                positions[0]
            )  # first element of positions returns site positions of in a string (second is env pos), process converts to list with numbers
            data_dict["site_pos"].append(
                site_pos
            )  # adds that list to site_pos column of dict
            site_res = list(process(residues[0])[0])
            data_dict["site_res"].append(site_res)
            # data_dict["env_pos"].append(process(positions[1]))
            # data_dict["env_res"].append(process(residues[1])[0])
            data_dict["len_site"].append(process(num_residues[0])[0])
            site_chains = list(process(site_chains[0])[0])
            data_dict["site_chains"].append(site_chains)
            data_dict["y"].append("_".join(process(header[0])))
            data_dict["node_labels"].append(
                [f"{c}_{n}_{s}" for c, n, s in zip(site_chains, site_pos, site_res)]
            )
    f.close()
    return data_dict


def get_aa_embeddings(model, alphabet):
    """_summary_

    Args:
        model (_type_): _description_
        alphabet (_type_): _description_

    Returns:
        _type_: _description_
    """
    embedding_matrix = model.embed_tokens.weight.data.numpy()
    aa_tokens = alphabet.standard_toks[:-2]
    aa_to_index = alphabet.to_dict()
    aa_to_embeddings = {aa: embedding_matrix[aa_to_index[aa]] for aa in aa_tokens}
    return aa_to_embeddings


def compute_contacts(coords, node_labels, threshold=6.9, binarize=True):
    """Compute the pairwise contacts.

    Here we define a contact as the C-alpha atom
    of 2 amino acids being within 9 Angstrom of each other.

    Args:
        coords (_type_): array of shape (num_residues, 3)
        node_labels (_type_): _description_
        threshold (float, optional): distance threshold to consider a contact. Defaults to 6.9.
        binarize (bool, optional): _description_. Defaults to True.

    Returns:
        contacts (pd.DataFrame): Dataframe of shape (num_residues, num_residues)
                                containing contacts or distances between residues
    """

    num_residues = coords.shape[0]
    contacts = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        for j in range(i + 1, num_residues):  # Skip self and already computed
            distance = np.linalg.norm(coords[i] - coords[j])
            if binarize:
                if distance <= threshold:
                    contacts[i, j] = 1
                    contacts[j, i] = 1  # The matrix is symmetric
            else:
                contacts[i, j] = distance
                contacts[j, i] = distance

    return pd.DataFrame(contacts, index=node_labels, columns=node_labels)


def load_pdb(pdb_id, esm_embeddings=None):
    """For a given protein pdb_id, load the pdb file and extract the sequence and coordinates.

    Args:
        pdb_id (str): pdb_id of the protein
        esm_embeddings (dict, optional): dictionary containing embeddings for the 20 residues. Defaults to None.

    Returns:
        node_labels (List[str]): label for each node to be considered while constructing graph
                                format: "chain_position_residue"
        embed_dict (dict): dictionary containing embeddings for each residue
        coords (np.array): array of shape (num_residues, 3) containing the centroid of each residue
    """
    esm_model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    pdbl = PDBList()
    pdbp = PDBParser()
    pdbl.retrieve_pdb_file(pdb_id.upper(), file_format="pdb", pdir="./pdb_files")
    struct = pdbp.get_structure("struct", f"./pdb_files/pdb{pdb_id.lower()}.ent")
    model = struct[0]

    # sequence = []
    coords = []
    # chain_ids = []
    embed_dict = {}
    node_labels = []
    for chain in model:
        # if chain.id != chain_id:
        #   continue
        residue_number = []
        current_sequence = []
        embed_values = []
        for residue in chain:
            if PDB.is_aa(residue, standard=True):
                atom_coords = [atom.get_coord() for atom in residue]
                centroid = np.mean(atom_coords, axis=0)
                coords.append(centroid)
                resname = AA_NAME_MAP[residue.resname]
                # sequence.append(resname)
                current_sequence.append(resname)
                # embed_values.append(esm_embeddings[resname])
                # chain_ids.append(chain.id)
                # info_dict = {k:(res, )} format (chain_id: (sequence, positions {get_id.join}))
                # print("residue id", res)
                residue_number.append(
                    "".join([str(x) for x in residue.get_id()]).strip()
                )
        # Prepare data (two protein sequences)
        data = [("protein1", "".join(current_sequence))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Extract per-residue embeddings (on CPU)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33])

        embed_values = results["representations"][33][0]
        embed_values = embed_values[1:]
        print(embed_values.shape)
        print(len(current_sequence))
        # print(chain.id, residue_number)
        # assert len(residue_number) == len(current_sequence) == len(embed_values)
        new_node_labels = [
            f"{chain.id}_{n}_{s}" for n, s in zip(residue_number, current_sequence)
        ]
        node_labels += new_node_labels
        # sequence.append('|')
        new_values = {k: v for k, v in zip(new_node_labels, embed_values)}
        embed_dict.update(new_values)
        assert len(residue_number) == len(new_values)
        assert len(node_labels) == len(embed_dict)
    return node_labels, embed_dict, np.array(coords)


def create_protein_graph(PDB_ID, df, esm_embeddings=None):
    """For a given protein pdb_id, extract all functional site annotations and create a graph where the contact map is
    the adjancency matrix, nodes are labelled according to "chain_position_residue" and functionality of nodes is depicted.

    Args:
        PDB_ID (str): PDB ID of the protein which is to be converted to a graph
        df (pd.DataFrame): Dataframe containing the PDBSite annotations

    Returns:
        protein_graph (nx.Graph): graph with attributes such as functionality, pdb_site_id, length of functional site,
                                edges are the contacts between residues
    """

    node_labels, embed_dict, coords = load_pdb(PDB_ID, esm_embeddings=esm_embeddings)
    contacts = compute_contacts(coords, node_labels)
    # plt.imshow(contacts)

    # print(node_labels, info_dict, coords)
    assert contacts.shape[0] == len(node_labels)
    protein_graph = nx.from_pandas_adjacency(contacts)

    # protein_graph.edges
    nx.set_node_attributes(
        protein_graph, name="y", values=0
    )  # 0 for non-functional, 1 for functional
    nx.set_node_attributes(protein_graph, name="x", values=embed_dict)
    nx.set_node_attributes(protein_graph, name="subtype", values="None")
    nx.set_node_attributes(protein_graph, name="pdbsiteid", values="None")
    nx.set_node_attributes(protein_graph, name="lensite", values="None")
    # nx.set_node_attributes(protein_graph, name="coords", values="None")
    for i in range(len(coords)):
        protein_graph.nodes[node_labels[i]]["coords"] = coords[i]

    ##### groupby protein structure, set all nodes to none
    ##### select functional sites, put label function, subtype: header
    # protein_graph.nodes('A_1_A')
    groups_df = df.groupby("pdb_id")
    functional_nodes = groups_df.get_group(PDB_ID).node_labels.values
    headers = groups_df.get_group(PDB_ID)["y"].values
    pdbsite_ids = groups_df.get_group(PDB_ID)["pdbsite_id"].values
    lensite = groups_df.get_group(PDB_ID)["len_site"].values

    func_node_attr = {}
    for header, pdbsite_id, lensite_num, nodes in zip(
        headers, pdbsite_ids, lensite, functional_nodes
    ):
        for node in nodes:
            if node not in func_node_attr:
                func_node_attr[node] = {
                    "y": 1,
                    "subtype": [header],
                    "pdbsiteid": [pdbsite_id],
                    "lensite": [lensite_num],
                    # "coords": coords[node_labels.index(node)],
                }
            else:
                print("common node found")
                func_node_attr[node]["subtype"].append(header)
                func_node_attr[node]["pdbsiteid"].append(pdbsite_id)
                func_node_attr[node]["lensite"].append(lensite_num)
                print(func_node_attr[node])

    nx.set_node_attributes(protein_graph, func_node_attr)
    return protein_graph


##### GROUND TRUTH

# subgraph = nx.ego_graph(g, 'A_1_A',radius=2)
# # nx.draw(subgraph)
# nx.get_node_attributes(subgraph, 'type')
# # num_nodes =
# # G2=nx.subgraph(G,[x for x in G.nodes() if pdb_site_id in G.])
# len(subgraph)


#### given a protein get all functional nodes, find their k-hop subgraph, in that extract all func nodes
## This is all for labeling a single protein, this needs to be done for all proteins


def ego_graph_label(protein_graph: nx.Graph):
    """Adds functionality labels for the ego graph anchor centered at each node in the protein graph
    If >70% of a functional site lies inside an anchor, it is labelled as 1, else 0
    sets ego label as node attributes in protein graph and returns ground truth annotations linked to each functional node

    Args:
        protein_graph (nx.Graph): takes in protein graph processed by create_protein_graph()

    Returns:
        label_graphs (dict): for each (ground truth = 1) functional node, what are the ground truth graph pdbsites
    """
    ego_label = {node: 0 for node, att in protein_graph.nodes(data=True)}
    label_graphs = (
        {}
    )  # for each (ground truth = 1) functional node what are the ground truth graph pdbsites
    functional_nodes = [
        node for node, att in protein_graph.nodes(data=True) if att["y"] == 1
    ]

    for functional_node in functional_nodes:
        subgraph = nx.ego_graph(protein_graph, functional_node, radius=2)
        # nx.draw(subgraph)
        pdbsite_dict = nx.get_node_attributes(subgraph, "pdbsiteid")
        # print(pdbsite_dict)
        lensite_dict = nx.get_node_attributes(subgraph, "lensite")
        list_pdbsites = pdbsite_dict[functional_node]
        list_lensite = lensite_dict[functional_node]
        # print(list_lensite)
        # print(list_pdbsites)
        for i, pdbsite_id in enumerate(list_pdbsites):
            # print(i, pdbsite_id)
            len_site = int(list_lensite[i])
            # print(len_site)
            func_subgraph_nodes = [
                k for (k, v) in pdbsite_dict.items() if pdbsite_id in v
            ]
            overlap_ratio = len(func_subgraph_nodes) / len_site
            if overlap_ratio >= 0.7:
                ego_label[functional_node] = 1
            if functional_node in label_graphs:
                label_graphs[functional_node].append(pdbsite_id)
            else:
                label_graphs[functional_node] = [pdbsite_id]
    nx.set_node_attributes(protein_graph, name="ego_label", values=ego_label)
    return label_graphs


def convert_pyg(protein_graph: nx.Graph):
    """Converts a protein graph with string node labels to one with integers and then converts to a pyg Data object

    Args:
        protein_graph (nx.Graph): protein graph with string node labels, and all attributes set

    Returns:
        torch_geometric.data.Data: converted protein graph in compatible pyg format
    """
    pyg_graph = nx.convert_node_labels_to_integers(
        protein_graph, first_label=0, ordering="default", label_attribute=None
    )
    pyg_graph = torch_geometric.utils.from_networkx(pyg_graph)
    return pyg_graph


### EXAMPLE USAGE
# PDB_ID = "1A47"
# data_dict = process_data(filename="PDBSite_results.txt")
# df = pd.DataFrame(data_dict)
# groups_df = df.groupby("pdb_id")
# functional_nodes = groups_df.get_group(PDB_ID).node_labels.values
# headers = groups_df.get_group(PDB_ID)["y"].values
# pdbsite_ids = groups_df.get_group(PDB_ID)["pdbsite_id"].values
# lensite = groups_df.get_group(PDB_ID)["len_site"].values
# g = create_protein_graph(PDB_ID="1A47", df=df)
# print(g.nodes(data=True))
# print(func_ground_truth(g))

# """
# Convert to pyg step-1
# add esm embeddings as node features to networkx
# convert each graph to pyg data object keeping node_features = esm_embeddings, labels=functionality ### one set
# What do I need? I need a functionality vector for each subgraph
# Each graph has two labels: subgraph label (i.e., ego label) and node label (y)
# for node classif, we use ego_label, for graph pruning we use node_label

# to get loss for pruning, we need to find gt_labels. That can be done by selecting nodes from data.y using
# node_list and then computing mse between that and pred_labels.

# """

# GOAL-1:
# Combine pyg_data_handling.py and preprocess.py into a single file and make sure ego labels and y are both given out by convert pyg
# GOAL-2:
# Loop through 5 pdb ids and create data objects for each pdb id, define data loader
# GOAL-3:
# Go through training loop
