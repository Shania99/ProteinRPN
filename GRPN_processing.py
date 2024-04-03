"""Processing output candidates of GRPN model and performing node clustering on candidate functional sites
"""

import numpy as np
import torch
from itertools import compress

# node_scores_list, node_list, func_probability


def get_proposed_candidates(
    node_scores_list,
    node_list,
    func_probability,
    func_threshold=0.0,
    node_score_threshold=0.0,
    check_overlap=False,
):
    """get proposed candidiates for a single protein"""
    functional_nodes = []

    # print(node_list)
    # print("\n\n", len(node_list))
    # print("\n\n", type(node_list))
    # print("\n\n", node_list[0])
    # print("\n\n", type(node_list[0]))

    node_score_list = [l.detach().numpy() for l in node_scores_list]
    node_list = [l.detach().numpy() for l in node_list]
    # print("printing values", node_score_list)
    numpy_score_list = np.array(node_score_list, dtype=object)
    numpy_node_list = np.array(node_list, dtype=object)
    print(
        "GRPN_processing: predicted functional probability of anchor nodes: max min",
        max(func_probability),
        min(func_probability),
    )
    print("number of nodes in protein graph", len(node_scores_list))
    selected = (
        func_probability > func_threshold
    ).flatten()  ## these are the nodes whose corresponding anchors have a high probab of being functional
    # print("in grpn processing selected nodes based on probab", selected)
    score_list_filtered = numpy_score_list[selected]  # list of arrays
    node_list_filtered = numpy_node_list[
        selected
    ]  ## selects score_lists and node_lists for anchors for which
    # there is a high probab of having a functional region
    anchor_chosen_list = []
    # print("printing values", score_list_filtered, node_list_filtered)
    # in these selected anchors we have o find the nodes for which the score is high (we have high confidence in them being functional)
    for i, (score_lis, node_lis) in enumerate(
        zip(score_list_filtered, node_list_filtered)
    ):
        if len(node_lis) <= 1:
            continue
        # print("each score list", score_list_filtered[i])
        selected_nodes = score_lis > node_score_threshold
        # print(
        #     "node_list_filtered[i]",
        #     node_list_filtered[i],
        # )
        # print(
        #     "checking length of node lis and score lis", len(node_lis), len(score_lis)
        # )
        # print("node_list_filtered_i", node_list_filtered_i)
        # print("selected nodes", selected_nodes.squeeze())
        anchor_chosen_list.append(node_lis[selected_nodes.squeeze()])
        # print("anchor chosen list", anchor_chosen_list)
        # from each anchor, these are the selected nodes

    if check_overlap:
        for i in range(len(anchor_chosen_list)):
            for j in range(i + 1, len(anchor_chosen_list)):
                metric = len(
                    set(anchor_chosen_list[i]).intersection(anchor_chosen_list[j])
                ) / (len(anchor_chosen_list[i]) + len(anchor_chosen_list[j]))
                if metric > 0.7:
                    functional_nodes.extend(
                        list(set(anchor_chosen_list[i]).union(anchor_chosen_list[j]))
                    )
    else:
        for lis in anchor_chosen_list:
            # print("lis", lis)
            functional_nodes.extend(list(lis))

    return functional_nodes  # returns node_numbers of functional sites


def get_proposed_candidates_batches(
    node_scores_list,
    node_list,
    func_probability,
    batch,
    func_threshold=0.0,
    node_score_threshold=0.0,
    check_overlap=False,
):
    one_hot_node_type_vec = torch.zeros(len(node_list))
    # print("one_hot_node_type_vec", one_hot_node_type_vec.shape)

    # node_scores_list, node_list, func_probability are lists of tensors for each node in a graph, for all graphs in the batch
    # batch is a tensor indicating the graph membership of each node
    # we want to output a list of 0 and 1, 1 if final cadidate, 0 if not (as continuous tensor for all graphs)

    for batch_num in batch.unique():
        # print("THIS IS BACTH", batch_num)

        # Select nodes belonging to the current graph
        nodes_scores = list(
            compress(node_scores_list, batch == batch_num)
        )  # list of np.arrays
        # one_hot_node_type_vec_batch = torch.zeros(len(nodes_scores))
        # print("proposed number of nodes", len(nodes_scores))
        # nodes_scores = node_scores_list[batch == batch_num]
        nodes = list(compress(node_list, batch == batch_num))
        # nodes = node_list[batch == batch_num]
        # func_prob = np.array(compress(func_probability, batch == batch_num))
        func_prob = func_probability[batch == batch_num]

        func_nodes = get_proposed_candidates(
            nodes_scores,
            nodes,
            func_prob,
            func_threshold=func_threshold,
            node_score_threshold=node_score_threshold,
            check_overlap=check_overlap,
        )

        func_nodes = list(set(func_nodes))

        print("func_nodes", func_nodes)
        one_hot_node_type_vec[func_nodes] = 1
        # print("one_hot_node_type_vec_batch", one_hot_node_type_vec_batch)
        # batch_functional_nodes.append(func_nodes)
    # print("inside get proposed cand", type(one_hot_node_type_vec))
    return one_hot_node_type_vec
