import plotly.graph_objs as go
import numpy as np


def visualize_labellings(protein_graph, functional_nodes):

    # print(protein_graph.coords.size())
    coords = protein_graph.coords.numpy()
    # print(functional_nodes)
    functional_coords = coords[functional_nodes]
    ground_truth_coords = coords[protein_graph.y]
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=12,
            color=z,  # set color to an array/list of desired values
            colorscale="Viridis",
            opacity=0.5,
        ),
    )

    func_x = functional_coords[:, 0]
    func_y = functional_coords[:, 1]
    func_z = functional_coords[:, 2]

    gt_x = ground_truth_coords[:, 0]
    gt_y = ground_truth_coords[:, 1]
    gt_z = ground_truth_coords[:, 2]

    trace2 = go.Scatter3d(
        x=func_x,
        y=func_y,
        z=func_z,
        mode="markers",
        marker=dict(
            size=8,
            color=x,  # set color to an array/list of desired values
            colorscale="Viridis",
        ),
    )

    trace3 = go.Scatter3d(
        x=gt_x,
        y=gt_y,
        z=gt_z,
        mode="markers",
        marker=dict(
            size=14,
            color=y,  # set color to an array/list of desired values
            colorscale="Oranges",
            opacity=0.3,
        ),
    )

    layout = go.Layout(title="3D Scatter plot")
    fig = go.Figure(data=[trace, trace2, trace3], layout=layout)
    fig.show()
