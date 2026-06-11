"""Visualization components for ChainScope Streamlit frontend."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go


def plot_radar(components: dict) -> matplotlib.figure.Figure:
    """Plot 4-component anomaly radar chart.

    Args:
        components: dict with keys s_attr, s_struct, s_flow, s_temp (0-1 floats)
    """
    cats = ["Attribute", "Structural", "Flow", "Temporal"]
    vals = [
        components.get("s_attr", 0),
        components.get("s_struct", 0),
        components.get("s_flow", 0),
        components.get("s_temp", 0),
    ]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    vals_closed = vals + vals[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, vals_closed, alpha=0.25, color="#e74c3c")
    ax.plot(angles_closed, vals_closed, color="#e74c3c", linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Anomaly Score Components", fontsize=12, pad=15)

    # Add threshold circle at 0.6
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta, [0.6] * 100, color="orange", linewidth=1, linestyle="--", alpha=0.7)
    return fig


def plot_radar_plotly(components: dict) -> go.Figure:
    """Plotly version of radar chart for Streamlit."""
    cats = ["Attribute", "Structural", "Flow", "Temporal"]
    vals = [
        components.get("s_attr", 0),
        components.get("s_struct", 0),
        components.get("s_flow", 0),
        components.get("s_temp", 0),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1],
        theta=cats + cats[:1],
        fill="toself",
        fillcolor="rgba(231,76,60,0.25)",
        line=dict(color="#e74c3c", width=2),
        name="Score",
    ))
    # Threshold line at 0.6
    fig.add_trace(go.Scatterpolar(
        r=[0.6] * 5,
        theta=cats + cats[:1],
        line=dict(color="orange", width=1, dash="dash"),
        name="Threshold (0.6)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        showlegend=True,
        title="Anomaly Score Components",
        height=400,
    )
    return fig


def build_network_html(snapshot, scores=None) -> str:
    """Build pyvis network HTML from graph snapshot.

    Args:
        snapshot: PyG Data with node_id, edge_index, x
        scores: optional numpy array of anomaly scores per node
    """
    net = Network(height="500px", directed=True, bgcolor="#ffffff",
                  font_color="#333333", select_menu=True)
    net.barnes_hut()

    node_ids = snapshot.node_id if hasattr(snapshot, "node_id") else []
    n_nodes = len(node_ids)
    if scores is None:
        scores = np.zeros(n_nodes)

    for i in range(n_nodes):
        score = float(scores[i]) if i < len(scores) else 0
        if score >= 0.6:
            color = "#e74c3c"
            size = 25
        elif score >= 0.3:
            color = "#f39c12"
            size = 18
        else:
            color = "#2ecc71"
            size = 12
        label = node_ids[i][:10] + "..." if len(node_ids[i]) > 10 else node_ids[i]
        title = f"{node_ids[i]}\nScore: {score:.4f}"
        net.add_node(i, label=label, color=color, size=size, title=title)

    ei = snapshot.edge_index
    for j in range(ei.shape[1]):
        src, dst = int(ei[0, j]), int(ei[1, j])
        if src < n_nodes and dst < n_nodes:
            net.add_edge(src, dst, color="#c0c0c0", width=0.5)

    html = net.generate_html()
    return html


def plot_grain_balls_plotly(snapshot, scores, grain_ball_ids=None) -> go.Figure:
    """Plot node grain ball membership with anomaly score as size.

    Args:
        snapshot: PyG Data with x, node_id
        scores: numpy array of anomaly scores
        grain_ball_ids: optional numpy array of grain ball assignments
    """
    import torch
    n_nodes = scores.shape[0] if hasattr(scores, 'shape') else len(scores)
    if n_nodes == 0:
        fig = go.Figure()
        fig.update_layout(title="No nodes to visualize")
        return fig

    x_feat = snapshot.x[:, 0].numpy() if hasattr(snapshot.x, 'numpy') else snapshot.x[:, 0]
    y_feat = snapshot.x[:, 1].numpy() if hasattr(snapshot.x, 'numpy') else snapshot.x[:, 1]

    if grain_ball_ids is None:
        grain_ball_ids = np.zeros(n_nodes, dtype=int)

    node_labels = []
    if hasattr(snapshot, "node_id"):
        node_labels = [nid[:8] for nid in snapshot.node_id[:n_nodes]]

    fig = px.scatter(
        x=x_feat[:n_nodes],
        y=y_feat[:n_nodes],
        color=grain_ball_ids[:n_nodes],
        size=np.clip(np.array(scores[:n_nodes]) * 15, 3, 25),
        color_continuous_scale="Viridis",
        title="Grain Ball Membership (size = anomaly score)",
        labels={"x": "Feature 1", "y": "Feature 2", "color": "Grain Ball"},
    )
    fig.update_layout(height=450)
    return fig
