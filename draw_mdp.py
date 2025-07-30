import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
import ast
import math
import itertools as it
import pydot
from networkx.drawing.nx_agraph import graphviz_layout
import my_networkx as my_nx

def draw_mdp(
    transitions:np.typing.NDArray,
    rewards:np.typing.NDArray,
    node_colors:np.typing.NDArray=None,
    pos=None,
    draw_labels=False
):
    """
    This visualization has two types of nodes, states, and available actions per state.
    """
    n_states, n_actions, _ = transitions.shape

    G = nx.DiGraph()

    state_labels = [rf"$s_{{{s + 1}}}$" for s in range(n_states)]
    state_action_labels = [rf"$s_{{{s + 1}}}, a_{{{a + 1}}}$" for s in range(n_states) for a in range(n_actions)]

    # Add state nodes
    for s in range(n_states):
        G.add_node(
            state_labels[s],
            label=fr"$s_{s + 1}$",
            type="state"
        )
    
    # Add action nodes
    for s in range(n_states):
        for a in range(n_actions):
            G.add_node(
                state_action_labels[s * n_actions + a],
                label=rf"$s_{s + 1}, a_{a + 1}$",
                type="action",
                color=plt.cm.Set2(a)
            )
    
    # Edge color map - use CenteredNorm to center at reward=0
    halfrange = np.max(np.abs(rewards))
    edge_color_norm = CenteredNorm(vcenter=0.0, halfrange=halfrange)

    # Add edges from state to action nodes
    for s in range(n_states):
        for a in range(n_actions):
            G.add_edge(
                state_labels[s],
                state_action_labels[s * n_actions + a],
                key=a,
                weight=np.sum(transitions[s, a, :]),
                color=plt.cm.coolwarm(edge_color_norm(rewards[s, a]))
            )

    # Add edges from action to state nodes
    for s in range(n_states):
        for a in range(n_actions):
            for s_prime in range(n_states):
                if transitions[s, a, s_prime] > 0:
                    G.add_edge(
                        state_action_labels[s * n_actions + a],
                        state_labels[s_prime],
                        key=a,
                        weight=transitions[s, a, s_prime],
                        color=plt.cm.coolwarm(edge_color_norm(rewards[s, a]))
                    )
    
    # compute positions if not given
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = pos(G)

    # separate node lists
    state_nodes  = [n for n, d in G.nodes(data=True) if d["type"] == "state"]
    action_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "action"]

    # draw states as circles with white fill and colored borders
    nx.draw_networkx_nodes(G, pos,
                           nodelist=state_nodes,
                           node_shape="o",
                           node_color="white",
                           edgecolors=node_colors or "lightblue",
                           linewidths=2,
                           node_size=200)
    # draw actions as squares with white fill and colored borders
    action_colors = [d["color"] for n, d in G.nodes(data=True) if d["type"] == "action"]
    nx.draw_networkx_nodes(G, pos,
                           nodelist=action_nodes,
                           node_shape="d",
                           node_color="white",
                           edgecolors=action_colors,
                           linewidths=2,
                           node_size=200)

    # identify straight and curved edges
    straight_edges = [(u, v) for (u, v) in G.edges() if (v, u) not in G.edges()]
    curved_edges = list(set(G.edges()) - set(straight_edges))
    
    # Extract colors for each edge type, ensuring proper matching
    straight_colors = []
    for u, v in straight_edges:
        edge_data = G.get_edge_data(u, v)
        color = edge_data['color']
        straight_colors.append(color)
    
    curved_colors = []
    for u, v in curved_edges:
        edge_data = G.get_edge_data(u, v)
        color = edge_data['color']
        curved_colors.append(color)

    # draw straight edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=straight_edges,
                           arrows=True,
                           edge_color=straight_colors,
                           width=2)
    # draw curved edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=curved_edges,
                           arrows=True,
                           connectionstyle=f"arc3,rad={math.pi / 16}",
                           edge_color=curved_colors,
                           width=2)
    # labels?
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add colorbar for reward scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=edge_color_norm)
    sm.set_array(rewards.flatten())
    plt.colorbar(sm, ax=plt.gca(), shrink=0.8, aspect=20)
    plt.axis("off")
    plt.show()