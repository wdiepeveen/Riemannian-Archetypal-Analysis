import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class MetricMST:
    def __init__(self, pairwise_distances, labels=None, use_graphviz=True):
        self.D = np.asarray(pairwise_distances, dtype=float)
        self.labels = np.asarray(labels) if labels is not None else None
        self.use_graphviz = use_graphviz

        if self.D.ndim != 2 or self.D.shape[0] != self.D.shape[1]:
            raise ValueError("pairwise_distances must be an N x N matrix")

        self.n = self.D.shape[0]
        self.T = None
        self.root = None

        if self.labels is not None and len(self.labels) != self.n:
            raise ValueError("labels must have the same length as pairwise_distances")

    def build_mst(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=float(self.D[i, j]))

        self.T = nx.minimum_spanning_tree(G, weight="weight")
        return self.T

    def choose_root(self, use_edge_weights=True):
        if self.T is None:
            self.build_mst()

        centers = nx.center(self.T, weight="weight" if use_edge_weights else None)
        self.root = centers[0]
        return self.root

    def total_cost(self):
        if self.T is None:
            self.build_mst()
        return self.T.size(weight="weight")

    def root_tree(self, root=None):
        if self.T is None:
            self.build_mst()
        if root is None:
            root = self.root if self.root is not None else self.choose_root()
        return nx.bfs_tree(self.T, source=root)

    def metric_tree_positions(self, root=None, horizontal_scale=1.0, vertical_scale=1.0):
        """
        Tree layout where y-coordinate reflects cumulative weighted distance from root.
        x-coordinate is assigned recursively so leaves are spread left-to-right.
        """
        if self.T is None:
            self.build_mst()
        if root is None:
            root = self.root if self.root is not None else self.choose_root()

        T_dir = self.root_tree(root=root)

        children = {u: list(T_dir.successors(u)) for u in T_dir.nodes()}

        x_pos = {}
        leaf_counter = [0]

        def assign_x(u):
            ch = children[u]
            if len(ch) == 0:
                x_pos[u] = leaf_counter[0]
                leaf_counter[0] += 1
            else:
                for v in ch:
                    assign_x(v)
                x_pos[u] = np.mean([x_pos[v] for v in ch])

        assign_x(root)

        dist_from_root = nx.single_source_dijkstra_path_length(self.T, root, weight="weight")

        pos = {}
        for u in self.T.nodes():
            pos[u] = (
                horizontal_scale * x_pos[u],
                -vertical_scale * dist_from_root[u]
            )

        return pos

    def kamada_kawai_positions(self, root=None):
        """
        Layout based on weighted graph distances on the MST.
        """
        if self.T is None:
            self.build_mst()
        return nx.kamada_kawai_layout(self.T, weight="weight")

    def tree_positions(self, root=None, layout="metric_tree"):
        if self.T is None:
            self.build_mst()
        if root is None:
            root = self.root if self.root is not None else self.choose_root()

        if layout == "metric_tree":
            return self.metric_tree_positions(root=root)

        elif layout == "kamada_kawai":
            return self.kamada_kawai_positions(root=root)

        elif layout == "graphviz":
            T_dir = self.root_tree(root=root)
            try:
                from networkx.drawing.nx_pydot import graphviz_layout
                return graphviz_layout(T_dir, prog="dot")
            except Exception:
                return self.metric_tree_positions(root=root)

        else:
            raise ValueError("layout must be one of: 'metric_tree', 'kamada_kawai', 'graphviz'")

    def plot(self, root=None, layout="metric_tree", annotate=False,
             show_edge_weights=False, figsize=(12, 8), node_size=120):
        if self.T is None:
            self.build_mst()
        if root is None:
            root = self.root if self.root is not None else self.choose_root()

        pos = self.tree_positions(root=root, layout=layout)

        if self.labels is not None:
            unique_labels = list(dict.fromkeys(self.labels))
            cmap = plt.get_cmap("tab20")
            color_map = {lab: cmap(i % 20) for i, lab in enumerate(unique_labels)}
            node_colors = [color_map[self.labels[node]] for node in self.T.nodes()]
        else:
            node_colors = "lightblue"

        plt.figure(figsize=figsize)

        nx.draw_networkx_edges(
            self.T,
            pos,
            edge_color="gray",
            width=1.5,
            alpha=0.9
        )

        nx.draw_networkx_nodes(
            self.T,
            pos,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="black",
            linewidths=0.5
        )

        if annotate:
            node_text = {i: str(i) for i in self.T.nodes()}
            nx.draw_networkx_labels(self.T, pos, labels=node_text, font_size=8)

        if show_edge_weights:
            edge_labels = {
                (u, v): f"{d['weight']:.2f}"
                for u, v, d in self.T.edges(data=True)
            }
            nx.draw_networkx_edge_labels(self.T, pos, edge_labels=edge_labels, font_size=7)

        plt.title(f"MST rooted at node {root} ({layout})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def fit(self, use_edge_weight_root=True):
        self.build_mst()
        self.choose_root(use_edge_weights=use_edge_weight_root)
        return self