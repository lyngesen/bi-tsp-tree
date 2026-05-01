# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.23.4",
#     "networkx==3.4.2",
#     "plotly==6.7.0",
#     "pandas==2.3.3",
# ]
# ///
import marimo

__generated_with = "0.23.4"
app = marimo.App()


# ============================================================
# Inline: utils.py
# ============================================================

with app.setup:
    import marimo as mo
    import random
    import math
    import time
    import networkx as nx
    import plotly.graph_objects as go
    import pandas as pd
    from itertools import combinations
    from functools import reduce
    from collections import deque
    from plotly.subplots import make_subplots

    # ---- utils ----

    def generate_weights(G: nx.Graph, type: str = "nondecreasing", M: int = 100):
        if type not in {"random", "inverse", "nondecreasing", "nonincreasing"}:
            raise ValueError(f"Unknown weight type: {type}")
        root = 0
        for u, v in G.edges():
            G[u][v]["w"] = random.randint(1, M)
        T = nx.bfs_tree(G, root)
        parent = {root: None}
        for u, v in T.edges():
            parent[v] = u
        G.nodes[root]["p"] = 0
        branch_base_ratio = {}
        for v in nx.topological_sort(T):
            if v == root:
                continue
            u = parent[v]
            w_v = G[u][v]["w"]
            if u == root:
                r0 = random.uniform(0.3, 1.5)
                branch_base_ratio[v] = r0
                r_v = r0
            else:
                r_u = branch_base_ratio[u]
                if type == "nondecreasing":
                    r_v = r_u + random.uniform(0.0, 0.6)
                elif type == "nonincreasing":
                    r_v = max(0.05, r_u - random.uniform(0.0, 0.6))
                else:
                    r_v = random.uniform(0.3, 1.5)
                branch_base_ratio[v] = r_v
            G.nodes[v]["p"] = max(1, int(math.floor(w_v * r_v)))
        if type == "random":
            for v in G.nodes():
                G.nodes[v]["p"] = random.randint(1, M) if v != root else 0
        elif type == "inverse":
            for v in G.nodes():
                if v == root:
                    G.nodes[v]["p"] = 0
                else:
                    u = parent[v]
                    w_v = G[u][v]["w"]
                    G.nodes[v]["p"] = max(1, M + 1 - w_v)

    def generate_tree(n, S, type="lines"):
        assert n >= S + 1, "Need at least S+1 nodes"
        assert type in {"lines", "random", "balanced"}
        G = nx.Graph()
        G.add_node(0)
        nodes = list(range(1, n))
        random.shuffle(nodes)
        root_children = nodes[:S]
        rest = nodes[S:]
        for c in root_children:
            G.add_edge(0, c)
        if type == "lines":
            parts = [[] for _ in range(S)]
            for i, v in enumerate(rest):
                parts[i % S].append(v)
            for c, part in zip(root_children, parts):
                parent = c
                for v in part:
                    G.add_edge(parent, v)
                    parent = v
        elif type == "random":
            parents = root_children.copy()
            for v in rest:
                parent = random.choice(parents)
                G.add_edge(parent, v)
                parents.append(v)
        elif type == "balanced":
            queues = {c: deque([c]) for c in root_children}
            for v in rest:
                c = min(queues, key=lambda x: len(queues[x]))
                parent = queues[c][0]
                G.add_edge(parent, v)
                queues[c].append(v)
                queues[c].popleft()
                queues[c].appendleft(parent)
        return G

    def pareto_filter_fast(front):
        if not front:
            return []
        front_sorted = sorted(front, key=lambda x: (x[0], -x[1]))
        efficient = []
        max_profit_so_far = -float("inf")
        for w, p in front_sorted:
            if p > max_profit_so_far:
                efficient.append((w, p))
                max_profit_so_far = p
        return efficient

    pareto_filter = pareto_filter_fast

    def subtours_generator(G: nx.Graph):
        root = 0
        all_nodes = list(G.nodes())
        others = [v for v in all_nodes if v != root]
        subtours = []
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                nodes = set(subset) | {root}
                if nx.is_connected(G.subgraph(nodes)):
                    subtours.append(nodes)
        return subtours

    def evaluate_tour(G, P):
        profit = sum(G.nodes[v]["p"] for v in P)
        weight = 0
        for u, v in G.subgraph(P).edges():
            weight += G[u][v]["w"]
        return 2 * weight, profit

    def efficient_tours(G, subtours):
        evaluated = []
        for P in subtours:
            w, p = evaluate_tour(G, P)
            evaluated.append((P, w, p))
        wp_pairs = [(w, p) for _, w, p in evaluated]
        efficient_wp = set(pareto_filter(wp_pairs))
        efficient = []
        for P, w, p in evaluated:
            if (w, p) in efficient_wp:
                efficient.append((P, w, p))
        return efficient

    def split_up_graph(G, S):
        root = 0
        children = list(G.neighbors(root))
        if len(children) != S:
            raise ValueError(f"Expected {S} children of root, got {len(children)}")
        T = nx.bfs_tree(G, root)
        subtrees = []
        for child in children:
            nodes = {root, child}
            nodes |= nx.descendants(T, child)
            H = G.subgraph(nodes).copy()
            subtrees.append(H)
        return subtrees

    def combine_two_fronts(Y1, Y2):
        combined = []
        for w1, p1 in Y1:
            for w2, p2 in Y2:
                combined.append((w1 + w2, p1 + p2))
        return pareto_filter(combined)

    def find_tour_with_weight(G, target):
        subtours = subtours_generator(G)
        for P in subtours:
            w, p = evaluate_tour(G, P)
            if target == (w, p):
                return P
        return None

    # ---- methods ----

    def full_approach(G):
        subtours = subtours_generator(G)
        efficient = efficient_tours(G, subtours)
        return efficient

    def decomposed_approach(G, S, return_fronts=False):
        fronts = []
        for subtree in split_up_graph(G, S):
            subtours = subtours_generator(subtree)
            front = [(w, p) for _, w, p in efficient_tours(subtree, subtours)]
            fronts.append(front)
        global_front = reduce(combine_two_fronts, fronts)
        if return_fronts:
            return global_front, fronts
        return global_front

    def tree_dp(G, root=0):
        T = nx.bfs_tree(G, root)
        children = {v: [] for v in T.nodes()}
        for u, v in T.edges():
            children[u].append(v)

        def dp(v):
            fronts = [[(0, 0)]]
            for c in children[v]:
                child_front = dp(c)
                edge_cost = G[v][c]["w"]
                use_child = [(w + 2 * edge_cost, p) for w, p in child_front]
                fronts.append(pareto_filter([(0, 0)] + use_child))
            combined = reduce(combine_two_fronts, fronts)
            return [(w, p + G.nodes[v]["p"]) for w, p in combined]

        return pareto_filter(dp(root))

    def dp_decomposed(G, S, return_fronts=False):
        fronts = []
        for subtree in split_up_graph(G, S):
            front = tree_dp(subtree)
            fronts.append(front)
        global_front = reduce(combine_two_fronts, fronts)
        if return_fronts:
            return global_front, fronts
        return global_front

    # ---- comp_study constants ----

    N_VALUES = [50, 60, 70, 80, 90, 100]
    S_VALUES = [4, 6, 8, 10, 20]
    TREE_TYPES = ["lines", "random", "balanced"]
    WEIGHT_TYPES = ["random", "inverse", "nondecreasing", "nonincreasing"]
    METHODS_TO_TEST = ["dp", "dp_decomp"]

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]


@app.cell
def _(mo):
    mo.md("""
    # Pareto‑Optimal Subtours on Rooted Trees

    Interactive demo for:
    - rooted tree generation (for given n and S)
    - different weight / profit models
    - Pareto fronts of feasible subtours
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(6, 40, value=12, label="Number of nodes (n)")
    S = mo.ui.slider(1, 5, value=3, label="Number of root subtrees (S)")

    tree_type = mo.ui.dropdown(
        ["lines", "random", "balanced"],
        value="lines",
        label="Tree structure",
    )

    weight_type = mo.ui.dropdown(
        ["random", "inverse", "nondecreasing", "nonincreasing"],
        value="nonincreasing",
        label="Profit / weight generation model",
    )

    mo.vstack(
        [
            mo.md("## Instance parameters"),
            n,
            S,
            tree_type,
            weight_type,
        ]
    )
    return S, n, tree_type, weight_type


@app.cell
def _(S, generate_tree, generate_weights, n, random, tree_type, weight_type):
    random.seed(0)

    G = generate_tree(
        n=n.value,
        S=S.value,
        type=tree_type.value,
    )

    generate_weights(G, type=weight_type.value)
    return (G,)


@app.cell
def _(G, go, mo, nx):
    _root = 0
    _T = nx.bfs_tree(G, _root)
    _parent = {_root: None}
    _depth = {_root: 0}
    for _u, _v in _T.edges():
        _parent[_v] = _u
        _depth[_v] = _depth[_u] + 1

    _depths = []
    _ratios = []
    _labels = []

    for _v in G.nodes():
        if _v == _root:
            continue
        _u = _parent[_v]
        _w = G[_u][_v]["w"]
        _p = G.nodes[_v]["p"]
        _depths.append(_depth[_v])
        _ratios.append(_p / _w)
        _labels.append(f"node {_v}<br>p={_p}, w={_w}")

    _fig = go.Figure(
        go.Scatter(
            x=_depths,
            y=_ratios,
            mode="markers",
            marker=dict(size=8),
            text=_labels,
            hoverinfo="text",
        )
    )
    _fig.update_layout(
        title="Ratio pᵢ / wᵢ along root–leaf paths",
        xaxis_title="Depth in tree",
        yaxis_title="pᵢ / wᵢ",
        height=400,
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(G, P, S, colors, go, mo, nx, split_up_graph):
    root = 0
    pos = nx.spring_layout(G, seed=2)

    edge_x, edge_y = [], []
    edge_label_x, edge_label_y, edge_labels = [], [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = G[u][v].get("w", "?")
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_label_x.append((x0 + x1) / 2)
        edge_label_y.append((y0 + y1) / 2)
        edge_labels.append(str(w))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="gray", width=2), hoverinfo="none", showlegend=False,
    )
    edge_label_trace = go.Scatter(
        x=edge_label_x, y=edge_label_y, mode="text", text=edge_labels,
        textposition="middle center", textfont=dict(size=12, color="black"),
        hoverinfo="none", showlegend=False,
    )

    subtrees = split_up_graph(G, S.value)
    subtree_traces = []

    for j, H in enumerate(subtrees):
        nodes = set(H.nodes()) - {root}
        xs, ys, labels = [], [], []
        for v in nodes:
            x, y = pos[v]
            profit = G.nodes[v].get("p", "?")
            xs.append(x)
            ys.append(y)
            labels.append(str(profit))
        subtree_traces.append(
            go.Scatter(
                x=xs, y=ys, mode="markers+text", text=labels,
                textposition="top center",
                marker=dict(size=18, color=colors[j % len(colors)],
                            line=dict(width=1, color="black")),
                name=f"Subtree {j+1}", hoverinfo="none",
            )
        )

    x0, y0 = pos[root]
    root_trace = go.Scatter(
        x=[x0], y=[y0], mode="markers+text", text=["0"],
        textposition="top center",
        marker=dict(size=20, color="black", line=dict(width=2, color="white")),
        hoverinfo="none", name="Root",
    )

    if P:
        xs, ys = [], []
        for v in P:
            x, y = pos[v]
            xs.append(x)
            ys.append(y)
        subtree_traces.append(
            go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=24, color="orange", symbol="star"),
                name="Selected subtour", hoverinfo="none",
            )
        )

    fig = go.Figure(
        data=[edge_trace, edge_label_trace, root_trace] + subtree_traces,
        layout=go.Layout(
            title="Tree instance (node profits & edge weights shown)",
            showlegend=True, hovermode=False,
            xaxis=dict(visible=False), yaxis=dict(visible=False), height=550,
        ),
    )
    mo.ui.plotly(fig)
    return


@app.cell
def _(mo):
    use_full = mo.ui.checkbox(value=False, label="Full enumeration")
    use_decomp = mo.ui.checkbox(value=False, label="Decomposed enumeration")
    use_dp = mo.ui.checkbox(value=True, label="DP")
    use_dp_decomp = mo.ui.checkbox(value=True, label="DP + decomposition")

    mo.vstack(
        [
            mo.md("## Enabled methods"),
            use_full,
            use_decomp,
            use_dp,
            use_dp_decomp,
        ]
    )

    use_full, use_decomp, use_dp, use_dp_decomp
    return use_decomp, use_dp, use_dp_decomp, use_full


@app.cell
def _(
    G,
    S,
    decomposed_approach,
    dp_decomposed,
    full_approach,
    time,
    tree_dp,
    use_decomp,
    use_dp,
    use_dp_decomp,
    use_full,
):
    results = {}
    subfronts = {}
    timings = {}

    if use_full.value:
        t0 = time.perf_counter()
        res = full_approach(G)
        timings["full"] = time.perf_counter() - t0
        results["full"] = [(w, p) for (_, w, p) in res]

    if use_decomp.value:
        t0 = time.perf_counter()
        _front = decomposed_approach(G, S.value)
        timings["decomp"] = time.perf_counter() - t0
        results["decomp"] = _front

    if use_dp.value:
        t0 = time.perf_counter()
        _front = tree_dp(G)
        timings["dp"] = time.perf_counter() - t0
        results["dp"] = _front

    if use_dp_decomp.value:
        t0 = time.perf_counter()
        global_front, fronts = dp_decomposed(G, S.value, return_fronts=True)
        timings["dp_decomp"] = time.perf_counter() - t0
        results["dp_decomp"] = global_front
        subfronts["dp_decomp"] = fronts
    return results, subfronts, timings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparing solution methods
    - Check the boxes to enable/disable methods (the enumeration methods get slow)
    - **Click on points in the Pareto front to see the corresponding subtour highlighted in the tree view** (this took some time to implement, so please try it :) )
    - Below the plot you can also see the running times for the enabled methods.
    """)
    return


@app.cell
def _(colors, go, mo, results, subfronts):
    _fig = go.Figure()
    _fig.update_layout(clickmode="event+select", hovermode="closest")
    for trace in _fig.data:
        trace.update(
            selected=dict(marker=dict(size=12, opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.4)),
        )

    for method, front in results.items():
        ws = [w for w, _ in front]
        ps = [p for _, p in front]
        _fig.add_trace(
            go.Scatter(x=ws, y=ps, mode="markers", name=method, marker=dict(size=9))
        )

    if "dp_decomp" in subfronts:
        _label = "(\u2295_{s\u2208S}Y^s_N)_N"
        _fig.add_trace(
            go.Scatter(
                x=ws, y=ps, mode="markers", name=_label,
                marker=dict(size=6, color="black", symbol="diamond"),
            )
        )
        for i, front in enumerate(subfronts["dp_decomp"]):
            ws = [w for w, _ in front]
            ps = [p for _, p in front]
            _fig.add_trace(
                go.Scatter(
                    x=ws, y=ps, mode="markers",
                    marker=dict(size=7, symbol="x", color=colors[i % len(colors)]),
                    name=f"Y^{i+1}", showlegend=True,
                )
            )

    _fig.update_layout(
        title="Pareto fronts",
        xaxis_title="Total tour weight",
        yaxis_title="Total profit",
        height=500,
    )

    pareto_plot = mo.ui.plotly(_fig)
    pareto_plot
    return (pareto_plot,)


@app.cell
def _(G, find_tour_with_weight, mo, pareto_plot):
    value = pareto_plot.value

    if not value:
        P = None
        _info = mo.md("_No point selected yet._")
    else:
        _x = value[0]["x"]
        _y = value[0]["y"]
        P = find_tour_with_weight(G, target=(_x, _y))
        _info = mo.md(f"**Selected:** weight={_x}, profit={_y}")

    _info
    return (P,)


@app.cell
def _(mo, timings):
    if not timings:
        output = mo.md("_No methods enabled._")
    else:
        rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in timings.items())
        output = mo.md(f"""
## Running times

| Method | Time (s) |
|--------|----------|
{rows}
""")
    output
    return


@app.cell
def _(METHODS_TO_TEST, N_VALUES, S_VALUES, TREE_TYPES, WEIGHT_TYPES, mo):
    table_md = (
        "| Parameter | Values |\n"
        "|-----------|--------|\n"
        f"| N (nodes) | {', '.join(map(str, N_VALUES))} |\n"
        f"| S (root subtrees) | {', '.join(map(str, S_VALUES))} |\n"
        f"| Tree types | {', '.join(TREE_TYPES)} |\n"
        f"| Weight types | {', '.join(WEIGHT_TYPES)} |\n"
        f"| Methods tested | {', '.join(METHODS_TO_TEST)} |\n"
    )

    mo.md(f"""
## Computational Study Configuration

The following table summarizes the parameter ranges and methods
used in the computational study.

{table_md}
""")
    return


@app.cell
def _(go, mo):
    _df = pd.read_csv(str(mo.notebook_location() / "public" / "res.csv"))

    _avg = _df.groupby(["tree_type", "weight_type", "n"], as_index=False)[
        ["dp", "dp_decomp"]
    ].mean()

    _tree_types = sorted(_avg["tree_type"].unique())
    _weight_types = sorted(_avg["weight_type"].unique())

    _method_colors = {
        "dp": "#1f77b4",
        "dp_decomp": "#ff7f0e",
    }

    _fig = make_subplots(
        rows=len(_tree_types),
        cols=len(_weight_types),
        shared_xaxes=True,
        shared_yaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
        subplot_titles=[f"{t} | {w}" for t in _tree_types for w in _weight_types],
    )

    for _i, _tree in enumerate(_tree_types, start=1):
        for _j, _weight in enumerate(_weight_types, start=1):
            _sub = _avg[(_avg["tree_type"] == _tree) & (_avg["weight_type"] == _weight)]
            if _sub.empty:
                continue
            _fig.add_trace(
                go.Scatter(
                    x=_sub["n"], y=_sub["dp"], mode="lines+markers", name="dp",
                    line=dict(dash="dash", color=_method_colors["dp"]),
                    marker=dict(size=6, color=_method_colors["dp"]),
                    showlegend=(_i == 1 and _j == 1),
                ),
                row=_i, col=_j,
            )
            _fig.add_trace(
                go.Scatter(
                    x=_sub["n"], y=_sub["dp_decomp"], mode="lines+markers",
                    name="dp_decomp",
                    line=dict(dash="solid", color=_method_colors["dp_decomp"]),
                    marker=dict(size=6, color=_method_colors["dp_decomp"]),
                    showlegend=(_i == 1 and _j == 1),
                ),
                row=_i, col=_j,
            )

    _fig.update_layout(
        title="Runtime vs n (averaged over S)",
        height=300 * len(_tree_types),
        width=350 * len(_weight_types),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _fig.update_xaxes(title_text="n")
    _fig.update_yaxes(title_text="Runtime (seconds)", type="log")

    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    _content = (mo.notebook_location() / "public" / "comp_study_res.md").read_text()
    mo.md(_content)
    return


if __name__ == "__main__":
    app.run()
