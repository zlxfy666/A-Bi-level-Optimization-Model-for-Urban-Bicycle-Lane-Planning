import numpy as np
import scipy.sparse as sp
import igraph as ig

class SparseGraph:
    def __init__(self, n: int, edge_list: list[tuple[int, int, float]]):
        """
        初始化稀疏图
        :param n: 节点数
        :param edge_list: [(u, v, w), ...] 格式的边列表
        """
        self.n = n

        # 构建 CSR 稀疏矩阵
        rows, cols, weights = zip(*edge_list) if edge_list else ([], [], [])
        self.adj = sp.csr_matrix((np.ones(len(edge_list)), (rows, cols)), shape=(n, n))

        # 初始权重（只读）
        self.orig_weights = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
        # 当前权重（可更新）
        self.curr_weights = self.orig_weights.copy()

        # 构建 igraph
        self.g = ig.Graph(directed=True)
        self.g.add_vertices(self.n)
        edges = [(u, v) for u, v, _ in edge_list]
        self.g.add_edges(edges)
        self.g.es["weight"] = list(weights)

        # 建立 (u,v) -> edge_id 的索引
        self.edge_index = {}
        for eid, (u, v) in enumerate(edges):
            self.edge_index[(u, v)] = eid

    def total_weight_sum(self, use_current: bool = False) -> float:
        """
        计算图中所有边的权重总和。

        :param use_current:
            True -> 使用当前权重 self.curr_weights
            False -> 使用初始权重 self.orig_weights
        :return: float，总权重之和
        """
        M = self.curr_weights if use_current else self.orig_weights
        # 直接用稀疏矩阵求和
        total = float(M.sum())
        return total

    def dijkstra_all_from_source(self, source: int):
        """
        从单个源点 source 出发，计算到所有节点的最短路径和距离。
        这是一个 1-to-all Dijkstra（只执行一次 igraph 的最短路计算）。

        返回:
            dist_vec: list[float]       # source 到所有节点的最短距离
            paths_dict: dict[int, list[int]]  # 目标节点 s -> 节点路径序列
        """
        # 1️⃣ 距离向量（r→所有节点）
        dist_vec = self.g.distances(
            source=source, weights="weight", mode="OUT", algorithm="dijkstra"
        )[0]

        # 2️⃣ 路径集合（r→s 的节点序列）
        all_nodes = list(range(self.n))
        paths_all = self.g.get_shortest_paths(
            source, to=all_nodes, weights="weight", mode="OUT", output="vpath"
        )

        # 构建结果字典：目标节点 -> 路径节点序列
        paths_dict = {s: paths_all[i] for i, s in enumerate(all_nodes)}

        return dist_vec, paths_dict

    def dijkstra_distance(self, source: int, target=None):
        """
        最短路距离（数值）
        """
        if target is None:
            return self.g.distances(source=source, weights="weight", mode="OUT", algorithm="dijkstra")[0]
        else:
            return self.g.distances(source=source, target=target, weights="weight", mode="OUT", algorithm="dijkstra")[0]

    def dijkstra_path(self, source: int, target: int):
        """
        最短路路径（节点序列 + 路径长度）
        """
        paths = self.g.get_shortest_paths(source, to=target, weights="weight", mode="OUT", output="vpath")
        edge_paths = self.g.get_shortest_paths(source, to=target, weights="weight", mode="OUT", output="epath")

        if not paths or not paths[0]:
            return [], float("inf")  # 没有路径

        path_nodes = paths[0]
        path_edges = edge_paths[0]
        path_weight = sum(self.g.es[e]["weight"] for e in path_edges)

        return path_nodes, path_weight

    def dijkstra_to_many(self, source: int, targets: list[int] | None = None):
        """
        从单个源点 source 到多个目标 targets 的最短路（一次性求解）。
        返回:
            paths_dict: dict[int, list[int]]   # 目标节点 s -> 节点路径序列（vpath）
            dist_dict:  dict[int, float]       # 目标节点 s -> 最短距离（标量）
        """
        # 统一成列表，并确保是 int
        if targets is None:
            targets = list(range(self.n))
        elif isinstance(targets, int):
            targets = [int(targets)]
        else:
            targets = [int(t) for t in targets]

        # ！！！注意：get_shortest_paths 要用位置参数传 source（v）
        vpaths = self.g.get_shortest_paths(
            source,  # <-- 用位置参数
            to=targets,
            weights="weight",
            mode="OUT",
            output="vpath",
        )

        # distances 也用位置参数形式更稳妥
        dmat = self.g.distances(
            source,  # <-- 用位置参数
            targets,
            weights="weight",
            mode="OUT",
            algorithm="dijkstra",
        )
        drow = dmat[0] if dmat else []

        paths_dict = {t: vpaths[i] for i, t in enumerate(targets)}
        dist_dict = {t: float(drow[i]) for i, t in enumerate(targets)}
        return paths_dict, dist_dict

    def update_edge(self, u: int, v: int, new_w: float):
        """
        更新单条边权重（不覆盖初始）
        """
        # 如果必须全为正，直接取绝对值
        new_w = abs(new_w)

        self.curr_weights[u, v] = new_w
        if (u, v) in self.edge_index:
            eid = self.edge_index[(u, v)]
            self.g.es[eid]["weight"] = new_w
        else:
            raise ValueError(f"边 ({u}, {v}) 不存在")

    def update_edges(self, updates: list[tuple[int, int, float]]):
        """
        批量更新权重
        """
        for u, v, w in updates:
            self.update_edge(u, v, w)

    def reset_weights(self):
        """
        恢复初始权重
        """
        self.curr_weights = self.orig_weights.copy()
        coo = self.curr_weights.tocoo()
        for u, v, w in zip(coo.row, coo.col, coo.data):
            if (u, v) in self.edge_index:
                eid = self.edge_index[(u, v)]
                self.g.es[eid]["weight"] = w

    def evaluate_total_cost(self, res, OD: dict, demand: dict, beta: dict, solver):
        """
        根据 res["y_opt"] 的解，计算:
        sum_g sum_{(r,s) in OD[g]} demand[g,r,s] * shortest_path_length_g(r,s)

        其中对每个 g，若 y[a] = 1，则对应弧的权重变为 orig_weight * (1 - beta[g])。

        :param res: solver.solve() 的结果 (包含 "y_opt")
        :param OD: dict[g] -> [(r,s)]
        :param demand: dict[(g,r,s)] -> float
        :param beta: dict[g] -> float
        :param solver: HPRSolver 对象 (用于取 arcs, orig_weights)
        :return: float，总费用
        """
        total_cost = 0.0

        for g in OD:
            # 1. 恢复初始权重
            self.reset_weights()

            # 2. 更新 y=1 的边权重
            updates = []
            for a, y_val in res["y_opt"].items():
                if y_val == 1:
                    u, v = solver.arcs[a]
                    w0 = solver.graph.orig_weights[u, v]
                    new_w = float(w0 * (1 - beta[g]))
                    updates.append((u, v, new_w))
            if updates:
                self.update_edges(updates)

            # 3. 计算该 g 下所有 OD 的距离*流量
            for (r, s) in OD[g]:
                dist = self.dijkstra_distance(r, target=s)
                flow = demand[(g, r, s)]
                total_cost += dist * flow

        return total_cost

if __name__ == "__main__":
    # 构建一个小图
    edges = [
        (0, 1, 2),
        (0, 2, 5),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 3)
    ]
    G = SparseGraph(4, edges)

    # 只看距离
    print("最短距离 from 0:", G.dijkstra_distance(0))

    # 具体路径
    print("最短路径 0 → 3:", G.dijkstra_path(0, 3))

    # 更新权重
    G.update_edge(0, 2, 1)
    print("更新后最短路径 0 → 3:", G.dijkstra_path(0, 3))

    # 恢复权重
    G.reset_weights()
    print("恢复后最短路径 0 → 3:", G.dijkstra_path(0, 3))