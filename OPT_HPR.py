from Network import *
import gurobipy as gp
from collections import defaultdict
BIG_INF_DEFAULT = 1e12
from collections import Counter
from gurobipy import Model, GRB, quicksum
BIG_INF_DEFAULT = 1e12

class HPRSolver:
    def __init__(self, graph, B, G, OD, demand, beta, verbose=False, time_ub=None):
        """
        :param graph: SparseGraph 实例
        :param B: 预算
        :param G: 流类别集合 (list 或 set)
        :param OD: dict[g] -> [(r,s)]
        :param demand: dict[(g,r,s)] -> float
        :param beta: dict[g] -> float
        """
        self.seen_paths = defaultdict(set)  # (g,r,s) -> {path_key,...}
        self.path_counter = {}  # (g,r,s) -> 下一个 p
        self._demand_rhs_set = set()  # 已设置 RHS 的 (g,r,s)
        self.graph = graph
        self.B = B
        self.G = G
        self.OD = OD
        self.demand = demand
        self.beta = beta
        self.verbose = verbose

        coo = graph.orig_weights.tocoo()
        self.arcs = list(zip(coo.row, coo.col))
        self.L = list(coo.data)   # 长度/成本
        self.T = list(coo.data)   # 时间（可与 L 不同，这里简化一致）
        self.A = len(self.arcs)

        self.m = Model("HPR_S")
        if not verbose:
            self.m.Params.OutputFlag = 0

        # 每类总需求，用于 Big-M
        self.Mg = {g: sum(self.demand[(g, r, s)] for (r, s) in self.OD[g])
                   for g in self.G}

        # 逐类逐弧的更紧上界 U_ga
        self.U_ga = self._compute_U_ga()

        # === 变量 ===
        self.x = {(g, a): self.m.addVar(lb=0.0, name=f"x[{g},{a}]")
                  for g in G for a in range(self.A)}
        self.y = {a: self.m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                   name=f"y[{a}]")
                  for a in range(self.A)}
        self.w = {(g, a): self.m.addVar(lb=0.0, name=f"w[{g},{a}]")
                  for g in G for a in range(self.A)}

        # === 预算约束 ===
        self.cons_budget = self.m.addConstr(
            quicksum(self.y[a] * self.L[a] for a in range(self.A)) <= B,
            name="budget"
        )

        # === 弧覆盖初始化：x[g,a] - Σ_{r,s,p: a∈p} h[g,r,s,p] ≥ 0 ===
        # 先建成 x[g,a] ≥ 0 的“壳”，add_path 时在 LHS 上对 h 加 -1 系数
        self.cons_flow_link = {}
        for g in G:
            for a in range(self.A):
                self.cons_flow_link[(g, a)] = self.m.addConstr(
                    self.x[g, a] >= 0.0,
                    name=f"flow_link[{g},{a}]"
                )

        # === 需求初始化：Σ_p h[g,r,s,p] ≥ demand[g,r,s]（一开始 LHS=0） ===
        # 直接建成“≥ demand”的约束，之后只在 LHS 对 h 加 +1 系数
        self.cons_demand = {}
        for g in G:
            for (r, s) in OD[g]:
                lhs = gp.LinExpr(0.0)
                self.cons_demand[(g, r, s)] = self.m.addConstr(
                    lhs >= float(self.demand[(g, r, s)]),
                    name=f"demand[{g},{r},{s}]"
                )


        # === Big-M 线性化 for w (更紧) ===
        self.cons_bigM = {}
        for g in G:
            M = self.Mg[g]
            for a in range(self.A):
                self.cons_bigM[(g,a,'w_le_x')] = self.m.addConstr(
                    self.w[(g,a)] <= self.x[(g,a)],
                    name=f"w_le_x[{g},{a}]"
                )
                self.cons_bigM[(g,a,'w_le_My')] = self.m.addConstr(
                    self.w[(g,a)] <= M * self.y[a],
                    name=f"w_le_My[{g},{a}]"
                )
                self.cons_bigM[(g,a,'w_ge_xMy')] = self.m.addConstr(
                    self.w[(g,a)] >= self.x[(g,a)] - M * (1 - self.y[a]),
                    name=f"w_ge_xMy[{g},{a}]"
                )

        # === 时间约束 ===
        self.time_expr = quicksum(
            self.T[a] * self.x[(g, a)] - (1-self.beta[g]) * self.T[a] * self.w[(g, a)]
            for g in self.G for a in range(self.A)
        )
        self.time_ub = BIG_INF_DEFAULT if time_ub is None else float(time_ub)
        self.cons_time_cap = self.m.addConstr(
            self.time_expr <= self.time_ub,
            name="time_cap"
        )

        # === 目标函数：min ∑ L_a (x_ga - t_ga) ===
        self.m.setObjective(
            quicksum((self.x[(g, a)] - self.w[(g, a)]) * self.L[a]
                     for g in G for a in range(self.A)),
            GRB.MINIMIZE
        )

        self.m.update()
        self._init_args = (graph, B, G, OD, demand, beta, verbose, time_ub)

        # 路径相关
        self.h = {}
        self.path_counter = {}

    def add_equal_y_constraints(self, arc_pairs):
        """
        给定弧对列表 arc_pairs = [(i, j), (i2, j2), ...]，
        强制这些弧对应的 y 变量必须相等。
        """
        for (i, j) in arc_pairs:
            if i < 0 or i >= self.A or j < 0 or j >= self.A:
                raise ValueError(f"非法弧索引: {(i, j)}")
            self.m.addConstr(self.y[i] - self.y[j] == 0,
                             name=f"equal_y[{i},{j}]")
        self.m.update()

    def add_symmetric_bike_lane_constraints(self):
        """
        对所有相反方向的弧 (i,j) 和 (j,i)，
        强制其对应的 y 变量必须相同。
        """
        arc_dict = {arc: idx for idx, arc in enumerate(self.arcs)}
        seen = set()

        for idx, (u, v) in enumerate(self.arcs):
            if (v, u) in arc_dict and (v, u) not in seen:
                idx2 = arc_dict[(v, u)]
                self.m.addConstr(self.y[idx] - self.y[idx2] == 0,
                                 name=f"symmetric_y[{u},{v}]")
                # 记住已处理过的
                seen.add((u, v))
                seen.add((v, u))

        self.m.update()

    def _compute_U_ga(self):
        """
        为每个 (g,a) 计算更紧的上界 U_{g,a}：
        仅在存在 r->i 与 j->s 可达时，才把 d_{grs} 计入。
        """
        U_ga = {}
        n = self.graph.n
        dist_from = [self.graph.g.distances(source=v, weights=None, mode="OUT")[0]
                     for v in range(n)]
        INF = float("inf")
        for g in self.G:
            for a, (i, j) in enumerate(self.arcs):
                cap = 0.0
                for (r, s) in self.OD[g]:
                    if dist_from[r][i] < INF and dist_from[j][s] < INF:
                        cap += self.demand[(g, r, s)]
                U_ga[(g, a)] = cap
        return U_ga

    def add_path(self, g, r, s, path_edges, *, consider_order: bool = True, update_model: bool = True):
        """
        形成两类约束：
          1) 需求： Σ_p h[g,r,s,p] ≥ demand[g,r,s]
          2) 弧覆盖：x[g,a] ≥ Σ_{r,s} Σ_{p: a∈path} h[g,r,s,p]

        判重：
          - consider_order=True  : 以路径的“边序列”判重（tuple(path_edges)）
          - consider_order=False : 以“边集合”判重（frozenset(path_edges)）

        返回：True=新增，False=重复已忽略
        """
        key_grs = (g, r, s)

        # 1) 空路径直接忽略
        if not path_edges:
            if self.verbose:
                print(f"[add_path] 空路径被忽略: g={g}, r={r}, s={s}")
            return False

        # 2) 生成判重键
        path_key = tuple(path_edges) if consider_order else frozenset(path_edges)

        # 3) 去重
        if path_key in self.seen_paths[key_grs]:
            if self.verbose:
                print(f"[add_path] 路径重复，跳过: g={g}, r={r}, s={s}, path={path_edges}")
            return False

        # 4) 新建路径变量
        p = self.path_counter.get(key_grs, 0)
        h_var = self.m.addVar(lb=0.0, name=f"h[{g},{r},{s},{p}]")
        self.h[(g, r, s, p)] = h_var

        # 5) 弧覆盖：x[g,a] - Σ h ≥ 0
        #    若同一弧在 path 中出现多次（极少见），合并计数减少多次 chgCoeff 调用
        counts = Counter(path_edges) if consider_order else {a: 1 for a in path_edges}
        for a, cnt in counts.items():
            self.m.chgCoeff(self.cons_flow_link[(g, a)], h_var, -float(cnt))

        # 6) 需求：Σ h ≥ demand[g,r,s]   （只需把 RHS 设一次）
        self.m.chgCoeff(self.cons_demand[(g, r, s)], h_var, +1.0)
        if key_grs not in self._demand_rhs_set:
            self.cons_demand[(g, r, s)].RHS = self.demand[(g, r, s)]
            self._demand_rhs_set.add(key_grs)

        # 7) 记录并收尾
        self.seen_paths[key_grs].add(path_key)
        self.path_counter[key_grs] = p + 1
        if update_model:
            self.m.update()

        if self.verbose:
            print(f"[add_path] 新路径: g={g}, r={r}, s={s}, p={p}, edges={path_edges}")
        return True

    def solve(self):
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            if self.m.Status == 3:
                return 3
            raise RuntimeError("LP 未最优")

        obj_val = self.m.ObjVal
        y_opt = {a: self.y[a].X for a in range(self.A)}

        # === 提取对偶变量 ===
        dual_demand = {key: cons.Pi for key, cons in self.cons_demand.items()}
        dual_flow_link = {key: cons.Pi for key, cons in self.cons_flow_link.items()}

        return {
            "obj_val": obj_val,
            "y_opt": y_opt,
            "dual_demand": dual_demand,
            "dual_flow_link": dual_flow_link
        }

    def set_time_upper_bound(self, ub=None):
        new_ub = BIG_INF_DEFAULT if ub is None else float(ub)
        self.cons_time_cap.RHS = new_ub
        self.time_ub = new_ub
        self.m.update()

    def get_arc_index(self, i, j):
        try:
            return self.arcs.index((i, j))
        except ValueError:
            return None

    def reset(self):
        args = self._init_args
        self.__init__(*args)

    def get_path_edges(self, node_list):
        path_edges = []
        for u, v in zip(node_list[:-1], node_list[1:]):
            arc_idx = self.get_arc_index(u, v)
            if arc_idx is None:
                raise ValueError(f"弧 ({u},{v}) 不存在于图中！")
            path_edges.append(arc_idx)
        return path_edges

    def fix_y(self, forbidden=None, set_to=None):
        """
        :param forbidden: list/iterable of arc indices -> 强制 y=0
        :param set_to: list/iterable of arc indices -> 强制 y=1
        """
        forbidden = set(forbidden) if forbidden else set()
        set_to = set(set_to) if set_to else set()

        for a in range(self.A):
            # 默认不改动，除非在 forbidden 或 set_to
            if a in forbidden:
                self.y[a].LB = 0
                self.y[a].UB = 0
            elif a in set_to:
                self.y[a].LB = 1
                self.y[a].UB = 1

        self.m.update()

    def reset_y_bounds(self):
        """恢复所有 y 的上下界为 [0,1]"""
        for a in range(self.A):
            self.y[a].LB = 0
            self.y[a].UB = 1
        self.m.update()

    def fix_outside_groups_to_zero(self, groups, verbose: bool = False):
        """
        将所有 *未出现在 groups 中* 的弧 (u,v) 的 y[a] 固定为 0。
        也就是只保留 groups 中的弧可选，其余弧被禁用。

        :param groups: list[list[tuple[int,int]]]
                       每个 group 是一组弧 (u,v)
        :param verbose: 是否打印被固定的弧数量
        """
        # 1️⃣ 建立节点对到弧索引的映射
        arc_dict = {arc: idx for idx, arc in enumerate(self.arcs)}

        # 2️⃣ 收集 groups 中所有出现过的弧索引
        allowed_indices = set()
        for group in groups:
            for (u, v) in group:
                if (u, v) not in arc_dict:
                    raise ValueError(f"图中不存在弧 ({u},{v})，请检查 groups 输入。")
                allowed_indices.add(arc_dict[(u, v)])

        # 3️⃣ 对未出现的弧全部固定 y=0
        all_indices = set(range(self.A))
        forbidden_indices = all_indices - allowed_indices

        for a in forbidden_indices:
            self.y[a].LB = 0.0
            self.y[a].UB = 0.0

        self.m.update()

        if verbose:
            print(f"[INFO] 已固定 {len(forbidden_indices)} 条弧的 y=0。")
            print(f"[INFO] 仍可选弧数量: {len(allowed_indices)} / 总弧数 {self.A}")

    def set_budget(self, new_B):
        """
        修改预算约束:
            Σ_a L_a * y[a] <= new_B
        参数:
            new_B : float, 新的预算上限
        """
        self.cons_budget.RHS = float(new_B)
        self.B = float(new_B)
        self.m.update()
        if self.verbose:
            print(f"[INFO] 已将预算上限修改为 {new_B}")

    def reset_budget(self):
        """
        恢复预算约束为初始化时的 B。
        """
        # 原始 B 保存在 self._init_args[1]
        original_B = float(self._init_args[1])
        self.cons_budget.RHS = original_B
        self.B = original_B
        self.m.update()
        if self.verbose:
            print(f"[INFO] 已将预算上限恢复为初始值 {original_B}")

    def evaluate_total_cost_given_y(self, y: dict[int, int]) -> float:
        """
        给定 0/1 的弧选择 y[a]（所有弧都应作为 key），计算：
            sum_g sum_{(r,s)∈OD[g]} demand[g,r,s] * dist_g(r,s)
        规则：对每个 g，将所有 y[a]==1 的弧 (u,v) 的权重设为 原始权重 * beta[g]，
             再基于该权重计算该 g 的所有最短路开销并加总。
        结束后会恢复图权重。
        """
        # ---- 基本校验 ----
        if len(y) != self.A:
            raise ValueError(f"y 的键数量({len(y)})与弧数 self.A({self.A}) 不一致")
        bad_keys = [a for a in y.keys() if not (0 <= int(a) < self.A)]
        if bad_keys:
            raise ValueError(f"y 含有非法弧索引: {bad_keys[:5]} ...")
        bad_vals = [v for v in y.values() if int(v) not in (0, 1)]
        if bad_vals:
            raise ValueError("y 的值必须为 0/1")
        self.graph.reset_weights()
        # ---- 预取 y==1 的弧以及其原始权重，避免每个 g 重复取权重 ----
        one_arcs = [int(a) for a, val in y.items() if int(val) == 1]
        uvw0 = []
        for a in one_arcs:
            u, v = self.arcs[a]
            w0 = self.graph.orig_weights[u, v]
            try:
                w0 = float(w0)  # 新版 scipy 稀疏索引大多可直接转 float
            except Exception:
                w0 = float(self.graph.orig_weights[u, v].toarray()[0, 0])
            uvw0.append((u, v, w0))

        total = 0.0
        try:
            for g in self.G:
                # 1) 恢复初始权重
                self.graph.reset_weights()

                # 2) 对 y==1 的弧按照 beta[g] 设置新权重
                if uvw0:
                    beta_g = float(self.beta[g])
                    updates = [(u, v, w0 * beta_g) for (u, v, w0) in uvw0]
                    self.graph.update_edges(updates)

                # 3) 计算该 g 下所有 (r,s) 的 dist * demand
                #    按 r 分组，一次 Dijkstra 求到所有点的距离，减少调用次数
                rs_by_r = defaultdict(list)
                for (r, s) in self.OD[g]:
                    rs_by_r[r].append(s)

                for r, targets in rs_by_r.items():
                    dist_vec = self.graph.dijkstra_distance(r)  # r 到所有节点
                    for s in targets:
                        dist = dist_vec[s]
                        if not np.isfinite(dist):
                            raise ValueError(f"OD 不可达：g={g}, r={r}, s={s}")
                        total += dist * self.demand[(g, r, s)]

            return total
        finally:
            # 4) 收尾：恢复初始权重，避免污染外部状态
            self.graph.reset_weights()

    def evaluate_unbuilt_cost(self, y: dict[int, int], *, use_current_weights: bool = True) -> float:
        """
        计算仅由“未建设弧(y==0)”贡献的总成本：
            cost = Σ_g Σ_(r,s∈OD[g]) demand[g,r,s] * (Σ_{a∈P_g(r,s)且y[a]==0} weight_a)

        其中 P_g(r,s) 是基于图当前权重求出的最短路。默认用当前权重与最短路一致；
        若想用初始权重参与乘积，可将 use_current_weights=False。

        :param y: 字典 a->0/1，包含所有弧（0..self.A-1）
        :param use_current_weights: True 使用 curr_weights；False 使用 orig_weights
        :return: float，总 cost
        """
        # 基本校验
        if len(y) != self.A:
            raise ValueError(f"y 的键数量({len(y)})与弧数 self.A({self.A}) 不一致")
        for a, v in y.items():
            a_int = int(a)
            if a_int < 0 or a_int >= self.A:
                raise ValueError(f"非法弧索引: {a}")
            if int(v) not in (0, 1):
                raise ValueError(f"y[{a}] 的值必须为 0/1，实际为 {v}")

        one_arcs = [int(a) for a, val in y.items() if int(val) == 1]
        uvw0 = []
        for a in one_arcs:
            u, v = self.arcs[a]
            w0 = self.graph.orig_weights[u, v]
            try:
                w0 = float(w0)  # 新版 scipy 稀疏索引大多可直接转 float
            except Exception:
                w0 = float(self.graph.orig_weights[u, v].toarray()[0, 0])
            uvw0.append((u, v, w0))
        self.graph.reset_weights()
        M = self.graph.curr_weights if use_current_weights else self.graph.orig_weights
        cost = 0.0

        for g in self.G:
            self.graph.reset_weights()
            # 2) 对 y==1 的弧按照 beta[g] 设置新权重
            if uvw0:
                beta_g = float(self.beta[g])
                updates = [(u, v, w0 * beta_g) for (u, v, w0) in uvw0]
                self.graph.update_edges(updates)
            for (r, s) in self.OD[g]:
                # 1) 求最短路（基于 igraph 当前权重）
                path_nodes, _ = self.graph.dijkstra_path(r, s)
                if not path_nodes:
                    raise ValueError(f"OD 不可达：g={g}, r={r}, s={s}")

                # 2) 将节点序列转成弧索引序列
                path_arc_indices = self.get_path_edges(path_nodes)

                # 3) 只累计 y==0 的弧的权重
                unbuilt_sum = 0.0
                for a in path_arc_indices:
                    if y.get(a, 0) == 0:
                        u, v = self.arcs[a]
                        w = M[u, v]
                        try:
                            w = float(w)
                        except Exception:
                            w = float(M[u, v].toarray()[0, 0])
                        unbuilt_sum += w

                # 4) 乘以需求得到 cost2，并累加
                demand_val = self.demand[(g, r, s)]
                cost2 = demand_val * unbuilt_sum
                cost += cost2
        return cost

    def evaluate_unbuilt_cost_1toall(self, y: dict[int, int], *, use_current_weights: bool = True) -> float:
        """
        计算仅由“未建设弧(y==0)”贡献的总成本：
            cost = Σ_g Σ_(r,s∈OD[g]) demand[g,r,s] * (Σ_{a∈P_g(r,s)且y[a]==0} weight_a)

        其中 P_g(r,s) 是基于图当前权重求出的最短路。默认用当前权重与最短路一致；
        若想用初始权重参与乘积，可将 use_current_weights=False。

        :param y: 字典 a->0/1，包含所有弧（0..self.A-1）
        :param use_current_weights: True 使用 curr_weights；False 使用 orig_weights
        :return: float，总 cost
        """
        # 基本校验
        if len(y) != self.A:
            raise ValueError(f"y 的键数量({len(y)})与弧数 self.A({self.A}) 不一致")
        for a, v in y.items():
            a_int = int(a)
            if a_int < 0 or a_int >= self.A:
                raise ValueError(f"非法弧索引: {a}")
            if int(v) not in (0, 1):
                raise ValueError(f"y[{a}] 的值必须为 0/1，实际为 {v}")

        one_arcs = [int(a) for a, val in y.items() if int(val) == 1]
        uvw0 = []
        for a in one_arcs:
            u, v = self.arcs[a]
            w0 = self.graph.orig_weights[u, v]
            try:
                w0 = float(w0)  # 新版 scipy 稀疏索引大多可直接转 float
            except Exception:
                w0 = float(self.graph.orig_weights[u, v].toarray()[0, 0])
            uvw0.append((u, v, w0))
        self.graph.reset_weights()
        # 选择计费所用的权重矩阵
        M = self.graph.curr_weights if use_current_weights else self.graph.orig_weights

        cost = 0.0

        # --- 按 g 循环 ---
        from collections import defaultdict
        for g in self.G:
            # 1) 恢复初始权重，并应用 y==1 折扣（乘 beta[g]）
            self.graph.reset_weights()
            if uvw0:
                beta_g = float(self.beta[g])
                # 注意：这里遵循你现有逻辑，折扣为 w0 * beta_g
                updates = [(u, v, w0 * beta_g) for (u, v, w0) in uvw0]
                self.graph.update_edges(updates)

            # 2) 将 OD[g] 按源点分组：同一个 r 的所有 s 一起求解
            by_source: dict[int, list[int]] = defaultdict(list)
            for (r, s) in self.OD[g]:
                by_source[int(r)].append(int(s))

            # 3) 每个 r 只跑一次最短路（到多个 s）
            for r, s_list in by_source.items():
                paths_dict, _ = self.graph.dijkstra_to_many(r, s_list)

                for s in s_list:
                    path_nodes = paths_dict.get(s, [])
                    if not path_nodes:
                        raise ValueError(f"OD 不可达：g={g}, r={r}, s={s}")

                    # 节点序列 -> 弧索引序列
                    path_arc_indices = self.get_path_edges(path_nodes)

                    # 仅累计 y==0 的弧的权重（从 M 取值）
                    unbuilt_sum = 0.0
                    for a in path_arc_indices:
                        if y.get(a, 0) == 0:
                            u, v = self.arcs[a]
                            w = M[u, v]
                            # 稳妥取标量
                            try:
                                w = float(w)
                            except Exception:
                                w = float(M[u, v].toarray()[0, 0])
                            unbuilt_sum += w

                    demand_val = self.demand[(g, r, s)]
                    cost += demand_val * unbuilt_sum

        return cost

    def evaluate_unbuilt_cost_1toall_min_unbuilt_on_shortest(
            self,
            y: dict[int, int],
            *,
            use_current_weights: bool = True,
            tol: float = 1e-9,
    ) -> float:
        """
        计算仅由“未建设弧(y==0)”贡献的总成本，但路径选择逻辑改为：

            对每个 (g, r, s)：
                1) 先保证总路径长度（按 M 的权重）最短；
                2) 在所有最短路径里，选 y==0 弧的权重和最小的那一条；
                3) 将 demand[g,r,s] 全部走在这条路径上，只累积其中 y==0 弧的权重。

        等价于用 (dist, unbuilt_dist) 做字典序 Dijkstra：
            dist         = 总距离
            unbuilt_dist = 只在 y==0 弧上累计的距离和

        :param y: 字典 a->0/1，包含所有弧（0..self.A-1）
        :param use_current_weights: True 使用 curr_weights；False 使用 orig_weights
        :param tol: 浮点比较容差
        :return: float，总 cost
        """
        # --- 基本校验，与原函数相同 ---
        if len(y) != self.A:
            raise ValueError(f"y 的键数量({len(y)})与弧数 self.A({self.A}) 不一致")
        for a, v in y.items():
            a_int = int(a)
            if a_int < 0 or a_int >= self.A:
                raise ValueError(f"非法弧索引: {a}")
            if int(v) not in (0, 1):
                raise ValueError(f"y[{a}] 的值必须为 0/1，实际为 {v}")

        # 预处理 y==1 的弧（已建设），用于按 g 施加 beta 折扣
        one_arcs = [int(a) for a, val in y.items() if int(val) == 1]
        uvw0 = []
        for a in one_arcs:
            u, v = self.arcs[a]
            w0 = self.graph.orig_weights[u, v]
            try:
                w0 = float(w0)
            except Exception:
                w0 = float(self.graph.orig_weights[u, v].toarray()[0, 0])
            uvw0.append((u, v, w0))

        # 初始重置权重
        self.graph.reset_weights()
        M = self.graph.curr_weights if use_current_weights else self.graph.orig_weights

        from collections import defaultdict
        import math
        import heapq

        total_cost = 0.0

        # --- 构建邻接表（基于弧索引），只做一次 ---
        # 这里通过 self.arcs 反推出节点数量，如果你类里有 self.N 可以直接用 self.N
        if not self.arcs:
            return 0.0

        max_node = max(max(u, v) for (u, v) in self.arcs)
        num_nodes = max_node + 1

        adj: list[list[int]] = [[] for _ in range(num_nodes)]  # u -> [arc_index,...]
        for a, (u, v) in enumerate(self.arcs):
            adj[u].append(a)

        # --- 二级 Dijkstra：返回从源 r 到所有点的 (dist, unbuilt_dist) ---
        def dijkstra_bicriteria(r: int) -> tuple[list[float], list[float]]:
            INF = float("inf")
            dist = [INF] * num_nodes  # 总距离
            unbuilt = [0.0] * num_nodes  # y==0 弧上的距离和
            dist[r] = 0.0
            unbuilt[r] = 0.0

            # 堆中元素是 (dist, unbuilt_dist, node)
            heap: list[tuple[float, float, int]] = [(0.0, 0.0, r)]

            while heap:
                d, ub, u = heapq.heappop(heap)

                # 如果当前弹出的标签比已经记录的差，就丢弃
                if d > dist[u] + tol:
                    continue
                if abs(d - dist[u]) <= tol and ub > unbuilt[u] + tol:
                    continue

                for a in adj[u]:
                    uu, v = self.arcs[a]
                    # 正常情况下 uu == u，这里防御性检查一下
                    if uu != u:
                        continue

                    w = M[u, v]
                    try:
                        w = float(w)
                    except Exception:
                        w = float(M[u, v].toarray()[0, 0])

                    if w == float("inf"):
                        continue  # 认为不可达

                    new_d = d + w
                    # 只有 y==0 的弧才在次目标里累加
                    extra = w if y.get(a, 0) == 0 else 0.0
                    new_ub = ub + extra

                    # 字典序比较 (new_d, new_ub) vs (dist[v], unbuilt[v])
                    better = False
                    if new_d < dist[v] - tol:
                        better = True
                    elif abs(new_d - dist[v]) <= tol and new_ub < unbuilt[v] - tol:
                        better = True

                    if better:
                        dist[v] = new_d
                        unbuilt[v] = new_ub
                        heapq.heappush(heap, (new_d, new_ub, v))

            return dist, unbuilt

        # --- 按 g 循环 ---
        for g in self.G:
            # 1) 恢复初始权重，并对 y==1 的弧施加 beta[g] 折扣
            self.graph.reset_weights()
            if uvw0:
                beta_g = float(self.beta[g])
                updates = [(u, v, w0 * beta_g) for (u, v, w0) in uvw0]
                self.graph.update_edges(updates)

            # 重取权重矩阵引用（如果 reset_weights/update_edges 是就地修改，这一步可以省略）
            if use_current_weights:
                M = self.graph.curr_weights
            else:
                M = self.graph.orig_weights

            # 2) 将 OD[g] 按源点分组
            by_source: dict[int, list[int]] = defaultdict(list)
            for (r, s) in self.OD[g]:
                by_source[int(r)].append(int(s))

            # 3) 每个 r 做一次“二级 Dijkstra”
            for r, s_list in by_source.items():
                dist, unbuilt_dist = dijkstra_bicriteria(r)

                for s in s_list:
                    if s < 0 or s >= num_nodes:
                        raise ValueError(f"节点索引越界：s={s}")

                    if math.isinf(dist[s]):
                        raise ValueError(f"OD 不可达：g={g}, r={r}, s={s}")

                    # 对于 (g,r,s)，选的路径是：
                    #   - dist 最小；
                    #   - 在 dist 最小的所有路径中，unbuilt_dist 最小。
                    # y==0 弧的距离和就是 unbuilt_dist[s]
                    unbuilt_sum = unbuilt_dist[s]

                    demand_val = self.demand[(g, r, s)]
                    total_cost += demand_val * unbuilt_sum

        return total_cost

    def add_shortest_paths_given_y(self, y: dict[int, int], consider_order: bool = True, verbose=False):
        """
        在固定的 y 下，更新道路网络权重并为每个 (g,r,s) 添加最短路径。
        利用 1-to-all Dijkstra 提升效率（每个 r 只执行一次最短路计算）。

        :param y: dict[int,int], 弧 -> 0/1 表示是否建设
        :param consider_order: 是否考虑路径顺序（传给 add_path）
        :param verbose: 是否打印详细信息
        :return: int，成功添加的路径数量
        """
        # --- 校验 ---
        if len(y) != self.A:
            raise ValueError(f"y 的长度({len(y)})与弧数({self.A})不一致")

        for a, val in y.items():
            if int(val) not in (0, 1):
                raise ValueError(f"y[{a}] 的值必须是 0/1，而不是 {val}")

        added_count = 0

        # --- 提取 y==1 的弧 ---
        one_arcs = [int(a) for a, v in y.items() if int(v) == 1]
        uvw0 = []
        for a in one_arcs:
            u, v = self.arcs[a]
            w0 = self.graph.orig_weights[u, v]
            try:
                w0 = float(w0)
            except Exception:
                w0 = float(self.graph.orig_weights[u, v].toarray()[0, 0])
            uvw0.append((u, v, w0))

        # --- 主循环：每个类别 g 单独处理 ---
        for g in self.G:
            beta_g = float(self.beta[g])
            # 每类开始前恢复原始权重
            self.graph.reset_weights()

            # 已建设弧 w = β_g * w0
            if uvw0:
                updates = [(u, v, w0 * beta_g) for (u, v, w0) in uvw0]
                self.graph.update_edges(updates)

            # --- 对每个起点 r 只跑一次 Dijkstra ---
            rs_by_r = defaultdict(list)
            for (r, s) in self.OD[g]:
                rs_by_r[r].append(s)

            for r, targets in rs_by_r.items():
                # 一次性获取 r → 所有节点的距离与路径
                dist_vec, paths_dict = self.graph.dijkstra_all_from_source(r)

                for s in targets:
                    path_nodes = paths_dict.get(s, [])
                    if not path_nodes:
                        if verbose:
                            print(f"[WARN] g={g}, OD({r}->{s}) 不可达，跳过。")
                        continue

                    # 转换为弧索引
                    try:
                        path_edges = self.get_path_edges(path_nodes)
                    except ValueError as e:
                        if verbose:
                            print(f"[WARN] 无法为 g={g}, r={r}, s={s} 添加路径: {e}")
                        continue

                    # 添加路径变量
                    added = self.add_path(g, r, s, path_edges, consider_order=consider_order, update_model=False)
                    if added:
                        added_count += 1
                        if verbose:
                            print(f"[add_path] 已为 g={g}, r={r}, s={s} 添加最短路径。")

        # --- 统一更新模型 ---
        self.m.update()

        if verbose:
            print(f"[INFO] 已完成所有最短路径添加，共新增 {added_count} 条。")

        return added_count

    def reset_given_groups_bounds(self, groups, verbose: bool = False):
        """
        将给定 groups 中出现的所有弧 (u,v) 对应的 y[a] 的上下界重置为 [0,1]。
        groups 格式示例：
            groups = [
                [(0,2),(2,11),(11,12)],
                [(3,10),(10,13),(13,22),(22,23)],
                ...
            ]
        每个 (u,v) 必须是 self.arcs 中存在的弧。

        :param groups: list[list[tuple[int,int]]]
        :param verbose: 是否打印被重置的弧索引
        """
        # 构建节点对 -> 弧索引的映射
        arc_dict = {arc: idx for idx, arc in enumerate(self.arcs)}
        reset_indices = set()

        # 遍历所有 group，收集弧索引
        for group in groups:
            for (u, v) in group:
                if (u, v) not in arc_dict:
                    raise ValueError(f"图中不存在弧 ({u},{v})，请检查 groups 输入。")
                reset_indices.add(arc_dict[(u, v)])

        # 对这些弧的 y 变量重置上下界为 [0,1]
        for a in reset_indices:
            self.y[a].LB = 0.0
            self.y[a].UB = 1.0

        self.m.update()

        if verbose:
            print(f"[INFO] 已重置 {len(reset_indices)} 条弧的 y 上下界为 [0,1]。")
            print("  弧索引列表:", sorted(reset_indices))

    def add_equal_y_groups_by_nodes(self, groups):
        """
        给定多个弧组，每组里的弧用 (u,v) 节点对表示。
        组内的所有弧对应的 y 必须完全相同。
        :param groups: list[list[tuple[int,int]]]
                       例如 [[(0,1),(1,0)], [(2,3),(3,4),(4,2)]]
        """
        arc_dict = {arc: idx for idx, arc in enumerate(self.arcs)}

        for g_idx, node_pairs in enumerate(groups):
            if len(node_pairs) < 2:
                continue  # 只有一个弧不需要约束

            # 把节点对转换为索引
            arc_indices = []
            for (u, v) in node_pairs:
                if (u, v) not in arc_dict:
                    raise ValueError(f"图中不存在弧 ({u},{v})")
                arc_indices.append(arc_dict[(u, v)])

            # 第一个为基准，组内其他与它相等
            base = arc_indices[0]
            for other in arc_indices[1:]:
                self.m.addConstr(self.y[base] - self.y[other] == 0,
                                 name=f"equal_y_nodes[group{g_idx},{base},{other}]")

        self.m.update()

    def evaluate_group_costs(self, groups):
        """
        计算每个 group 的建设成本。
        :param groups: list[list[tuple[int,int]]], 每个 group 是一组节点对
        :return: dict[group_idx -> cost]
        """
        arc_dict = {arc: idx for idx, arc in enumerate(self.arcs)}
        results = {}

        for g_idx, node_pairs in enumerate(groups):
            arc_indices = []
            for (u, v) in node_pairs:
                if (u, v) not in arc_dict:
                    raise ValueError(f"图中不存在弧 ({u},{v})")
                arc_indices.append(arc_dict[(u, v)])

            # 计算该 group 的成本
            cost = sum(self.L[a] for a in arc_indices)
            results[g_idx] = cost

        return results

    def compute_time_expression_value(self):
        """
        计算 ∑_{g,a} (T_a * x[g,a] - β_g * L_a * w[g,a]) 的数值（基于当前解）
        仅在模型求解完成后（self.m.optimize() 执行过）调用。
        """
        # 确保模型已优化完成
        if self.m.Status != GRB.OPTIMAL:
            print(self.m.Status)
            self.write_iis()
            raise RuntimeError("模型尚未求得最优解，无法计算时间表达式值。")

        total = 0.0
        for g in self.G:
            for a in range(self.A):
                total += self.T[a] * self.x[(g, a)].X - (1-self.beta[g]) * self.L[a] * self.w[(g, a)].X

        return total

    def evaluate_optimal_x_given_y(self, y: dict[int, int]) -> dict[tuple[int, int], float]:
        """
        在给定 y 的情况下，基于最短路计算各类别 g 下的最优弧流量 x[g,a]。
        返回: x_values[(g,a)] = 流量值
        """
        x_values = {(g, a): 0.0 for g in self.G for a in range(self.A)}

        # 校验 y 是否匹配弧数
        if len(y) != self.A:
            raise ValueError("y 的长度与弧数不一致")

        # 获取 y==1 的弧及原始权重
        one_arcs = [int(a) for a, val in y.items() if int(val) == 1]
        uvw0 = []
        for a in one_arcs:
            u, v = self.arcs[a]
            w0 = float(self.graph.orig_weights[u, v])
            uvw0.append((u, v, w0))

        for g in self.G:
            self.graph.reset_weights()
            # 对已建设弧使用 β_g 权重修正
            if uvw0:
                beta_g = float(self.beta[g])
                updates = [(u, v, w0 * beta_g) for (u, v, w0) in uvw0]
                self.graph.update_edges(updates)

            # 对每个 OD 对，找到最短路并分配需求
            for (r, s) in self.OD[g]:
                path_nodes, _ = self.graph.dijkstra_path(r, s)
                if not path_nodes:
                    continue  # 若不可达则跳过
                path_arc_indices = self.get_path_edges(path_nodes)
                d_rs = self.demand[(g, r, s)]
                for a in path_arc_indices:
                    x_values[(g, a)] += d_rs

        return x_values

    def add_time_constraint_given_x(self, x_given):
        """
        添加新的时间约束：
            Σ_g Σ_a [T_a*x[g,a] - (1-β_g)*L_a*w[g,a]]
            <= Σ_g Σ_a x_given[g,a]*(T_a - (1-β_g)*L_a*y[a])

        参数:
            x_given : dict[(g,a)] -> float
                给定的流量（可来自某次求解得到的 x）
        """
        # 左侧：变量形式（模型中的表达式）
        lhs_expr = gp.quicksum(
            self.L[a] * self.x[(g, a)] - (1 - self.beta[g]) * self.L[a] * self.w[(g, a)]
            for g in self.G for a in range(self.A)
        )

        # 右侧：根据给定 x 和当前 y 变量构建线性表达式
        rhs_expr = gp.quicksum(
            x_given.get((g, a), 0.0) * (self.L[a] - (1 - self.beta[g]) * self.L[a] * self.y[a])
            for g in self.G for a in range(self.A)
        )

        # 添加约束
        self.m.addConstr(lhs_expr <= rhs_expr+0.1, name="time_constraint_given_x")
        self.m.update()

        if self.verbose:
            print("[约束添加] Σ_g,a[T_a*x - (1-β_g)L_a*w] <= Σ_g,a x*(T_a-(1-β_g)L_a*y)")

    def write_iis(self, filename="model_iis.ilp"):
        """
        若模型不可行，则输出 IIS 文件（Irreducible Inconsistent Subsystem）。

        参数:
            filename: 输出文件名（默认 "model_iis.ilp"）
        """
        self.m.computeIIS()
        self.m.write(filename)
        print(f"[IIS 已生成] 模型不可行，IIS 已写入文件: {filename}")

        # 打印不可行约束的基本信息
        '''
        print("\n=== 不可行约束列表 ===")
        for constr in self.m.getConstrs():
            if constr.IISConstr:
                print(f"约束不可行: {constr.ConstrName}")
        for var in self.m.getVars():
            if var.IISLB:
                print(f"变量下界不可行: {var.VarName}")
            if var.IISUB:
                print(f"变量上界不可行: {var.VarName}")
        '''

    def count_paths(self, detail=False):
        total = len(self.h)
        if not detail:
            print(f"[INFO] 当前路径总数: {total}")
            return total
        # 分类别输出
        count_by_grs = {}
        for (g, r, s, p) in self.h:
            count_by_grs[(g, r, s)] = count_by_grs.get((g, r, s), 0) + 1
        print(f"[INFO] 当前路径总数: {total}")
        for key, cnt in count_by_grs.items():
            print(f"  g={key[0]}, r={key[1]}, s={key[2]} -> {cnt} 条路径")
        return total

    def clear_all_paths(self, *, keep_rhs: bool = True, verbose: bool = False):
        """
        清空所有已加入的路径：
          - 从模型中移除所有 h 变量（列）
          - 清空 self.h / self.seen_paths / self.path_counter
          - （可选）维持 demand 约束 RHS 不变（默认 True）

        注意：移除变量后，相关约束中的该列系数会被自动移除，
              不需要手动还原 x[g,a] - Σ h >= 0 的“壳”。
        """
        n_paths = len(self.h)
        if n_paths == 0:
            if verbose:
                print("[INFO] 没有路径可清空。")
            return

        # 1) 从模型中移除所有路径变量
        self.m.remove(list(self.h.values()))
        self.m.update()

        # 2) 清空内部记录
        self.h.clear()
        self.seen_paths.clear()
        self.path_counter.clear()

        # 3) 是否重置 demand 约束 RHS（通常保持不变即可）
        if not keep_rhs:
            for (g, r, s), cons in self.cons_demand.items():
                cons.RHS = float(self.demand[(g, r, s)])

        self.m.update()

        if verbose:
            print(f"[INFO] 已清空路径变量列：{n_paths} 个，当前路径总数=0。")
            print(f"[模型规模] Vars={self.m.NumVars:,}, Constrs={self.m.NumConstrs:,}, NonZeros={self.m.NumNZs:,}")

    def clear_paths_of(self, g, r, s, *, verbose: bool = True):
        """
        仅清空某个 (g,r,s) 对应的路径。
        """
        # 收集该 (g,r,s) 的 h 变量
        to_remove = []
        for (gg, rr, ss, p), hvar in list(self.h.items()):
            if gg == g and rr == r and ss == s:
                to_remove.append((gg, rr, ss, p, hvar))

        if not to_remove:
            if verbose:
                print(f"[INFO] (g={g}, r={r}, s={s}) 无路径可清空。")
            return

        # 移除变量
        self.m.remove([hvar for (_, _, _, _, hvar) in to_remove])
        self.m.update()

        # 清理字典与去重集合
        for (gg, rr, ss, p, _) in to_remove:
            self.h.pop((gg, rr, ss, p), None)
        # 去重集合：直接删掉该 OD 对
        self.seen_paths.pop((g, r, s), None)
        # 计数器置零
        self.path_counter[(g, r, s)] = 0

        self.m.update()

        if verbose:
            print(f"[INFO] 已清空 (g={g}, r={r}, s={s}) 的路径 {len(to_remove)} 个。")
            print(f"[模型规模] Vars={self.m.NumVars:,}, Constrs={self.m.NumConstrs:,}, NonZeros={self.m.NumNZs:,}")

    def save_path_snapshot(self):
        """
        保存当前路径相关状态的快照：
          - seen_paths
          - path_counter
          - h 变量列表（用于删除时参考）
        返回可用于 restore 的快照字典。
        """
        snapshot = {
            "seen_paths": {k: set(v) for k, v in self.seen_paths.items()},
            "path_counter": dict(self.path_counter),
            "h_keys": list(self.h.keys()),
        }
        print(f"[Snapshot] 已保存路径状态: 共 {len(snapshot['h_keys'])} 条路径。")
        return snapshot

    def restore_path_snapshot(self, snapshot, *, verbose=False):
        """
        根据 save_path_snapshot() 保存的状态，恢复模型路径。
        会自动移除快照后新增的路径变量，并恢复内部字典。
        """
        if not snapshot or "h_keys" not in snapshot:
            raise ValueError("无效的 snapshot 对象")

        # 当前路径集合
        current_h_keys = set(self.h.keys())
        target_h_keys = set(snapshot["h_keys"])

        # 需要移除的路径（新增的）
        extra_keys = current_h_keys - target_h_keys

        if not extra_keys:
            if verbose:
                print("[Snapshot] 无新增路径，无需恢复。")
            return

        # 移除变量列
        vars_to_remove = [self.h[k] for k in extra_keys if k in self.h]
        self.m.remove(vars_to_remove)
        self.m.update()

        # 从字典中清除这些路径
        for k in extra_keys:
            self.h.pop(k, None)

        # 恢复 seen_paths 和 path_counter
        self.seen_paths = {k: set(v) for k, v in snapshot["seen_paths"].items()}
        self.path_counter = dict(snapshot["path_counter"])

        self.m.update()

        if verbose:
            print(f"[Snapshot] 已恢复路径状态，删除 {len(extra_keys)} 条新路径。")
            print(f"[模型规模] Vars={self.m.NumVars:,}, Constrs={self.m.NumConstrs:,}, NonZeros={self.m.NumNZs:,}")

    def check_unique_active_path(self, tol=1e-6, verbose=False):
        """
        检查每个 (g, r, s) 是否仅有一个 h[g,r,s,p] 为非零。
        :param tol: 判断非零的阈值（默认 1e-6）
        :param verbose: 若为 True，则打印检查结果
        :return: (result_dict, false_list)
                 result_dict[(g,r,s)] -> bool
                 false_list -> 所有不满足条件的 (g,r,s)
        """
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError("模型尚未求得最优解，无法检查 h 的非零情况。")

        from collections import defaultdict
        counts = defaultdict(int)

        # 统计每个 (g,r,s) 中非零 h 的数量
        for (g, r, s, p), var in self.h.items():
            if abs(var.X) > tol:
                counts[(g, r, s)] += 1

        result = {}
        false_list = []
        all_grs = {(g, r, s) for (g, r, s, _) in self.h.keys()}

        for grs in all_grs:
            only_one = (counts.get(grs, 0) == 1)
            result[grs] = only_one
            if not only_one:
                false_list.append(grs)

            if verbose:
                print(f"(g={grs[0]}, r={grs[1]}, s={grs[2]}) -> "
                      f"{'✔️ 仅一个非零 h' if only_one else '❌ 多个或无非零 h'}")

        if verbose and false_list:
            print("\n⚠️ 以下 (g,r,s) 未满足“仅一个非零 h”的条件：")
            for grs in false_list:
                print(f"  - g={grs[0]}, r={grs[1]}, s={grs[2]}")

        return result, false_list


if __name__ == "__main__":
    # 小网络
    edges = [(0,1,2),(0,2,5),(1,2,1),(1,3,2),(2,3,3)]
    graph = SparseGraph(4, edges)

    G = [0]
    OD = {0:[(0,3)]}
    demand = {(0,0,3): 10.0}
    beta = {0:0.7}
    B = 7

    solver = HPRSolver(graph, B, G, OD, demand, beta, verbose=True)

    # 给定路径，用 arcs 的索引 (graph.arcs) 表示
    # arcs = [(0,1),(0,2),(1,2),(1,3),(2,3)]
    solver.add_path(0,0,3,[0,3])  # path1: (0->1->3)
    solver.add_path(0,0,3,[1,4])  # path2: (0->2->3)

    res = solver.solve()
    print("\n=== 结果 ===")
    print("Obj* =", res["obj_mip"])
    print("y* =", res["y_opt"])
    print("\n=== 对偶结果 ===")
    print("Flow-link 对偶值:")
    for (g, a), val in res["pi_flow_link"].items():
        print(f"pi_flow_link[{g},{a}] = {val}")

    print("\nDemand 对偶值:")
    for (g, r, s), val in res["pi_demand"].items():
        print(f"pi_demand[{g},{r},{s}] = {val}")