from read_ import *
from collections import defaultdict, deque
from B_and_B import *
import json
import time
import csv
import os
from datetime import datetime
import itertools

def append_to_table(file_path: str, headers: list[str], row: dict):
    """
    在 CSV 文件中追加一行记录；若文件不存在则自动创建并写入表头。

    参数:
        file_path : str
            CSV 文件路径（如 'results.csv'）
        headers : list[str]
            表头字段名（列名）
        row : dict
            要追加的数据（键为列名，值为内容）
            未在 headers 中的键将被忽略，缺失的列自动填空字符串

    示例:
        append_to_table(
            'results.csv',
            headers=['time', 'seed', 'n', 'k', 'mean_cost'],
            row={'time': datetime.now().isoformat(), 'seed': 42, 'n': 5, 'k': 2, 'mean_cost': 3.14}
        )
    """
    # 确认文件是否存在
    file_exists = os.path.exists(file_path)

    # 若文件不存在，先创建并写表头
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        # 确保只写入表头定义中的字段
        filtered_row = {h: row.get(h, "") for h in headers}
        writer.writerow(filtered_row)

def generate_OD_and_demand(demand_matrix: np.ndarray, G: list, P: list):
    """
    根据需求矩阵和比例向量 P 生成 OD 和 demand 字典

    :param demand_matrix: numpy 2D array, demand_matrix[r,s] 表示从 r 到 s 的总需求
    :param G: 类别集合 (list 或 set)，例如 [0,1,2]
    :param P: 每个类别的分配比例 (list)，正数且和为 1
    :return: (OD, demand)
             OD[g] = [(r,s), ...]
             demand[(g,r,s)] = float
    """
    # 检查 P 的合法性
    P = np.array(P, dtype=float)
    if len(P) != len(G):
        raise ValueError("P 的长度必须和 G 的类别数相同")
    if not np.all(P >= 0):
        raise ValueError("P 必须为非负数")
    if not np.isclose(P.sum(), 1.0):
        raise ValueError("P 的和必须等于 1")

    n = demand_matrix.shape[0]
    OD = {g: [] for g in G}
    demand = {}

    for r in range(n):
        for s in range(n):
            if demand_matrix[r, s] > 0 and r != s:  # 忽略 0 需求和自环
                total_d = demand_matrix[r, s]
                for g_idx, g in enumerate(G):
                    d_g = total_d * P[g_idx]
                    if d_g > 0:
                        OD[g].append((r, s))
                        demand[(g, r, s)] = d_g

    return OD, demand

def complete_groups_with_all_arcs(solver, groups):
    """
    给定已有 groups，补充所有未出现的弧，
    每条孤立的弧单独成一个 group。

    :param solver: HPRSolver 实例
    :param groups: list[list[tuple[int,int]]]
    :return: list[list[tuple[int,int]]]  补全后的 groups
    """
    # 先把已有 groups 里包含的所有弧收集起来
    covered = set()
    for group in groups:
        for (u, v) in group:
            covered.add((u, v))

    # 遍历 solver 里的所有弧
    all_arcs = solver.arcs
    for (u, v) in all_arcs:
        if (u, v) not in covered:
            groups.append([(u, v)])  # 单独成一个 group

    return groups

def add_reverse_arcs_to_groups(solver, groups):
    """
    对每个 group，如果其中有弧 (u,v)，并且图中存在反向弧 (v,u)，
    则将 (v,u) 也加入该 group。

    :param solver: HPRSolver 实例
    :param groups: list[list[tuple[int,int]]]
    :return: list[list[tuple[int,int]]]  扩展后的 groups
    """
    arc_set = set(solver.arcs)  # 所有弧的集合，方便查找
    new_groups = []

    for group in groups:
        extended = set(group)  # 用集合避免重复
        for (u, v) in group:
            if (v, u) in arc_set:
                extended.add((v, u))
        new_groups.append(list(extended))

    return new_groups

def compute_all_zero_value_cut(solver):
    """
    计算所有 y=0（即无任何弧被建设）的 value_cut 值。

    参数:
        solver: HPRSolver 实例

    返回:
        value_cut: float，总出行成本
    """
    # 1️⃣ 构造 y=0 的方案
    y_zero = {a: 0 for a in range(solver.A)}

    # 2️⃣ 调用 solver 的评估函数
    value_cut = solver.evaluate_total_cost_given_y(y_zero)

    # 3️⃣ 输出结果
    print(f"[INFO] 所有 y=0 时的 value_cut = {value_cut:.6f}")

    return value_cut

def complete_and_add_reverse_groups(solver, groups):
    """
    一次性完成以下两步：
    1️⃣ 先补全所有未出现的弧（但跳过那些已有反向弧已被包含的情况）
    2️⃣ 再让每个 group 中的弧自动添加它的反向弧（若存在于图中）

    :param solver: HPRSolver 实例
    :param groups: list[list[tuple[int,int]]]
    :return: list[list[tuple[int,int]]]  完整补全 + 反向弧扩展后的 groups
    """

    # === 第一步：补全所有未出现的弧 ===
    covered = set()
    for group in groups:
        for (u, v) in group:
            covered.add((u, v))

    all_arcs = solver.arcs
    all_arcs_set = set(all_arcs)

    for (u, v) in all_arcs:
        # 若此弧和它的反向弧都未出现在已有组中，则新建组
        if (u, v) not in covered and (v, u) not in covered:
            groups.append([(u, v)])
            covered.add((u, v))

    # === 第二步：为每个 group 添加反向弧 ===
    new_groups = []
    for group in groups:
        extended = set(group)
        for (u, v) in group:
            if (v, u) in all_arcs_set:
                extended.add((v, u))
        new_groups.append(list(extended))

    return new_groups

def compute_all_zero_value_cut(solver):
    """
    计算所有 y=0（即无任何弧被建设）的 value_cut 值。

    参数:
        solver: HPRSolver 实例

    返回:
        value_cut: float，总出行成本
    """
    # 1️⃣ 构造 y=0 的方案
    y_zero = {a: 0 for a in range(solver.A)}

    # 2️⃣ 调用 solver 的评估函数
    value_cut = solver.evaluate_total_cost_given_y(y_zero)

    return value_cut

def group_edges_connected_balanced(graph: SparseGraph, n: int):
    """
    将连通图的所有弧划分为 n 个连通组，满足：
    1. 相反方向弧在同一组；
    2. 每组内部连通；
    3. 所有弧都分配；
    4. 各组成本尽量相近。
    """
    # === 1️⃣ 生成弧单元 ===
    edges = []
    for eid, e in enumerate(graph.g.es):
        u, v = e.tuple
        w = float(e["weight"])
        edges.append(((u, v), w))

    pair_map = defaultdict(list)
    for (u, v), w in edges:
        key = (min(u, v), max(u, v))
        pair_map[key].append(((u, v), w))

    units = []
    for key, arcs in pair_map.items():
        cost = sum(w for _, w in arcs)
        nodes = set(sum(([u, v] for (u, v), _ in arcs), []))
        units.append({
            "arcs": [a for a, _ in arcs],
            "cost": cost,
            "nodes": nodes
        })

    # === 2️⃣ 构建单元邻接关系（共享节点的单元相连） ===
    node_to_units = defaultdict(list)
    for i, unit in enumerate(units):
        for node in unit["nodes"]:
            node_to_units[node].append(i)

    adj = {i: set() for i in range(len(units))}
    for idxs in node_to_units.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i].add(j)

    # === 3️⃣ 计算总成本 & 目标均衡值 ===
    total_cost = sum(u["cost"] for u in units)
    target_cost = total_cost / n

    # === 4️⃣ 初始化 ===
    unassigned = set(range(len(units)))
    groups = [[] for _ in range(n)]
    group_costs = [0.0] * n

    # === 5️⃣ BFS 连通分组（尽量平衡） ===
    for g_idx in range(n):
        if not unassigned:
            break

        # 从未分配中选当前最大权重的单元作为起点
        start = max(unassigned, key=lambda i: units[i]["cost"])
        queue = deque([start])
        visited = set([start])
        current_cost = units[start]["cost"]

        while queue and current_cost < target_cost * 0.9:  # 允许一点松弛
            u = queue.popleft()
            # 遍历邻居单元
            for v in adj[u]:
                if v in unassigned and v not in visited:
                    queue.append(v)
                    visited.add(v)
                    current_cost += units[v]["cost"]
                    if current_cost >= target_cost:
                        break

        # 分配这一组
        for idx in visited:
            if idx in unassigned:
                unassigned.remove(idx)
        groups[g_idx] = [arc for i in visited for arc in units[i]["arcs"]]
        group_costs[g_idx] = sum(units[i]["cost"] for i in visited)

    # === 6️⃣ 若有剩余单元，加入最小成本组 ===
    for idx in list(unassigned):
        min_g = min(range(n), key=lambda i: group_costs[i])
        groups[min_g].extend(units[idx]["arcs"])
        group_costs[min_g] += units[idx]["cost"]
        unassigned.remove(idx)

    return groups, group_costs

def group_edges_by_budget_with_noise(graph: 'SparseGraph', B: float, n: int,
                                     seed: int = None, verbose: bool = True):
    """
    将图的弧划分为 n 个连通组，使得：
      1. 相反方向弧在同一组；
      2. 每组内部连通；
      3. 每组成本 ≤ B + k，k ~ N(0, (B/5)^2)，且保证 B+k ≥ B/2；
      4. 尽量平衡，使每组成本接近 B；
      5. 不要求覆盖所有弧（剩余的直接忽略）。

    参数:
        graph : SparseGraph
        B : float                # 目标预算上限
        n : int                  # 希望生成的组数
        seed : int, optional     # 随机种子（可复现）
        verbose : bool           # 是否打印日志

    返回:
        groups : list[list[tuple[int,int]]]
        group_costs : list[float]
    """
    if seed is not None:
        np.random.seed(seed)

    # === 1️⃣ 提取弧信息 ===
    edges = []
    for e in graph.g.es:
        u, v = e.tuple
        w = float(e["weight"])
        edges.append(((u, v), w))

    # === 2️⃣ 相反方向弧合并成单元 ===
    pair_map = defaultdict(list)
    for (u, v), w in edges:
        key = (min(u, v), max(u, v))
        pair_map[key].append(((u, v), w))

    units = []
    for key, arcs in pair_map.items():
        cost = sum(w for _, w in arcs)
        nodes = set(sum(([u, v] for (u, v), _ in arcs), []))
        units.append({
            "arcs": [a for a, _ in arcs],
            "cost": cost,
            "nodes": nodes
        })

    # === 3️⃣ 构建单元间邻接 ===
    node_to_units = defaultdict(list)
    for i, unit in enumerate(units):
        for node in unit["nodes"]:
            node_to_units[node].append(i)

    adj = {i: set() for i in range(len(units))}
    for idxs in node_to_units.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i].add(j)

    # === 4️⃣ 初始化 ===
    unassigned = set(range(len(units)))
    groups, group_costs = [], []
    n = min(n, len(units))

    # === 5️⃣ 主循环：生成 n 个组 ===
    for g_idx in range(n):
        if not unassigned:
            break

        # 预算上限带随机波动
        k = np.random.normal(0, B / 5)
        if B + k < B / 2:
            k = -B / 2
        budget_limit = B + k

        # 从未分配中选起点（优先高成本）
        start = max(unassigned, key=lambda i: units[i]["cost"])
        queue = deque([start])
        visited = set([start])
        current_cost = units[start]["cost"]

        # BFS 扩展直到接近预算上限
        while queue and current_cost < budget_limit * 0.95:
            u = queue.popleft()
            for v in adj[u]:
                if v in unassigned and v not in visited:
                    new_cost = current_cost + units[v]["cost"]
                    if new_cost <= budget_limit:
                        queue.append(v)
                        visited.add(v)
                        current_cost = new_cost
                    if current_cost >= budget_limit:
                        break

        # 分配本组
        for idx in visited:
            unassigned.discard(idx)
        group_arcs = [a for i in visited for a in units[i]["arcs"]]
        groups.append(group_arcs)
        group_costs.append(current_cost)

    if verbose:
        print(f"[INFO] 成功生成 {len(groups)} 个组 (目标 {n})")
        print(f"[INFO] 成本均值: {np.mean(group_costs):.3f}, 最大: {max(group_costs):.3f}, 最小: {min(group_costs):.3f}")
        print(f"[INFO] 未分配单元数量: {len(unassigned)}")

    return groups, group_costs

import numpy as np
from collections import defaultdict, deque

def group_edges_connected_fixed(graph: 'SparseGraph', n: int, k: int, seed: int = None, verbose: bool = True):
    """
    随机将图的弧划分为 n 个互不相交且连通的组，
    每组 **必须** 含有恰好 k 个“正反路径对单元”。
    若无法形成满足条件的划分则抛出 ValueError。

    参数:
        graph : SparseGraph
            输入稀疏图对象
        n : int
            希望生成的组数
        k : int
            每组的单元数量（必须相等）
        seed : int, optional
            随机种子（可复现）
        verbose : bool
            是否打印日志信息

    返回:
        groups : list[list[tuple[int,int]]]
            每组包含的有向弧 (u,v)
        group_costs : list[float]
            每组的总权重
    """
    if seed is not None:
        np.random.seed(seed)

    # === 1️⃣ 提取弧信息 ===
    edges = []
    for e in graph.g.es:
        u, v = e.tuple
        w = float(e["weight"])
        edges.append(((u, v), w))

    # === 2️⃣ 合并正反弧成单元 ===
    pair_map = defaultdict(list)
    for (u, v), w in edges:
        key = (min(u, v), max(u, v))
        pair_map[key].append(((u, v), w))

    units = []
    for arcs in pair_map.values():
        cost = sum(w for _, w in arcs)
        nodes = set(sum(([u, v] for (u, v), _ in arcs), []))
        units.append({
            "arcs": [a for a, _ in arcs],
            "cost": cost,
            "nodes": nodes
        })

    m = len(units)
    if m < n * k:
        raise ValueError(f"无法划分：总单元数 {m} < n*k = {n*k}")

    # === 3️⃣ 构建单元间邻接图 ===
    node_to_units = defaultdict(list)
    for i, unit in enumerate(units):
        for node in unit["nodes"]:
            node_to_units[node].append(i)

    adj = {i: set() for i in range(m)}
    for idxs in node_to_units.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i].add(j)

    # === 4️⃣ 初始化 ===
    unassigned = set(range(m))
    groups, group_costs = [], []

    # === 5️⃣ 逐组构建 ===
    for g_idx in range(n):
        if len(unassigned) < k:
            raise ValueError(f"剩余单元 {len(unassigned)} 不足以再形成一个包含 {k} 单元的组")

        # 选一个未分配单元作为起点（优先高成本）
        start = max(unassigned, key=lambda i: units[i]["cost"])
        queue = deque([start])
        visited = {start}
        current_cost = units[start]["cost"]

        # BFS 扩展保持连通性直到正好 k 个单元
        while queue and len(visited) < k:
            u = queue.popleft()
            neighbors = list(adj[u])
            np.random.shuffle(neighbors)
            for v in neighbors:
                if v in unassigned and v not in visited:
                    visited.add(v)
                    queue.append(v)
                    current_cost += units[v]["cost"]
                    if len(visited) == k:
                        break

        if len(visited) != k:
            raise ValueError(f"第 {g_idx+1} 组无法找到恰好 {k} 个连通单元 (仅找到 {len(visited)})")

        # 分配该组
        for idx in visited:
            unassigned.discard(idx)

        group_arcs = [a for i in visited for a in units[i]["arcs"]]
        groups.append(group_arcs)
        group_costs.append(current_cost)

    # === ✅ 成功 ===
    if verbose:
        print(f"[INFO] 成功生成 {len(groups)} 个连通组，每组恰好 {k} 个单元。")
        print(f"[INFO] 总单元数: {m}, 已使用: {n*k}, 剩余: {len(unassigned)}")
        print(f"[INFO] 成本均值: {np.mean(group_costs):.3f}, 最大: {max(group_costs):.3f}, 最小: {min(group_costs):.3f}")

    return groups, group_costs

import os
import csv
import copy
import time
import json
import numpy as np
from datetime import datetime

# ===== 你已有的 append_to_table 函数 =====
def append_to_table(file_path: str, headers: list[str], row: dict):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        filtered_row = {h: row.get(h, "") for h in headers}
        writer.writerow(filtered_row)


# ===== 实验主函数 =====
def run_experiment(network: int, G, beta, P, n, k, B: float,value_cut, seed=10100):
    """
    运行一次交通网络设计实验并保存结果。

    参数:
        network : int
            1 = Sioux Falls, 2 = Anaheim, 3 = Eastern Massachusetts
        G : list[int]
            固定节点集合
        beta : list[float]
            弹性参数
        P : list[int]
            OD选择参数
        n : int
            分组数量
        k : int
            每组边数或路径长度
        B : float
            预算系数（0~1 或者自定义数值）
        seed : int, default=10100
            随机种子
    """
    start_time = time.time()
    if value_cut == 1:
        value_cut = True
    elif value_cut == 0:
        value_cut = False
    # 1️⃣ 选择网络
    if network == 1:
        edges, demand = read_sioux_falls(
            r"SiouxFalls/SiouxFalls_net.tntp",
            r"SiouxFalls/SiouxFalls_trips.tntp"
        )
        net_name = "SF"
    elif network == 2:
        edges, demand = read_sioux_falls(
            r"Anaheim/Anaheim_net.tntp",
            r"Anaheim/Anaheim_trips.tntp"
        )
        net_name = "Anaheim"
    elif network == 3:
        edges, demand = read_sioux_falls(
            r"Eastern-Massachusetts/EMA_net.tntp",
            r"Eastern-Massachusetts/EMA_trips.tntp"
        )
        net_name = "EMA"
    elif network == 4:
        edges, demand = read_sioux_falls(r"Berlin-Mitte-Center/Berlin-Mitte-Center_net.tntp",
                                         r"Berlin-Mitte-Center/Berlin-Mitte-Center_trips.tntp")
        net_name = "BMC"
    else:
        raise ValueError("network 参数必须是 1, 2, 或 3。")

    # 2️⃣ 初始化参数
    graph = SparseGraph(demand.shape[0], edges)
    #total_cost = graph.total_weight_sum()

    B_initial = copy.deepcopy(B)
    OD, demand = generate_OD_and_demand(demand, G, P)

    # 生成边分组
    try:
        groups, cost = group_edges_connected_fixed(graph, n, k, seed=seed, verbose=False)
    except:
        return

    # 实际预算 = B 系数 * 总成本
    B = B * np.sum(cost)
    print(groups)

    solver = HPRSolver(graph, B, G, OD, demand, beta, verbose=False, time_ub=None)
    solver.add_equal_y_groups_by_nodes(groups)
    solver.fix_outside_groups_to_zero(groups)

    Opt_solution, Opt_value, tt1, tt2, tt3, tt4, N, UB, LB, gap = branch_and_bound(
        solver, OD, demand, beta, groups,
        output=False, value_cut_open=value_cut
    )
    print(Opt_solution.set)
    print(Opt_solution.forbidden)
    print(Opt_value)

    # gap修正
    if gap > 1 or gap < 0:
        LB = UB
        gap = 0

    # 4️⃣ 保存结果
    data_to_save = {
        "network": net_name,
        "B": B_initial,
        "seed": seed,
        "g": len(G),
        "groups": n,
        "length": k,
        'value_cut': value_cut,
        "Opt": Opt_value,
        "iter": N,
        "T_all": time.time() - start_time,
        "T_Hpr": tt1,
        "T_Short_path": tt2,
        'T_Update': tt3,
        'T_value_cut':tt4,
        "UB": UB,
        "LB": LB,
        "gap": gap
    }

    headers = [
        "network", "B", "seed", "g", "groups", "length",'value_cut',"Opt",  "iter",
        "T_all", "T_Hpr", "T_Short_path",'T_Update','T_value_cut', "UB", "LB", "gap"
    ]
    print(data_to_save)
    append_to_table('results.csv', headers=headers, row=data_to_save)

# ===== 实验主函数 =====
def run_experiment2(network: int, G, beta, P, n, k, B: float,value_cut, seed=10100):
    """
    运行一次交通网络设计实验并保存结果。

    参数:
        network : int
            1 = Sioux Falls, 2 = Anaheim, 3 = Eastern Massachusetts
        G : list[int]
            固定节点集合
        beta : list[float]
            弹性参数
        P : list[int]
            OD选择参数
        n : int
            分组数量
        k : int
            每组边数或路径长度
        B : float
            预算系数（0~1 或者自定义数值）
        seed : int, default=10100
            随机种子
    """
    start_time = time.time()
    if value_cut == 1:
        value_cut = True
    elif value_cut == 0:
        value_cut = False
    # 1️⃣ 选择网络
    if network == 1:
        edges, demand = read_sioux_falls(
            r"SiouxFalls/SiouxFalls_net.tntp",
            r"SiouxFalls/SiouxFalls_trips.tntp"
        )
        net_name = "SF"
    elif network == 2:
        edges, demand = read_sioux_falls(
            r"Anaheim/Anaheim_net.tntp",
            r"Anaheim/Anaheim_trips.tntp"
        )
        net_name = "Anaheim"
    elif network == 3:
        edges, demand = read_sioux_falls(
            r"Eastern-Massachusetts/EMA_net.tntp",
            r"Eastern-Massachusetts/EMA_trips.tntp"
        )
        net_name = "EMA"
    elif network == 4:
        edges, demand = read_sioux_falls(r"Berlin-Mitte-Center/Berlin-Mitte-Center_net.tntp",
                                         r"Berlin-Mitte-Center/Berlin-Mitte-Center_trips.tntp")
        net_name = "BMC"
    else:
        raise ValueError("network 参数必须是 1, 2, 或 3。")

    # 2️⃣ 初始化参数
    graph = SparseGraph(demand.shape[0], edges)
    #total_cost = graph.total_weight_sum()

    B_initial = copy.deepcopy(B)
    OD, demand = generate_OD_and_demand(demand, G, P)

    # 生成边分组
    try:
        groups, cost = group_edges_connected_fixed(graph, n, k, seed=seed, verbose=False)
    except:
        return

    # 实际预算 = B 系数 * 总成本
    B = B * np.sum(cost)
    print(groups)

    solver = HPRSolver(graph, B, G, OD, demand, beta, verbose=False, time_ub=None)
    solver.add_equal_y_groups_by_nodes(groups)
    solver.fix_outside_groups_to_zero(groups)

    Opt_solution, Opt_value, tt1, tt2, tt3, tt4, N, UB, LB, gap,value_cut_number = branch_and_bound(
        solver, OD, demand, beta, groups,
        output=False, value_cut_open=value_cut
    )
    print(Opt_solution.set)
    print(Opt_solution.forbidden)
    print(Opt_value)

    # gap修正
    if gap > 1 or gap < 0:
        LB = UB
        gap = 0

    # 4️⃣ 保存结果
    data_to_save = {
        "network": net_name,
        "B": B_initial,
        "seed": seed,
        "g": len(G),
        "groups": n,
        "length": k,
        'value_cut': value_cut,
        "Opt": Opt_value,
        "iter": N,
        "T_all": time.time() - start_time,
        "T_Hpr": tt1,
        "T_Short_path": tt2,
        'T_Update': tt3,
        'T_value_cut':tt4,
        "UB": UB,
        "LB": LB,
        "gap": gap,
        "value_cut_number": value_cut_number
    }

    headers = [
        "network", "B", "seed", "g", "groups", "length",'value_cut',"Opt",  "iter",
        "T_all", "T_Hpr", "T_Short_path",'T_Update','T_value_cut', "UB", "LB", "gap","value_cut_number"
    ]
    print(value_cut_number)
    print(data_to_save)
    append_to_table('results.csv', headers=headers, row=data_to_save)

if __name__ == '__main__':
    network = [1,2,3]
    B = [0.5]
    n = [20]
    k = [1]
    G = [1]
    value_cut = [0]
    param_list = list(itertools.product(network, B, n, k, G, value_cut))
    total = len(param_list)
    for i, (network1, B1, n1, k1, G1, value_cut1) in enumerate(param_list, 1):
        print(f"=== 运行 {i}/{total} : net={network1}, B={B1}, n={n1}, k={k1}, G={G1}, cut={value_cut1} ===")
        P = [1 / G1] * G1
        Beta = np.linspace(0, 1, G1 + 2)[1:-1]
        G2 = list(range(G1))
        run_experiment2(network1, G2, Beta, P, n1, k1, B1, value_cut1, seed=10100)