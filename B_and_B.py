from collections import deque
import time
from UB_LB import *
import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, candidate):
        self.budget = 0
        self.candidate = candidate
        self.forbidden = []
        self.set = []
        self.LB = 0
        self.y = {i: 0 for i in range(len(candidate))}
        # 其他需要的属性可以再补充

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
    #print(f"[INFO] 所有 y=0 时的 value_cut = {value_cut:.6f}")

    return value_cut

def split_arcs_by_status(dist):
    """
    将字典 dist 拆分为两个列表：
    - built_list: value == 1 的键
    - unbuilt_list: value == 0 的键

    参数:
        dist (dict[int,int]): 弧状态字典，key 是弧索引，value 为 0 或 1

    返回:
        tuple[list[int], list[int]]: (built_list, unbuilt_list)
    """
    built_list = [a for a, v in dist.items() if int(v) == 1]
    unbuilt_list = [a for a, v in dist.items() if int(v) == 0]
    return built_list, unbuilt_list


def check_budget_conditions(node: Node, weight, total_budget: float):
    """
    检查节点在给定总预算下的情况:
    (0) 两个条件都不满足
    (1) 剩余预算 >= candidate 全部元素权重之和 + forbidden 中的最小权重
    (2) 剩余预算 < candidate 中的最小权重
    (3) candidate总和 < 剩余预算 < candidate总和 + forbidden最小

    参数:
        node: Node 对象
        weight: list[float]，每个 group 的权重
        total_budget: float，总预算

    返回:
        int，0 / 1 / 2 / 3
    """
    remaining_budget = total_budget - node.budget

    if node.candidate:
        sum_cand = sum(weight[c] for c in node.candidate)
        min_cand = min(weight[c] for c in node.candidate)
        if remaining_budget >= sum_cand:
            return 0
        if node.forbidden:
            min_forbid = min(weight[f] for f in node.forbidden)
            # (1) 大预算，能覆盖 candidate + forbid最小
            if remaining_budget >= sum_cand + min_forbid:
                return 0
            # (3) 介于 candidate总和 和 candidate总和+forbid最小之间
            if sum_cand < remaining_budget < sum_cand + min_forbid:
                return 0
        # (2) 小预算，比 candidate最小还小
        if remaining_budget < min_cand:
            return 2

    return 0


def build_y(result: int, node: Node):
    """
    根据 result 和 node 构建 y 列表:
      - list[0]: value=1 的索引
      - list[1]: value=0 的索引
      - result 为 0 或 1 时返回 False
    """
    # 如果是 0 或 1，直接返回 False
    if result == 0:
        return False

    y = [[], []]  # y[0] -> 1 的集合, y[1] -> 0 的集合

    # set → 1
    y[0].extend(node.set)

    # forbidden → 0
    y[1].extend(node.forbidden)

    # candidate → 根据 result 赋值
    if result == 2:
        y[1].extend(node.candidate)
    elif result == 3 or result == 1:
        y[0].extend(node.candidate)
    return y

def fill_missing_y_with_zero(y: dict[int, int], solver: 'HPRSolver', verbose: bool = True):
    """
    给定部分弧的 y 字典（key=弧索引, value=0或1），
    将 solver 中所有未出现的弧自动补充为 0。

    参数：
        y : dict[int, int]
            现有的 y 字典，部分弧已给出 0/1。
        solver : HPRSolver
            HPRSolver 实例，包含全部弧 self.A 和变量 self.y。
        verbose : bool
            是否打印补充信息。

    返回：
        full_y : dict[int, int]
            完整的 y 字典（包含全部弧，缺失的补为 0）
    """
    A = solver.A  # 总弧数
    full_y = dict(y)  # 拷贝

    # 统计缺失弧
    missing = [a for a in range(A) if a not in y]

    if verbose:
        print(f"[INFO] 输入 y 含 {len(y)} 条弧，模型共有 {A} 条弧。")
        print(f"[INFO] 自动补齐 {len(missing)} 条缺失弧为 0。")

    # 1️⃣ 补全字典
    for a in missing:
        full_y[a] = 0

    # 2️⃣ 在 solver 模型中强制这些弧 y[a] = 0
    for a in missing:
        solver.y[a].LB = 0
        solver.y[a].UB = 0

    solver.m.update()

    if verbose and missing:
        print(f"[INFO] 已将 {len(missing)} 条缺失弧的 y 变量固定为 0。")

    return full_y


def get_y_from_budget(node: Node, weight, total_budget: float):
    """
    输入 node 和 total_budget，直接返回 y (dict) 或 False

    参数:
        node: Node 对象
        weight: list[float]，每个 group 的权重
        total_budget: float，总预算

    返回:
        dict[int,int] 或 False
    """
    # 先计算 result
    result = check_budget_conditions(node, weight, total_budget)

    # 再构造 y
    y = build_y(result, node)

    return y


def pop_min_LB(Q):
    if not Q:
        return None

    # 找到最小 LB 的节点
    min_node = min(Q, key=lambda node: node.LB)

    # 从 deque 中移除
    Q.remove(min_node)

    return min_node


def get_min_LB(Q):
    """
    输入: Q (deque[Node])
    输出: 最小的 LB 值，如果 Q 为空则返回 None
    """
    if not Q:
        return None
    return min(node.LB for node in Q)

def is_feasible(node: Node) -> bool:
    # 判断节点是否是可行解
    pass

def generate_children(node: Node, weight):
    # 在 candidate 中筛选出 y > 0 的元素
    valid_candidates = [c for c in node.candidate if node.y.get(c, float("inf")) > 0]

    if valid_candidates:
        # 在 valid_candidates 中找到 y 值最小的元素
        elem = min(valid_candidates, key=lambda x: node.y.get(x, float("inf")))
    else:
        # 如果没有符合条件的，直接选择 candidate 的第一个元素
        elem = node.candidate[0]

    # 去掉这个元素后的新候选集
    new_candidates = [c for c in node.candidate if c != elem]

    # forbidden 分支：不选择 elem
    child_forbidden = Node(new_candidates)
    child_forbidden.budget = node.budget
    child_forbidden.forbidden = node.forbidden + [elem]
    child_forbidden.set = node.set.copy()
    child_forbidden.y = node.y.copy()  # y 需要继承

    # set 分支：选择 elem
    child_set = Node(new_candidates)
    child_set.budget = node.budget + weight[elem]
    child_set.forbidden = node.forbidden.copy()
    child_set.set = node.set + [elem]
    child_set.y = node.y.copy()  # y 需要继承

    return [child_forbidden, child_set]



def update_bounds(solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node):
    """
    根据给定解 y 更新上界和最优解。

    参数:
        solver: 求解器对象，需实现 evaluate_total_cost_given_y, evaluate_unbuilt_cost, et_time_upper_bound 等方法
        y: 当前解
        value_cut: 当前的截断值 (upper bound cut)
        UB_star: 当前的 UB_star
        Opt_value: 当前的最优目标值
        Opt_solution: 当前的最优解
        child_node: 当前子节点（候选解）

    返回:
        value_cut, UB_star, Opt_value, Opt_solution
    """
    # 更新 value_cut 并同步 time_ub
    #relax_UB = solver.compute_time_expression_value()
    #y = fill_missing_y_with_zero(y, solver, verbose = False)
    #True_UB = solver.evaluate_total_cost_given_y(y)

    #print(True_UB)
    #print(relax_UB)
    #print(True_UB - relax_UB)
    # 更新 UB_star
    #print(f'True_Total_time:{True_UB},HPR_total_time:{relax_UB},Gap{(relax_UB-True_UB)/True_UB}')

    # 更新最优解
    current_cost = solver. evaluate_unbuilt_cost_1toall_min_unbuilt_on_shortest(y)
    #current_cost = solver.evaluate_unbuilt_cost(y)
    UB_star = np.minimum(current_cost, UB_star)

    if current_cost <= Opt_value:
        Opt_value = current_cost
        Opt_solution = child_node

    return value_cut, UB_star, Opt_value, Opt_solution,solver

def get_all_group_arc_indices(solver, groups):
    """
    从 groups 中收集所有弧的索引，返回一个不重复的列表。
    :param solver: HPRSolver 实例
    :param groups: list[list[tuple[int,int]]]
    :return: list[int] 弧索引
    """
    arc_dict = {arc: idx for idx, arc in enumerate(solver.arcs)}
    arc_indices = set()

    for group in groups:
        for (u, v) in group:
            if (u, v) not in arc_dict:
                raise ValueError(f"图中不存在弧 ({u},{v})")
            arc_indices.add(arc_dict[(u, v)])

    return sorted(list(arc_indices))

def select_y2_with_budget(solver, tol=1e-6):
    """
    从 HPRSolver 求解结果中，找到 w==x 的弧及其 y 值，
    构造子集 S' 使得 sum(T_a)>=B 且刚刚超过，最大化 sum(y_a*L_a)。

    参数：
        solver : HPRSolver 已求解实例
        tol     : float, 判断 w==x 的容差

    返回：
        result : dict，包含以下字段
            'selected_indices' : list[int]，选中的弧索引 S'
            'total_L_value'    : float，∑ y_a * L_a （最大化结果）
            'total_T_value'    : float，选中弧的 ∑ T_a
            'y2'               : dict[int,int]，选中弧=1，其余=0
    """
    # 1️⃣ 检查模型状态
    if solver.m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError("模型尚未最优求解，无法执行此函数")

    # 2️⃣ 找出满足 w≈x 的弧
    candidate_arcs = []
    for a in range(solver.A):
        w_val = sum(solver.w[(g, a)].X for g in solver.G)
        x_val = sum(solver.x[(g, a)].X for g in solver.G)
        y_val = solver.y[a].X
        if abs(w_val - x_val) < tol and y_val > tol:
            candidate_arcs.append((a, y_val))

    if not candidate_arcs:
        print("[INFO] 没有找到 w≈x 且 y>0 的弧。")
        return None

    # 3️⃣ 收集对应的 L, T
    arc_info = []
    for a, y_val in candidate_arcs:
        arc_info.append({
            "a": a,
            "y": y_val,
            "L": solver.L[a],
            "T": solver.T[a]
        })

    # 4️⃣ 按照 “L/T 比值” 从大到小排序
    arc_info.sort(key=lambda d: d['y'], reverse=True)

    # 5️⃣ 逐个选入直到 sum(T)>=B
    total_T = 0.0
    selected = []
    for arc in arc_info:
        selected.append(arc)
        total_T += arc["T"]
        if total_T >= solver._init_args[1]- 1e-6:
            break

    if total_T <= solver._init_args[1] - 1e-6:
        return False

    # 6️⃣ 构造 y2
    y2 = {a: 0 for a in range(solver.A)}
    for arc in selected:
        y2[arc["a"]] = 1

    # 7️⃣ 计算目标值
    total_L_val = sum(arc["y"] * arc["L"] for arc in selected)
    total_T_val = sum(arc["T"] for arc in selected)

    result = {
        "selected_indices": [arc["a"] for arc in selected],
        "total_L_value": total_L_val,
        "total_T_value": total_T_val,
        "y2": y2
    }
    return result

def adaptive_budget_iteration(solver,
                              threshold=0.00001,
                              max_iter=2,
                              init_path=None,
                              verbose=False):
    """
    自适应预算循环：
    - 求解 HPR；
    - 调用 select_y2_with_budget() 生成 y2；
    - 更新预算 B；
    - 重复直到两次 B 变化比例小于阈值。

    参数：
        solver     : HPRSolver 实例（已初始化）
        threshold  : float，预算相对变化阈值（默认1%）
        max_iter   : int，最多循环次数
        init_path  : 可选，传给 HPR_solve 的初始化模式
        verbose    : 是否打印中间过程

    返回：
        dict: {
            "final_B": float,
            "final_y2": dict,
            "iterations": int,
            "B_history": list[float]
        }
    """
    B_old = solver.B
    B_history = [B_old]

    if verbose:
        print(f"[INIT] 初始预算 B = {B_old:.4f}")

    for k in range(1, max_iter + 2):
        # 1️⃣ 求解
        res, solver = HPR_solve(solver, init_path=init_path)
        if res == 3:
            print('wujie')
        if res == 3:
            print("[STOP] 模型无可行解或中断。")
            break

        # 2️⃣ 调用 select_y2_with_budget
        result = select_y2_with_budget(solver)
        if result == False:
            break
        if result is None:
            print("[STOP] 没有找到满足条件的 y2。")
            break

        res_old = res

        y2 = result["y2"]
        B_new = np.minimum(B_old, result["total_L_value"])
        #print(abs(B_new - B_old) / (B_old + 1e-9))

        # 3️⃣ 更新预算
        solver.set_budget(B_new)
        B_history.append(B_new)

        if verbose:
            diff = abs(B_new - B_old) / (B_old + 1e-9)
            print(f"[ITER {k}] B_old={B_old:.4f}, B_new={B_new:.4f}, Δ={diff*100:.2f}%")

        # 4️⃣ 判断收敛
        if abs(B_new - B_old) / (B_old + 1e-9) <= threshold:
            if verbose:
                print(f"[CONVERGED] 在第 {k} 次迭代时收敛 (Δ ≤ {threshold*100:.2f}%)")
            break
        # 更新旧预算
        B_old = B_new

    solver.reset_budget()
    # 最终结果
    return res_old, solver

def branch_and_bound(solver,OD,demand,beta, groups,output = False,value_cut_open = True):
    t_hpr = 0
    t_short_path = 0
    t_update = 0
    t_value_cut = 0
    y = {a: 0 for a in range(solver.A)}
    solver.reset_given_groups_bounds(groups)
    built_list, unbuilt_list = split_arcs_by_status(y)
    _, solver,tt1,tt2 = HPR_solve(solver, init_path=1)
    t_hpr = tt2
    t_short_path += tt1
    solver.fix_y(forbidden=unbuilt_list, set_to=built_list)
    _, solver,tt1,tt2 = HPR_solve(solver, init_path=0)
    t_hpr += tt2
    t_short_path += tt1
    solver.set_time_upper_bound(compute_all_zero_value_cut(solver))
    #snapshot = solver.save_path_snapshot()
    root = Node(list(range(len(groups))))
    value_cut = math.inf
    UB_star = math.inf
    Opt_value = math.inf
    LB_all = 0
    All_LB = 0
    Opt_solution = None
    Q = deque([root])
    weight = solver.evaluate_group_costs(groups)
    weight = [float(v) for k, v in sorted(weight.items())]
    N_count = 0
    value_cut_number = 0
    x_history = []
    while Q:
        N = pop_min_LB(Q)
        N_count += 1
        child = generate_children(N,weight)
        for child_node in child:
            if child_node.budget <= solver.B:
                cond = get_y_from_budget(child_node, weight, solver.B)
                if cond:
                    y = {k: 1 for k in
                         get_all_group_arc_indices(solver, [groups[i] for i in cond[0]])}  # 先把 set 里的设为 1
                    y.update({k: 0 for k in get_all_group_arc_indices(solver, [groups[i] for i in
                                                                               cond[1]])})  # 再把 forbidden 里的设为 0
                    y = fill_missing_y_with_zero(y, solver, verbose=False)
                    solver.reset_given_groups_bounds(groups)
                    built_list, unbuilt_list = split_arcs_by_status(y)
                    solver.fix_y(forbidden=unbuilt_list, set_to=built_list)
                    #st1 = time.time()
                    #solver.add_shortest_paths_given_y(y)
                    #t_hpr += time.time() - st1
                    #_, solver,tt1,tt2 = HPR_solve(solver, init_path=None)
                    #t_hpr += tt2
                    #t_short_path += tt1
                    up_start = time.time()
                    value_cut, UB_star, Opt_value, Opt_solution, solver = update_bounds(
                        solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node
                    )
                    t_update += time.time() - up_start
                    if value_cut_open:
                        value_cut_start = time.time()
                        y = fill_missing_y_with_zero(y, solver, verbose=False)
                        x_result = solver.evaluate_optimal_x_given_y(y)
                        x_history.append(x_result)
                        solver.add_time_constraint_given_x(x_result)
                        t_value_cut += time.time() - value_cut_start
                        value_cut_number += 1
                    continue
                y = {k: 1 for k in
                     get_all_group_arc_indices(solver, [groups[i] for i in child_node.set])}  # 先把 set 里的设为 1
                y.update({k: 0 for k in get_all_group_arc_indices(solver, [groups[i] for i in
                                                                           child_node.forbidden])})  # 再把 forbidden 里的设为 0
                y = fill_missing_y_with_zero(y, solver, verbose=False)
                st1 = time.time()
                solver.add_shortest_paths_given_y(y)
                t_short_path += time.time()-st1
                if child_node.candidate:
                    solver.reset_given_groups_bounds(groups)
                    solver.fix_y(forbidden=get_all_group_arc_indices(solver, [groups[i] for i in child_node.forbidden]), set_to=get_all_group_arc_indices(solver,[groups[i] for i in child_node.set]))
                    #print(f'depth:{len(child_node.forbidden)+len(child_node.set)}')
                    res,solver,tt1,tt2 = HPR_solve(solver,init_path = None)
                    t_hpr += tt2
                    t_short_path += tt1
                    #res , solver = adaptive_budget_iteration(solver)
                    #print(res["obj_val"])
                    if res == 3:
                        continue
                    y2 = res["y_opt"]
                    if all((0 <= val <= 0.001) or (0.999 <= val <= 1) for val in res['y_opt']):
                        y = {k: round(v) for k, v in res['y_opt'].items()}
                        y = fill_missing_y_with_zero(y, solver, verbose=False)
                        up_start = time.time()
                        value_cut, UB_star, Opt_value, Opt_solution,solver = update_bounds(
                            solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node
                        )
                        t_update += time.time() - up_start
                        if value_cut_open:
                            value_cut_start = time.time()
                            y = fill_missing_y_with_zero(y, solver, verbose=False)
                            x_result = solver.evaluate_optimal_x_given_y(y)
                            x_history.append(x_result)
                            solver.add_time_constraint_given_x(x_result)
                            t_value_cut += time.time() - value_cut_start
                            value_cut_number += 1
                    if res["obj_val"] < UB_star:
                        child_node.LB = res["obj_val"]
                        child_node.y = res["y_opt"]
                        Q.append(child_node)
                else:
                    y = {k: 1 for k in get_all_group_arc_indices(solver,[groups[i] for i in child_node.set])}  # 先把 set 里的设为 1
                    y.update({k: 0 for k in get_all_group_arc_indices(solver, [groups[i] for i in child_node.forbidden])})  # 再把 forbidden 里的设为 0
                    y = fill_missing_y_with_zero(y, solver, verbose=False)
                    solver.reset_given_groups_bounds(groups)
                    built_list, unbuilt_list = split_arcs_by_status(y)
                    solver.fix_y(forbidden=unbuilt_list, set_to=built_list)
                    #_, solver,tt1,tt2 = HPR_solve(solver, init_path=None)
                    #t_hpr += tt2
                    #t_short_path += tt1
                    up_start = time.time()
                    value_cut, UB_star, Opt_value, Opt_solution,solver = update_bounds(
                        solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node
                    )
                    t_update += time.time() - up_start
                    if value_cut_open:
                        value_cut_start = time.time()
                        y = fill_missing_y_with_zero(y, solver, verbose = False)
                        x_result = solver.evaluate_optimal_x_given_y(y)
                        x_history.append(x_result)
                        solver.add_time_constraint_given_x(x_result)
                        t_value_cut += time.time() - value_cut_start
                        value_cut_number += 1
        if Q:
            LB_all = np.maximum(0, get_min_LB(Q))
        if output == True and N_count%100 == 0:
            print(f'explore {len(N.forbidden) + len(N.set)}. do {N_count} times')
            print(f'UB_star: {UB_star}',f"LB_all: {LB_all}",f"gap: {(UB_star -  LB_all)/(LB_all+0.001)}")
        if UB_star - LB_all<= 0.01 * LB_all:
            #print(f'do {N_count} times')
            break
    return Opt_solution, Opt_value, t_hpr, t_short_path,t_update,t_value_cut ,N_count,UB_star,LB_all,(UB_star -  LB_all)/(LB_all+0.00000001),value_cut_number


def branch_and_bound2(solver, OD, demand, beta, groups,
                      output=False, value_cut_open=True,
                      return_time_history=False,
                      plot_history=False):
    history_start = time.time()
    t_hpr = 0
    t_short_path = 0
    t_update = 0
    t_value_cut = 0
    y = {a: 0 for a in range(solver.A)}
    solver.reset_given_groups_bounds(groups)
    built_list, unbuilt_list = split_arcs_by_status(y)
    _, solver, tt1, tt2 = HPR_solve(solver, init_path=1)
    t_hpr = tt2
    t_short_path += tt1
    solver.fix_y(forbidden=unbuilt_list, set_to=built_list)
    _, solver, tt1, tt2 = HPR_solve(solver, init_path=0)
    t_hpr += tt2
    t_short_path += tt1
    solver.set_time_upper_bound(compute_all_zero_value_cut(solver))
    # snapshot = solver.save_path_snapshot()
    root = Node(list(range(len(groups))))
    value_cut = math.inf
    UB_star = math.inf
    Opt_value = math.inf
    LB_all = 0
    All_LB = 0
    Opt_solution = None
    Q = deque([root])
    weight = solver.evaluate_group_costs(groups)
    weight = [float(v) for k, v in sorted(weight.items())]
    N_count = 0
    x_history = []

    # --- tracking for plotting ---
    iter_history = []
    time_history = []
    UB_history = []
    LB_history = []
    # -----------------------------

    while Q:
        N = pop_min_LB(Q)
        N_count += 1
        child = generate_children(N, weight)
        for child_node in child:
            if child_node.budget <= solver.B:
                cond = get_y_from_budget(child_node, weight, solver.B)
                if cond:
                    y = {k: 1 for k in
                         get_all_group_arc_indices(solver, [groups[i] for i in cond[0]])}
                    y.update({k: 0 for k in get_all_group_arc_indices(
                        solver, [groups[i] for i in cond[1]])})
                    y = fill_missing_y_with_zero(y, solver, verbose=False)
                    solver.reset_given_groups_bounds(groups)
                    built_list, unbuilt_list = split_arcs_by_status(y)
                    solver.fix_y(forbidden=unbuilt_list, set_to=built_list)

                    up_start = time.time()
                    value_cut, UB_star, Opt_value, Opt_solution, solver = update_bounds(
                        solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node
                    )
                    t_update += time.time() - up_start

                    if value_cut_open:
                        value_cut_start = time.time()
                        y = fill_missing_y_with_zero(y, solver, verbose=False)
                        x_result = solver.evaluate_optimal_x_given_y(y)
                        x_history.append(x_result)
                        solver.add_time_constraint_given_x(x_result)
                        t_value_cut += time.time() - value_cut_start
                    continue

                y = {k: 1 for k in get_all_group_arc_indices(
                    solver, [groups[i] for i in child_node.set])}
                y.update({k: 0 for k in get_all_group_arc_indices(
                    solver, [groups[i] for i in child_node.forbidden])})
                y = fill_missing_y_with_zero(y, solver, verbose=False)
                st1 = time.time()
                solver.add_shortest_paths_given_y(y)
                t_short_path += time.time() - st1

                if child_node.candidate:
                    solver.reset_given_groups_bounds(groups)
                    solver.fix_y(
                        forbidden=get_all_group_arc_indices(
                            solver, [groups[i] for i in child_node.forbidden]),
                        set_to=get_all_group_arc_indices(
                            solver, [groups[i] for i in child_node.set])
                    )
                    res, solver, tt1, tt2 = HPR_solve(solver, init_path=None)
                    t_hpr += tt2
                    t_short_path += tt1

                    if res == 3:
                        continue
                    y2 = res["y_opt"]
                    if all((0 <= val <= 0.001) or (0.999 <= val <= 1) for val in res['y_opt']):
                        y = {k: round(v) for k, v in res['y_opt'].items()}
                        y = fill_missing_y_with_zero(y, solver, verbose=False)
                        up_start = time.time()
                        value_cut, UB_star, Opt_value, Opt_solution, solver = update_bounds(
                            solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node
                        )
                        t_update += time.time() - up_start

                        if value_cut_open:
                            value_cut_start = time.time()
                            y = fill_missing_y_with_zero(y, solver, verbose=False)
                            x_result = solver.evaluate_optimal_x_given_y(y)
                            x_history.append(x_result)
                            solver.add_time_constraint_given_x(x_result)
                            t_value_cut += time.time() - value_cut_start

                    if res["obj_val"] < UB_star:
                        child_node.LB = res["obj_val"]
                        child_node.y = res["y_opt"]
                        Q.append(child_node)
                else:
                    y = {k: 1 for k in get_all_group_arc_indices(
                        solver, [groups[i] for i in child_node.set])}
                    y.update({k: 0 for k in get_all_group_arc_indices(
                        solver, [groups[i] for i in child_node.forbidden])})
                    y = fill_missing_y_with_zero(y, solver, verbose=False)
                    solver.reset_given_groups_bounds(groups)
                    built_list, unbuilt_list = split_arcs_by_status(y)
                    solver.fix_y(forbidden=unbuilt_list, set_to=built_list)

                    up_start = time.time()
                    value_cut, UB_star, Opt_value, Opt_solution, solver = update_bounds(
                        solver, y, value_cut, UB_star, Opt_value, Opt_solution, child_node
                    )
                    t_update += time.time() - up_start

                    if value_cut_open:
                        value_cut_start = time.time()
                        y = fill_missing_y_with_zero(y, solver, verbose=False)
                        x_result = solver.evaluate_optimal_x_given_y(y)
                        x_history.append(x_result)
                        solver.add_time_constraint_given_x(x_result)
                        t_value_cut += time.time() - value_cut_start

        if Q:
            LB_all = np.maximum(0, get_min_LB(Q))

        # --- record UB & LB after this iteration ---
        if UB_star < math.inf:
            iter_history.append(N_count)
            time_history.append(time.time() - history_start)
            UB_history.append(UB_star)
            LB_history.append(LB_all)
        # ------------------------------------------

        if output and N_count % 100 == 0:
            print(f'explore {len(N.forbidden) + len(N.set)}. do {N_count} times')
            print(f'UB_star: {UB_star}', f"LB_all: {LB_all}",
                  f"gap: {(UB_star - LB_all) / (LB_all + 0.001)}")
        if UB_star - LB_all <= 0.01 * LB_all:
            break

    # --- plot UB and LB vs iteration ---
    if plot_history and iter_history:
        plt.figure()
        plt.plot(iter_history, UB_history, label='UB')
        plt.plot(iter_history, LB_history, label='LB')
        plt.xlabel('Iteration')
        plt.ylabel('Bound value')
        plt.title('Branch-and-Price: UB and LB over iterations')
        plt.legend()
        plt.grid(True)
        plt.show()
    # -----------------------------------

    result = (Opt_solution, Opt_value, t_hpr, t_short_path,
              t_update, t_value_cut, N_count, UB_star, LB_all,
              (UB_star - LB_all) / (LB_all + 1e-8),
              iter_history, UB_history, LB_history)
    if return_time_history:
        return result + (time_history,)
    return result
