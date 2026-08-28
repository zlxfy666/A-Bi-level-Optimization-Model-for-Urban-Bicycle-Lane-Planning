from OPT_HPR import *
import numpy as np
import copy
import time
def init_res(solver):
    """
    构造一个初始 res 字典:
    - pi_flow_link: 所有 (g,a) 对偶值 = 1
    - pi_demand: 所有 (g,r,s) 对偶值 = np.inf
    """
    pi_flow_link = {(g, a): 1.0 for g in solver.G for a in range(solver.A)}
    pi_demand = {(g, r, s): np.inf for g in solver.G for (r, s) in solver.OD[g]}
    return pi_flow_link,pi_demand

def apply_dual_flow_link_weights(dual_flow_link, solver, g):
    """
    将 solver.graph 的所有边权更新为 dual_flow_link[g,a] 的值

    :param dual_flow_link: solve() 返回的 "dual_flow_link" 字典
    :param solver: HPRSolver 对象
    :param g: 流类别
    """
    updates = []
    for a, (u, v) in enumerate(solver.arcs):
        val = dual_flow_link[(g, a)]
        updates.append((u, v, float(val)))
    solver.graph.update_edges(updates)

def HPR_solve(solver, init_path=None, verbose=False):
    #solver.clear_all_paths()
    if init_path == 1:
        tt1 = 0
        tt2 = 0
        pi_flow_link, pi_demand  = init_res(solver)
    else:
        tt1 = 0
        tt2 = 0
        t1 = time.time()
        res = solver.solve()
        tt2+=time.time()-t1
        if res == 3:
            if verbose:
                print("[HPR_solve] solver returned infeasible/status=3")
            return 3,solver, tt1, tt2
        pi_flow_link, pi_demand = res['dual_flow_link'], res['dual_demand']
        #pi_flow_link, pi_demand = np.abs(pi_flow_link), np.abs(pi_demand)
    new = 0
    iter = 0
    while True:
        iter+=1
        new_paths = 0
        start = time.time()
        for g in solver.OD:
            apply_dual_flow_link_weights(pi_flow_link, solver, g)

            # 按起点 r 分组所有 OD 对
            rs_by_r = defaultdict(list)
            for (r, s) in solver.OD[g]:
                rs_by_r[r].append(s)

            for r, targets in rs_by_r.items():
                # 一次性求出 r -> 所有节点的最短路与路径
                dist_vec, paths_dict = solver.graph.dijkstra_all_from_source(r)

                for s in targets:
                    T_DC = dist_vec[s]
                    if T_DC < pi_demand[g, r, s]:
                        path_nodes = paths_dict[s]
                        path_edges = solver.get_path_edges(path_nodes)
                        New = solver.add_path(g, r, s, path_edges)
                        if New:
                            new_paths += 1
                            new = 1
        if verbose:
            print(f"[HPR_solve] new paths added: {new_paths}")
        middle = time.time()
        res = solver.solve()
        #_,result = solver.check_unique_active_path()
        #print(result)
        end = time.time()
        tt1 += (middle - start)
        tt2 += (end - middle)
        if new == 0 or init_path ==1:
            break
        new = 0
        pi_flow_link, pi_demand = res['dual_flow_link'], res['dual_demand']

    #print(f'add path:{tt1},solve:{tt2},iter:{iter},path_number:{len(solver.h)},[模型规模] Vars={solver.m.NumVars:,}, Constrs={solver.m.NumConstrs:,}, NonZeros={solver.m.NumNZs:,}')
    return res, solver, tt1, tt2





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
