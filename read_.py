import numpy as np
import scipy.sparse as sp

def read_sioux_falls(net_file: str, trips_file: str):
    """
    读取 SiouxFalls 网络和需求数据
    :param net_file: 链路文件 (SiouxFalls_net.tntp)
    :param trips_file: 需求文件 (SiouxFalls_trips.tntp)
    :return:
        edges : list of (i, j, w)   # 边列表，i->j，权重w
        demand_matrix : numpy.ndarray   # OD需求矩阵
    """
    # 读取网络文件
    edges = []
    with open(net_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('~') or line.startswith('<'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                i = int(parts[0]) - 1  # 节点编号从1开始，这里转为0索引
                j = int(parts[1]) - 1
                length = float(parts[3])  # length作为权重
                edges.append((i, j, length))

    # 读取需求文件
    with open(trips_file, 'r') as f:
        lines = f.readlines()

    n = max(max(i, j) for i, j, _ in edges) + 1
    demand_matrix = np.zeros((n, n))

    current_origin = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('~') or line.startswith('<'):
            continue
        if line.lower().startswith('origin'):
            try:
                current_origin = int(line.split()[1]) - 1
            except:
                continue
        else:
            # 行格式示例: "   2 : 100 ;  3 : 200 ;"
            segments = line.split(';')
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                if ':' in seg:
                    dest_str, val_str = seg.split(':')
                    dest = int(dest_str.strip()) - 1
                    dem = float(val_str.strip())
                    demand_matrix[current_origin, dest] = dem

    return edges, demand_matrix