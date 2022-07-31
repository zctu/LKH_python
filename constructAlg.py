import numpy as np
from candidate import  geneCandidate
from util import totaldist
from numba import njit

@njit
def greedyConstruct(c, i=0):
    num = len(c)
    uv = np.array(list(range(i)) + list(range(i + 1, num)))
    t = np.array([i] + [0] * (num - 1))
    for i in np.arange(1, num):
        cur = t[i - 1]
        best_dist = c[cur][uv[0]]
        best_index = 0
        for uvi, j in enumerate(uv):
            dis = c[cur][j]
            if dis < best_dist:
                best_dist = dis
                best_index = uvi
        t[i] = uv[best_index]
        uv = np.delete(uv, best_index)
    return t


@njit
def LKConstructJIT(alpha, candidate_index, first_id=None):
    n = len(alpha)
    uv = np.ones(n, np.int32)
    if first_id is None:
        i = np.random.randint(n)
    else:
        i = first_id
    route = -1 * np.ones(n, np.int32)
    route[0] = i
    route_idx = 1
    uv[i] = 0
    while np.sum(uv) > 0:
        candidates = candidate_index[i]
        no_candi_flag = True
        for j in candidates:
            if i != j and uv[j] == 1:
                route[route_idx] = j
                route_idx += 1
                uv[j] = 0
                i = j
                no_candi_flag = False
                break

        # if no candidate is ok, then
        if no_candi_flag:
            j = 0
            ss = np.arange(len(uv))
            np.random.shuffle(ss)
            for uv_idx in ss:
                if uv[uv_idx] == 1:
                    uv[uv_idx] = 0
                    j = uv_idx
                    break
            route[route_idx] = j
            route_idx += 1
    return route


def LKConstruct(alpha, candidate_index):
    n = len(alpha)
    uv = set(range(n))
    i = np.random.randint(n)
    route = [i]
    uv.remove(i)

    while len(uv) > 0:
        candidates = candidate_index[i]
        no_candi_flag = True
        for j in candidates:
            if i != j and j in uv and alpha[i][j] == 0:
                route.append(j)
                uv.remove(j)
                i = j
                no_candi_flag = False
                break
        if no_candi_flag:
            j = uv.pop()
            route.append(j)
    return np.array(route)

import tsplib95
import numpy as np
from scipy.spatial.distance import cdist
from mstree import getLT

if __name__ == '__main__':
    problem = tsplib95.load('/Users/chen/work/lk_heuristic/src/lk_heuristic/xqf131.tsp')
    pos = np.array([(coord[0], coord[1]) for city, coord in problem.node_coords.items()])
    c = np.round(cdist(pos, pos, metric='euclidean')).astype(np.int32)
    # w, pi = ascent(np.round(c))
    # r, best_i, best_j = getLT(c, pi)[2:]
    # c = (c.T + pi).T + pi
    r, best_i, best_j = getLT(c, np.zeros(len(c)))[2:]
    alpha, candidate_index = geneCandidate(r, c, best_i)
    route = LKConstructJIT(alpha, candidate_index, 0)
    print(len(route),totaldist(c, route))