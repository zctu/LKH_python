import tsplib95
import numpy as np
from scipy.spatial.distance import cdist
import heapq
import logging
from mstree import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def geneCandidate_bak(r, c, first_node):
    x, y = np.where(r > 0)
    alpha = c.copy()
    beta = np.zeros_like(c)
    for i in range(len(x)):
        beta[x[i]][y[i]] = beta[y[i]][x[i]] = c[y[i]][x[i]]

    neighbour = beta > 0
    for i in range(len(c)):
        if i == first_node:
            for j in range(len(c)):
                if i == j:
                    continue
                if j in np.where(neighbour[i])[0]:
                    alpha[i][j] = alpha[j][i] = 0
                else:
                    second_cost = np.sort(c[i])[2]
                    alpha[i][j] = alpha[j][i] = c[i][j] - second_cost
        else:
            mst = [i]
            heapq.heapify(mst)
            uv = set(range(len(c)))
            while len(mst) > 0:
                j = mst.pop()
                uv.remove(j)
                for nj in np.where(neighbour[j])[0]:
                    if nj in uv:
                        mst.append(nj)
                        if j != first_node:
                            d = beta[i][nj] = beta[nj][i] = max(beta[i][j], c[nj][j])
                            a = alpha[i][nj] = alpha[nj][i] = c[i][nj] - beta[i][nj]
    #    alpha[best_i][best_j] = alpha[best_j][best_i] = 0
    return alpha


def geneCandidate(r, c, first_node, max_candicate=10,  suc=None, max_alpha=None):
    x, y = np.where(r > 0)
    alpha = np.zeros_like(c, float)
    beta = np.zeros_like(c, float)
    for i in range(len(x)):
        beta[x[i]][y[i]] = beta[y[i]][x[i]] = c[y[i]][x[i]]
        r[y[i]][x[i]] = 1

    if suc is None:
        suc = [first_node]
        build_suc = True
    else:
        build_suc = False

    neighbour = r > 0
    dad = {first_node: None}
    mst = [first_node]
    heapq.heapify(mst)
    uv = set(range(len(c)))
    while len(mst) > 0:
        j = mst.pop()
        uv.remove(j)
        for nj in np.where(neighbour[j])[0]:
            if nj in uv:
                mst.append(nj)
                dad[nj] = j
                if build_suc:
                    suc.append(nj)

    for idx, i in enumerate(suc):
        if i == first_node:
            for jdx, j in enumerate(suc):
                if i == j or j == first_node:
                    continue
                if j == dad[i] or i == dad[j]:
                    alpha[i][j] = alpha[j][i] = 0
                else:
                    second_cost = np.sort(c[i])[2]
                    alpha[i][j] = alpha[j][i] = c[i][j] - second_cost
        else:
            for jdx, j in enumerate(suc):
                if idx >= jdx or j == first_node:
                    continue
                if beta[i][dad[j]] == 0:
                    beta[i][j] = c[j][dad[j]]
                else:
                    beta[i][j] = max(beta[i][dad[j]], c[j][dad[j]])
                if j == dad[i] or i == dad[j]:
                    alpha[i][j] = alpha[j][i] = 0
                else:
                    alpha[i][j] = alpha[j][i] = c[i][j] - beta[i][j]

    max_candicate = min(max_candicate, len(c))
    candidate_index = np.argsort(alpha)[:, :max_candicate]
    if max_alpha is not None:
        keep_nums = np.sum(alpha < max_alpha, axis=1)
        for i, keep_num in enumerate(keep_nums):
            candidate_index[i][keep_num:] = -1
    return alpha, candidate_index