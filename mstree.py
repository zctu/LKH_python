import tsplib95
import numpy as np
from scipy.spatial.distance import cdist
from numba import njit
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit
def minimum_spanning_tree(matrix):
    n = len(matrix)
    non_matrix = np.zeros_like(matrix)
    v = np.zeros(n)
    v[0] = 1
    while sum(v) < n:
        mst_len = np.inf
        for i in np.arange(n):
            if v[i] > 0:
                for j in np.arange(n):
                    if v[j] == 0:
                        if matrix[i][j] < mst_len:
                            mst_len = matrix[i][j]
                            bi = i
                            bj = j

        v[bj] = 1
        non_matrix[bi][bj] = 1
    return non_matrix


@njit
def getLT(c, pi):
    c = (c.T + pi).T + pi
    r = minimum_spanning_tree(c)
    max_sec_dist = 0
    v = np.sum(r, axis=0) + np.sum(r, axis=1) - 2
    best_i = 0
    best_j = 0
    for i in range(len(v)):
        if v[i] < 0:
            first_dist = np.inf
            sec_dist = np.inf
            first_ind = 0
            sec_ind = 0
            for j in range(len(v)):
                if i != j:
                    if c[i][j] < first_dist:
                        sec_dist = first_dist
                        first_dist = c[i][j]
                        sec_ind = first_ind
                        first_ind = j
                    elif c[i][j] < sec_dist:
                        sec_dist = c[i][j]
                        sec_ind = j
            if max_sec_dist < sec_dist:
                max_sec_dist = sec_dist
                best_i = i
                best_j = sec_ind
    v[best_i] += 1
    v[best_j] += 1
    return np.sum(r*c) - 2*sum(pi) + max_sec_dist, v, r, best_i, best_j

@njit
def ascent(c):
    pi = np.zeros(len(c))
    t = 1
    best_w, v = getLT(c, pi)[:2]
    best_deg = np.sum(v*v)
    last_v = v
    period = max(len(c)//2, 100)
    while period > 0 and t > 0.01:
        period_continue = False
        for k in range(period):
            for i in range(len(v)):
                if v[i] == 0:
                    last_v[i] = 0
            pi = pi+t*(0.7*v+0.3*last_v)
            last_v = v
            w, v = getLT(c, pi)[:2]
            deg = np.sum(v*v)
            if np.sum(np.abs(v)) == 0:
                return w, pi, 0
            elif w > best_w or (w == best_w and deg < best_deg):
                best_w = w
                best_deg = deg
                t *= 2
                if k == period-1:
                    period_continue = True
            else:
                if k > period / 2:
                    t = (t*3)/4
        if not period_continue:
            t = t/2
            period //= 2
    return best_w, pi, best_deg