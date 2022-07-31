import tsplib95
import numpy as np
from scipy.spatial.distance import cdist
import heapq
from util import *
import logging
from mstree import *
from candidate import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit
def TwoOptMove(i, j, t):
    n = len(t)
    if i < j:
        t[i:j] = t[i:j][::-1]
    else:
        temp = np.zeros(n + j - i, np.int32)
        temp[:n - i] = t[i:]
        temp[n - i:] = t[:j]
        temp = temp[::-1]
        t[i:] = temp[:n - i]
        t[:j] = temp[n - i:]
    return t


@njit
def bestOptMove(c, i, G0, t):
    n = len(t)
    best_g2 = -np.inf
    best_j = -1
    for j in np.arange(n):
        if (i + 1) % n != j and i != j and (i - 1) != j and (j + 1) % n != i:
            y1 = (t[i], t[j])
            G1 = G0 - c[y1[0]][y1[1]]
            if G1 > 0:
                x2 = (t[j], t[j-1])
                y2 = (t[j-1], t[i-1])
                G2 = G1 + c[x2[0]][x2[1]]
                G = G2 - c[y2[0]][y2[1]]
                if G > 0.01:
                    t = TwoOptMove(i, j, t)
                    return t, G2, G, j
                if G2 > best_g2:
                    best_g2 = G2
                    best_j = j
    if best_j == -1:
        return t, G0, 0, -1
    return TwoOptMove(i, best_j, t), best_g2, 0, best_j


"""
introduce candidates
"""
@njit
def bestOptMove2(c, i, G0, t, candidate):
    n = len(t)
    best_g2 = -np.inf
    best_j = -1
    for tj in candidate:
        if tj == -1:
            continue
        j = get_idx(tj, t)
        if (i + 1) % n != j and i != j and (i - 1) != j and (j + 1) % n != i:
            y1 = (t[i], t[j])
            G1 = G0 - c[y1[0]][y1[1]]
            if G1 > 0:
                x2 = (t[j], t[j-1])
                y2 = (t[j-1], t[i-1])
                G2 = G1 + c[x2[0]][x2[1]]
                G = G2 - c[y2[0]][y2[1]]
                if G > 0.01:
                    t = TwoOptMove(i, j, t)
                    return t, G2, G, j
                if G2 > best_g2:
                    best_g2 = G2
                    best_j = j
    if best_j == -1:
        return t, G0, 0, -1
    return TwoOptMove(i, best_j, t), best_g2, 0, best_j

@njit
def LK2b(c, t):
    n = len(t)
    max_swaps = len(t)
    for i in range(n):
        nt = t.copy()
        x1 = (nt[i-1], nt[i])
        G0 = c[x1[0]][x1[1]]
        for swaps in range(max_swaps):
            nt, G0, G, j = bestOptMove(c, i, G0, nt)
            if G > 0.01:
                return nt, G
    return t, 0


@njit
def get_idx(i, t):
    for idx in range(len(t)):
        if i == t[idx]:
            return idx
    return -1


@njit
def LK2c(c, t, candidates):
    n = len(t)
    max_swaps = len(t)
    for i in range(n):
        nt = t.copy()
        x1 = (nt[i-1], nt[i])
        G0 = c[x1[0]][x1[1]]
        for swaps in range(max_swaps):
            nt, G0, G, j = bestOptMove2(c, i, G0, nt, candidates[nt[i]])
            if G > 0.01:
                return nt, G
    return t, 0