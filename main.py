import numpy as np
import tsplib95
import argparse
from scipy.spatial.distance import cdist
from alns import *
from candidate import *
from mstree import *
from constructAlg import *
import time
import logging
from LinKernighan import *
from numba import njit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def solve(nc, pi, try_num=100, candidate_index=None):
    pi_sum = 2*np.sum(pi)
    bestG = np.inf
    # r, best_i, best_j = getLT(c, pi)[2:]
    # alpha, candidate_index = geneCandidate(r, nc, best_i, 10)
    for i in range(try_num):
        t = np.arange(len(nc))
        np.random.shuffle(t)
        # if candidate_index is not None:
        #     t = LKConstructJIT(alpha, candidate_index)
        while 1:
            if candidate_index is None:
                t, G = LK2b(nc, t)
            else:
                t, G = LK2c(nc, t, candidate_index)
            if G <= 0: break
        G = totaldist(nc, t)
        if G < bestG:
            bestG = G
            print('*', i, ': %d' % np.round(G-pi_sum))
        else:
            print(' ', i, ': %d' % np.round(G-pi_sum))
    print('Best result is %d,lower bound is %.2f' % (np.round(bestG-pi_sum), w))


def exhaustSolve(c, iter_num=1000, break_num=3, max_break_num=3, try_num=300):
    best_t = None
    best_dist = np.inf
    for j in range(try_num):
        t = np.arange(len(c))
        np.random.shuffle(t)
        delta = 1
        while delta > 0:
            t, delta = LK2(c, t)
        ni = np.random.randint(len(c))
        t = np.array(list(t[ni:]) + list(t[:ni]))
        while delta > 0:
            t, delta = LK2(c, t)
        for i in range(1000):
            breaks_list = generateBreaks(t, iter_num, break_num)
            positions, directions, deltas = zip(
                *[exhaustRepair(c, t, breaks, getPos(breaks), getDir(breaks)) for breaks in breaks_list])
            if max(deltas) > 0:
                best_index = np.argmax(np.array([deltas]))
                breaks, position, direction = breaks_list[best_index], positions[best_index], directions[best_index]
                t = restorePieceBreaks(t, breaks, position, direction)
            elif break_num < max_break_num:
                break_num += 1
            else:
                break
        dist = totaldist(c, t)
        if dist < best_dist:
            best_dist = dist
            best_t = t
            print('*', j, best_dist)
    return best_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem")
    parser.add_argument("-t", "--try_number", type=int, default=10)
    args = parser.parse_args()

    problem = tsplib95.load(args.problem)
    coords = np.array([(coord[0], coord[1]) for city, coord in problem.node_coords.items()])
    distA = cdist(coords, coords, metric='euclidean')
    c = np.round(distA).astype(np.int32)

    st = time.perf_counter()
    w, pi, deg = ascent(np.round(c))
    print('finish ascent, w=%.2f, deg=%d' % (w, deg))

    if deg == 0:
        r, i, j = getLT(c, pi)[2:]
        r[i][j] = 1
        t = constructTourFromMatrix(r)
        print('Best result is %d,lower bound is %d' % (totaldist(c, t), int(w)))
    else:
        nc = ((c.T + pi).T + pi)
        r, best_i, best_j = getLT(c, pi)[2:]
        candidate_index = geneCandidate(r, nc, best_i, 10)[1]
        solve(nc, pi, try_num=args.try_number, candidate_index=candidate_index)
    et = time.perf_counter()
    print('Consumes %.3f seconds' % (et - st))

