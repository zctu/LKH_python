from numba import njit
import numpy as np


@njit
def constructTourFromMatrix(r):
    n = len(r)
    suc = np.zeros(n, np.int32)
    visited = np.zeros(n, np.int32)
    for i in range(n):
        for j in range(i):
            if r[i][j] > 0:
                r[j][i] = r[i][j]
                r[i][j] = 0

    for i in range(n):
        for j in range(n):
            if r[i][j] > 0:
                if visited[i] == 0:
                    suc[i] = j
                    visited[i] = 1
                else:
                    suc[j] = i
                    visited[j] = 1

    tour = np.zeros(n, np.int32)
    node = 0
    index = 1
    while suc[node] != 0:
        tour[index] = suc[node]
        node = suc[node]
        index += 1
    return tour


@njit
def totaldist(c, t):
    s = 0
    for i in np.arange(len(t)):
        s += c[t[i - 1]][t[i]]
    return s


@njit
def LK2(c, t):
    n = len(t)
    for i in range(n - 2):
        x1 = (t[i], t[i + 1])
        best_gain = 0
        best_j = 0
        for j in range(n - 1):
            y1 = (t[i + 1], t[j + 1])
            G1 = c[x1[0]][x1[1]] - c[y1[0]][y1[1]]
            if ((i + 1) < j or (j + 1) < i):
                # if G1 > 0 and ((i + 1) < j or (j + 1) < i):
                x2 = (t[j + 1], t[j])
                y2 = (t[j], t[i])
                G = G1 + c[x2[0]][x2[1]] - c[y2[0]][y2[1]]
                if G > best_gain:
                    best_gain = G
                    best_j = j
        if best_gain > 0:
            j = best_j
            if j < i: i, j = j, i
            t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            return t, best_gain
    return t, 0


@njit
def getNode(index, breaks):
    if index % 2 == 0:
        return breaks[index // 2 - 1]
    else:
        return breaks[index // 2] - 1


@njit
def partTotalDist(c, t, breaks, position, direction):
    r = np.zeros(len(position) * 2, np.int64)
    for i in range(len(position)):
        r[2 * i] = t[getNode(position[i] * 2 + direction[i], breaks)]
        r[2 * i + 1] = t[getNode(position[i] * 2 + 1 - direction[i], breaks)]
    return totaldist(c, r)


# def partTotalDist(c, t, breaks, position, direction):
#     nodes = np.zeros(len(position) * 2, int)
#     for i in range(len(position)):
#         nodes[2 * i] = breaks[i - 1]
#         nodes[2 * i + 1] = breaks[i] - 1
#     r = np.zeros(len(position) * 2, int)
#     for i in range(len(position)):
#         r[2 * i] = t[nodes[position[i] * 2 + direction[i]]]
#         r[2 * i + 1] = t[nodes[position[i] * 2 + 1 - direction[i]]]
#     return totaldist(c, r)


@njit
def ind(t, ti):
    for i in range(len(t)):
        if t[i] == ti:
            return i
    return -1
