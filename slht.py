import numpy as np


def slht(s, idx, n, m, nk):
    idx = np.asarray(idx, dtype= np.int)
    S_const = np.zeros(nk)

    for r in range(n):
        S_const[idx[r]] += s[r]
    S = np.zeros(nk)
    for j in range(nk):
        S[j] = (1/m[j]) * S_const[j]
    GS = (1/nk) * np.sum(S)

    return S, GS