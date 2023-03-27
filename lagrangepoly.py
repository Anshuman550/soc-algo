import numpy as np

# lagrange's interpolation

def lagrangepoly(X, Y):
   
    N = len(X)
    pvals = np.zeros((N, N))

    for i in range(N):
        pp = np.poly(X[(np.arange(N) != i)])
        if np.polyval(pp, X[i] ) != 0 :
          pvals[i, :] = pp / np.polyval(pp, X[i] )
        else:
          pvals[i, :] = 0

    P = np.matmul(Y, pvals)
    P[N-1] = P[ N-1] -1
    r = np.roots(P)
    if len(r) == 0:
        return "no"
    nk = r.shape[0] + 1
    sumn= np.zeros( r.shape )
    sumn = sumn.astype(complex)
    for i in range(nk-1):
        #sumn[i] = np.polyval(P, r[i])
        for j in range(nk):
            sumn[i]=sumn[i] + P[nk-j-1]*r[i]**(j);
        
    
    label = np.argmax(np.abs(sumn - 1))
    eta = abs(r[label])
    print("printing eta", eta)
    return eta/X #new Beta, where X is delta for each cluster