import numpy as np
import silhouette_width
import soc
import lagrangepoly
import slht


def factorcal(x, nk, iter):
    flag = 0
    factor = np.ones((12,nk))
    GS = np.zeros(12)
    while iter <= 10:
        print("itration no. ", iter)
        result = soc(x, nk, factor[iter, :])
        s = silhouette_width(x, result["idx"])
        
        S, GS[iter] = slht(s, result["idx"], result["n"], result["m"], nk)
        print("printing del and S for eache cluster", result["d"], S)
        if min(result["m"]) == 0:
            flag == 1
            break
        
        for g in range(nk-1):
            for gg in range(g+1, nk):
                if result["d"][g] == result["d"][gg]:
                    flag = 1
                    break
            if flag == 1:
                break

        aa = lagrangepoly(result["d"], S)
        if aa == "no":
          break
        factor[iter+1] = aa
        iter += 1

    label = np.argmax(GS)
    fac = factor[label, :]
    return fac

