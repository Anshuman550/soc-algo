import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def soc(x, nk, factor):
    # To implement Improved Mountain Clustering Technique(IMC)
    # for images containing RGB data points or for any (n x k) matrix
    # WITH ALL THE DATA POINTS INCLUDED
    g = np.arange(1, nk+1)
    n, k = x.shape
    x = np.asarray(x , dtype= np.float64)
    # STEP 1
    # to Normalize each dimension of hyperspace
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
  
    z = x_max - x_min

    u = np.zeros((n, k))*1.0
    for j in range(n):
        y = x[j, :] - x_min
        if z.any() != 0:
            y = y * 1.0
            w = y / z
        else:
            w = np.zeros(k)
        u[j, :] = w
    

    m = np.zeros(nk)
    d = np.zeros(nk)*1.000
    d1 = np.zeros(nk)*1.0000
    P = np.zeros([n, nk])
   
    zmax = np.asarray(np.zeros(nk) , dtype = np.int)

    points1 = np.asarray(np.array(range(n)), dtype= np.int)
    points2 = np.asarray(np.array([]), dtype= np.int)
    idx = np.zeros(n)
    cluster_coordinate = np.zeros([nk, x.shape[1]])*1.0
    for v in range(nk):
        if points1.shape[0] != 0:
            
            # STEP 2
            # to Determine the parameter d1 for each window
            for j in range( points1.shape[0] ):
                if np.sum(x[ points1[j], :]) != 0:
                    d[v] += np.min(x[points1[j], :]) / np.sum(x[points1[j], :])
                    

            d1[v] = ((1.0/(2*points1.shape[0]))*d[v]) * factor[v]
            # STEP 3
            # to Calculate the potential value of each point using a mountain function

            #for r in range(points1.shape[0]):
                #P[r,v] = 0 # we don't have to do this
                #for j in range(points1.shape[0]):
                    #P[r, v] +=  np.exp( - ( np.sum((u[r, :] - u[j, :])**2) / d1[v]**2 ) )
            count = 0
            for r in range(points1.shape[0]):
                #count += 1
                #print("inloop", count)
                for j in range(points1.shape[0]):
                    P[ points1[r], v] +=  np.exp( - ( np.sum((u[points1[r], :] - u[points1[j], :])**2) / d1[v]**2 ) )

            # STEP 4
            # to Select the first cluster center according to the highest value of P1
            zmax[v] = np.argmax(P[:, v]) 
            cluster_coordinate[v, : ] =  u[zmax[v], :]
            P = np.zeros([n, nk])
            # STEP 5
            # to Assign concerned data points to the first cluster
            
            for i in range( points1.shape[0] ):

                if ( euclidean_distance( u[points1[i], :], cluster_coordinate[v, :] )**2 <= d1[v]) :
                    idx[points1[i]] = v
                    m[v] += 1
                else:
                    
                    points2 = np.append(points2, points1[i])
            
            points1 = points2
            points2 = points2 = np.asarray(np.array([]), dtype= np.int)

    
    min_D = 99999999.9
    index_cluster = 0
    
    for i in range(points1.shape[0]):
        for j in range(nk):
            D = euclidean_distance( cluster_coordinate[j, :] , u[points1[i], :])
            if D < min_D:
                min_D = D
                index_cluster = j
    
        idx[points1[i]] = index_cluster # assigning the point to cluster with closer cluster
        m[index_cluster] += 1


    result = { 

        "idx" : idx,
        "m"   : m,
        "d"   : d1,
        "n"   : n,
        "cluster_coordinate" : cluster_coordinate
    }

    return result


