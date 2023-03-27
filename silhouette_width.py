import numpy as np

def silhouette_width(data, labels):
    n_samples = len(data)
    n_clusters = len(np.unique(labels))
    
    # Calculate the pairwise distance matrix
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            D[i][j] = (np.linalg.norm(data[i] - data[j]))**2
    
    # Calculate the silhouette score for each sample
    
    s = np.zeros(n_samples)
    for i in range(n_samples):
        a_i = np.mean([D[i][j] for j in range(n_samples) if labels[j] == labels[i] and i != j]) 
        if np.isnan(a_i):
          a_i = 0.0
        arr = [np.mean([D[i][j] for j in range(n_samples) if labels[j] == k]) for k in range(n_clusters) if k != labels[i]]
        if len(arr) == 0 :
          b_i = 0.0
        else:
          b_i = np.min(arr)
        
        if np.isnan(b_i):
          b_i = 0.0
        
        s[i] = (b_i - a_i) / max(a_i, b_i)
    
    # Return the average silhouette score
    return s