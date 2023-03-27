
from PIL import Image
import numpy as np
import factorcal
import soc
import slht
import silhouette_width


nk = 3 #No of cluster
image = Image.open('/test_image.jpeg') # you can give location of photo according to your own photo location
image.show(image)
data_image = np.array(image)

a = data_image

[f, h, k] = a.shape
R, G, B = np.zeros((f, h)), np.zeros((f, h)), np.zeros((f, h))
for j in range(f):
    for i in range(h):
        R[j, i] = a[j, i, 0]                                  # extracting R value from original matrix a
        G[j, i] = a[j, i, 1]                                  # extracting G value from original matrix a
        B[j, i] = a[j, i, 2]                                  # extracting B value from original matrix a

n = f * h                                                       # n is total no. of data points
R = np.reshape(R, f*h) 
G = np.reshape(G, f*h) 
B = np.reshape(B, f*h) 
x = np.zeros((n, 3))
for j in range(n):
    x[j, :] = [R[j], G[j], B[j]]

print("printing n", n)

print(x.shape)

fac = factorcal(x,nk,1)
result = soc(x,nk,fac) # result is a dictionary

s = silhouette_width(x, result["idx"])
[S, GSI] = slht(s, result["idx"], result["n"], result["m"], nk)

print(result)
print(GSI) # GSI is a partition index, measure how good clustring has been done


