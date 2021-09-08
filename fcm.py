import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
df = pd.read_excel('Documents/MATLAB/data.xlsx')

df_train = df.iloc[:,0:3]

print(df_train)
df_train = np.transpose(df_train)

cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(df_train, 3, 2, error=0.005, maxiter=1000)


for pt in cntr:
    print (pt)
  
df_test = np.array([[66.2,
68.6,
90,
68.7,
89.6,
60.3,
61.7,
90,
89.8,
69.6,
62.4],[23.6,
7.2,
9.6,
7.2,
8.4,
13,
7.2,
9.6,
18.5,
19.6,
10.1],[217.68,
33.37,
50,
33.55,
66.22,
25.32,
49.38,
50,
102.33,
37.8,
29.32]])
print(df_test)



u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(df_test, cntr, 2, error=0.005, maxiter=1000)
print(u)