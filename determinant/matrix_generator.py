import numpy as np
from scipy.stats import ortho_group

dim = 100
m = ortho_group.rvs(dim=dim)
mm = np.matmul(m.T, m)

f = open("mat_orthogonal.txt", "w")
f.write(str(dim) + " " + str(dim) + "\n")
for r in range(dim):
    for c in range(dim):
        f.write(str(m[r][c]) + " ")
    f.write("\n")
f.close()