import numpy as np

levels = 4
kmax = 2
n = kmax**levels
m = 6

seq = np.random.randint(0,60, m)
predictions = []
inputs = np.zeros((n,m))
inputs[:] = seq
next_ids = []
for i in range(levels):
    spacing = n/(kmax**(i))
    inputs[:,:-1] = inputs[:, 1:]
    next_idx = [k for k in range(n) if k % spacing == 0][1:]
    next_idx.append(n)
    prev_idx = 0
    for idx in next_idx:
        kargmax = np.random.randint(0,60, kmax).tolist()
        vec = np.repeat(kargmax,spacing/kmax)
        inputs[prev_idx:idx,-1] = vec
        prev_idx = idx





