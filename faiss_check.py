import faiss
import faiss.contrib.torch_utils

import torch
import numpy as np

xb = torch.randn(1024,10).cuda()
xq = torch.randn(32,10).cuda()

# index = faiss.index_factory(10, "Flat")
index = faiss.IndexFlatL2(10)

res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0 , index)

# index.train(xb)
index.add(xb)
distances, neighbors = index.search(xq, 20)
print(neighbors)