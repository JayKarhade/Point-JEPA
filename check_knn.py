import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

from knn_cuda import KNN
"""
if transpose_mode is True, 
    ref   is Tensor [bs x nr x dim]
    query is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    ref   is Tensor [bs x dim x nr]
    query is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

knn = KNN(k=10, transpose_mode=True)

ref = torch.rand(32, 1000, 5).cuda()
query = torch.rand(32, 50, 5).cuda()
dist, indx = knn(ref, query)  # 32 x 50 x 10
print(dist[10])

# import torch

# data = torch.randn(2,5, 10)
# test = torch.randn(2,1,10)

# # data = torch.randn(32,1024, 3)
# # test = torch.randn(32,64,3)

# # print("data")
# # print(data)
# # print("test")
# # print(test)
# dist = torch.norm(data - test, dim=2, p=None)
# print(dist.shape)
# knn = dist.topk(3, largest=False)
# print(knn.indices.shape)
# print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
