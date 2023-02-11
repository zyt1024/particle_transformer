import torch
 
def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
 
if __name__ == '__main__':
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0]])
    y = torch.tensor([[3.0, 1.0, 2.0, 5.0], [2.0, 3.0, 4.0, 6.0]])
    dist_matrix = euclidean_dist(x, y)
    print(dist_matrix)