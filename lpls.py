import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csgraph
import open3d as o3d

def visualazition(np_matrix):

    # 可以根据实际情况对smoothed_point_cloud进行可视化或者其他进一步处理
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_matrix)
    o3d.visualization.draw_geometries([point_cloud])

def Laplace(source_data,n_neighbors,shrinkage_factor):
    point_cloud = source_data
    # 构建邻接关系
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)
    # 构建邻接矩阵
    adjacency_matrix = np.zeros((len(point_cloud), len(point_cloud)))
    for i, point_indices in enumerate(indices):
        adjacency_matrix[i, point_indices] = 1
        adjacency_matrix[point_indices, i] = 1

    # 计算拉普拉斯矩阵
    laplacian_matrix = csgraph.laplacian(adjacency_matrix, normed=False)

    # 对拉普拉斯矩阵进行收缩操作

    shrinked_laplacian = laplacian_matrix + shrinkage_factor * np.identity(len(point_cloud))

    # 从收缩后的拉普拉斯矩阵中恢复平滑化后的点云数据
    smoothed_point_cloud = np.linalg.pinv(shrinked_laplacian).dot(point_cloud)
    print(np.asarray(smoothed_point_cloud).shape)
    return smoothed_point_cloud

def point2numpy(piont):
    xyz_load = np.asarray(point.points)
    return  xyz_load

# 这里的smoothed_point_cloud即为平滑化后的点云数据
#source_data = np.random.rand(1000, 3)  # 100x3
point = o3d.io.read_point_cloud("light.pcd")

source_data = point2numpy(point)
print(source_data)
n_neighbors = 100  # 选择每个点的邻居数
shrinkage_factor = 0.9  # 设定收缩因子
smoothed_point_cloud = Laplace(source_data,n_neighbors,shrinkage_factor)
visualazition(smoothed_point_cloud)