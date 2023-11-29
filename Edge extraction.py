import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
# 读取点云文件
def read_pointcloud(file_path):
    extension = file_path.split(".")[-1].lower()

    if extension == "ply":
        return o3d.io.read_point_cloud(file_path)
    elif extension == "pcd":
        return o3d.io.read_point_cloud(file_path)
    elif extension == "obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        return mesh.sample_points_uniformly(number_of_points=1000000)  # 将OBJ文件转换为点云采样
    elif extension == "txt":
        data = np.loadtxt(file_path, delimiter=" ", skiprows=1)  # 跳过第一行标题行
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前三列为坐标
        if data.shape[1] >= 6:  # 至少包含6列数据，才认为有RGB颜色信息
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)  # 第4列到第6列为RGB颜色信息，除以255以将值归一化
        if data.shape[1] >= 7:  # 至少包含7列数据，才认为有常量标签信息
            pcd.labels = data[:, 6]  # 第7列为常量标签信息
        return pcd
    else:
        raise ValueError("Unsupported point cloud format")

# 沿边界裁剪骨架点集
def clip_skeleton_by_boundary(skeleton, bbox):

    print("->正在将骨架点集沿边界裁剪...")
    bbox_min, bbox_max = bbox.get_min_bound(), bbox.get_max_bound()
    bbox_center, bbox_extent = bbox.get_center(), bbox.get_max_bound() - bbox.get_min_bound()
    bbox_extent *= 1.1  # 这里为了让边界稍微扩大一点，避免误差

    cropped_skeleton = o3d.geometry.PointCloud()
    cropped_indices = []
    for i in range(len(skeleton.points)):
        p = skeleton.points[i]
        if (p[0] >= bbox_min[0] and p[1] >= bbox_min[1] and p[2] >= bbox_min[2] and
            p[0] <= bbox_max[0] and p[1] <= bbox_max[1] and p[2] <= bbox_max[2]):
            cropped_indices.append(i)

    cropped_skeleton.points = o3d.utility.Vector3dVector(np.asarray(skeleton.points)[cropped_indices])
    cropped_skeleton.paint_uniform_color([1, 0.706, 0])

    clipped_skeleton = cropped_skeleton.crop(bbox)
    clipped_skeleton.translate(-bbox_center)

    print("->成功将骨架点集沿边界裁剪。")
    return clipped_skeleton

def compute_shortest_path(point_cloud, skeleton, start, end):
    def calculate_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    def simple_dijkstra(graph, start):
        n_vertices = len(graph)
        dist = np.full(n_vertices, np.inf)
        dist[start] = 0
        prev = np.arange(n_vertices)
        visited = np.zeros(n_vertices, dtype=bool)

        for _ in range(n_vertices):
            v = np.argmin(dist + np.where(visited, np.inf, 0))
            visited[v] = True
            for w in range(n_vertices):
                if not visited[w] and graph[v, w] > 0 and dist[v] + graph[v, w] < dist[w]:
                    dist[w] = dist[v] + graph[v, w]
                    prev[w] = v

        return dist, prev

    def simple_optimize_path(point_cloud, skeleton, path):
        # 这里假设直接返回不进行优化
        return path

    # 构建骨架线段列表
    lines = []
    for i in range(len(skeleton) - 1):
        lines.append([i, i + 1])

    # 计算骨架线段的长度
    line_lengths = [calculate_distance(skeleton[line[0]], skeleton[line[1]]) for line in lines]

    # 构建图模型
    n_vertices = len(skeleton)
    graph = np.zeros((n_vertices, n_vertices))
    for i, (u, v) in enumerate(lines):
        length = line_lengths[i]
        graph[u, v] = length
        graph[v, u] = length

    # 计算起点到终点的最短路径
    start_idx = np.argmin(np.linalg.norm(skeleton - start, axis=1))
    end_idx = np.argmin(np.linalg.norm(skeleton - end, axis=1))
    dist_matrix, _ = simple_dijkstra(graph, start_idx)
    path = [end_idx]
    while path[-1] != start_idx:
        path.append(int(_[path[-1]]))
    path.reverse()

    # 对路径进行优化
    path = simple_optimize_path(point_cloud, skeleton, path)

    # 可视化路径
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='.')
    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], c='r')

    path_points = skeleton[path]
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], c='g', linewidth=2)

    ax.scatter(start[0], start[1], start[2], c='r', marker='o')
    ax.scatter(end[0], end[1], end[2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return skeleton[path]

def compute_path(point_cloud, skeleton, start, end):
    """
    计算两点之间的最短路径
    1.构建骨架线段列表：首先遍历骨架点集，将相邻的点两两组合成线段，形成线段列表。

    2.计算骨架线段的长度：利用numpy中的函数计算每个线段的长度，并存储在line_lengths数组中。

    3.构建图模型：根据骨架线段的连接关系和长度，构建了一个表示图的矩阵graph，其中每个元素graph[i][j]表示点i到点j的距离，如果i和j之间没有连接，则距离为0。

    4.计算最短路径：利用Dijkstra算法计算起点到终点的最短路径。首先找到最接近起点和终点的骨架点的索引start_idx和end_idx，然后利用Dijkstra算法计算出起点到各个点的最短距离，并记录路径上的前驱节点，最后根据前驱节点信息回溯得到起点到终点的最短路径。

    5.路径优化：最后对计算出的路径进行优化，可能是去除冗余点或者平滑路径，不过优化的具体方法并没有在提供的代码中给出。

    :param point_cloud: Open3D点云对象
    :param skeleton: 骨架点集
    :param start: 起始点坐标
    :param end: 终止点坐标
    :return: 路径点集
    """
    # 构建骨架线段列表
    lines = []
    for i in range(len(skeleton)-1):
        lines.append([i, i+1])
    # 计算骨架线段的长度
    print(lines)
    line_lengths = np.linalg.norm(skeleton[lines[:, 0]] - skeleton[lines[:, 1]], axis=1)
    # 构建图模型
    n_vertices = len(skeleton)
    n_edges = len(lines)
    graph = np.zeros((n_vertices, n_vertices))
    for i in range(n_edges):
        u, v = lines[i]
        length = line_lengths[i]
        graph[u, v] = length
        graph[v, u] = length
    # 计算起点到终点的最短路径
    start_idx = np.argmin(np.linalg.norm(skeleton - start, axis=1))
    end_idx = np.argmin(np.linalg.norm(skeleton - end, axis=1))
    dist_matrix, _ = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=start_idx, return_predecessors=True)
    path = [end_idx]
    while path[-1] != start_idx:
        path.append(int(_[path[-1]]))
    path.reverse()

    # 对路径进行优化
    # 未优化！
    #path = optimize_path(point_cloud, skeleton, path)

    return skeleton[path]

def optimize_path(point_cloud, skeleton, path):
    """
    对路径进行优化
    :param point_cloud: Open3D点云对象
    :param skeleton: 骨架点集
    :param path: 路径点集
    :return: 优化后的路径点集
    """
    path = smooth_path(skeleton, path)
    path = simplify_path(point_cloud, path)
    return path

def smooth_path(skeleton, path):
    """
    对路径进行平滑处理
    :param skeleton: 骨架点集
    :param path: 路径点集
    :return: 平滑后的路径点集
    """


    path = np.asarray(skeleton[path])
    x, y, z = path[:, 0], path[:, 1], path[:, 2]
    t = np.arange(len(path))
    f = interp1d(t, path, kind='cubic')
    new_t = np.linspace(0, len(path)-1, len(path)*5)
    new_path = np.stack([f(new_t)[:, 0], f(new_t)[:, 1], f(new_t)[:, 2]], axis=1)
    new_path_idx = np.argmin(np.linalg.norm(skeleton[:, None] - new_path[None], axis=-1), axis=-1)
    return new_path_idx.tolist()


def simplify_path(point_cloud, path):
    """
    对路径进行简化处理
    :param point_cloud: Open3D点云对象
    :param path: 路径点集
    :return: 简化后的路径点集
    """
    keep_points = np.ones(len(path), dtype=bool)
    while True:
        # 计算当前路径的总长度
        total_length = 0
        for i in range(len(path)-1):
            total_length += np.linalg.norm(point_cloud[path[i]] - point_cloud[path[i+1]])
        # 依次判断相邻的三个点是否可以合并
        for i in range(1, len(path)-1):
            if not keep_points[i]:
                continue
            new_path = path.copy()
            new_path.pop(i)
            new_total_length = 0
            for j in range(len(new_path)-1):
                new_total_length += np.linalg.norm(point_cloud[new_path[j]] - point_cloud[new_path[j+1]])
            if new_total_length < total_length:
                path = new_path
                keep_points[i] = False
                break
        else:
            break
    return path


if __name__ == '__main__':
#     # 读取点云文件
#     point_cloud_file = r"G:\text\demo\cloud_point\results\平滑后.txt"
#     point_cloud = read_point_cloud(point_cloud_file)
#
#     # 对点云进行下采样
#     point_cloud = downsample_point_cloud(point_cloud, voxel_size=0.02)
#
#     # 将点云转换成网格模型
#     mesh = mesh_point_cloud(point_cloud, radius=0.05)
#
#     # 计算网格模型的Medial Axis Transform（MAT）
#     skeleton = compute_medial_axis(mesh)
#
#     # 将骨架点集裁剪到点云边界内
#     bbox = np.asarray(point_cloud.get_axis_aligned_bounding_box().get_box_points())
#     skeleton = clip_skeleton(skeleton, bbox)
#
#     # 计算起点和终点
#     start = np.array([0.5, -0.5, 0.0])
#     end = np.array([-0.5, 0.5, 0.0])
#
#     # 计算最短路径
#     path = compute_path(point_cloud, skeleton, start, end)
#     print("最短路径点集：", path)
    # 定义点云文件路径
    pointcloud_path = r"D:\UavHyperSpectrum\HyperSpectrum\统计滤波后.txt"

    # 读取点云文件
    pcd = read_pointcloud(pointcloud_path)

    # 将点云转换为numpy数组
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print("->正在估计法线并可视化...")
    radius = 0.01   # 搜索半径
    max_nn = 30     # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    print("->正在打印前10个点的法向量...")
    print(np.asarray(pcd.normals)[:10, :])

    print("->正在将点云转换为网格模型...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0)

    # 根据法线调整顶点位置
    mesh.compute_vertex_normals()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + np.asarray(mesh.vertex_normals) * radius)

    # 三角面片列表
    triangles = np.asarray(mesh.triangles)

    print("->成功将点云转换为网格模型。")
    # 可视化网格模型
    #o3d.visualization.draw_geometries([mesh])


    # 进一步处理骨架点集
    simplified_mesh = mesh.simplify_quadric_decimation(100000)
    simplified_mesh = simplified_mesh.filter_smooth_taubin(number_of_iterations=5)
    simplified_mesh.compute_vertex_normals()

    # 将骨架点集转换回点云数据
    skeleton_points = np.asarray(simplified_mesh.vertices)

    # 创建新的点云对象
    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)

    # 裁剪骨架点集
    clipped_skeleton = clip_skeleton_by_boundary(skeleton_pcd, skeleton_pcd.get_axis_aligned_bounding_box())

    # 可视化结果
    #o3d.visualization.draw_geometries([clipped_skeleton])

    point_cloud=np.asarray(pcd.points)
    skeleton=np.asarray(simplified_mesh.vertices)
    start=0
    end=2000
    path = compute_path(point_cloud, skeleton, start, end)
    print("最短路径点集：", path)