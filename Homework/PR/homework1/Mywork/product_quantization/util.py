import numpy as np

def log_nb_clusters_to_np_int_type(log_nb_clusters: int) -> type:
    """
    根据聚类数的对数值选择合适的 NumPy 整数类型，用于存储压缩数据中的聚类索引。

    :param log_nb_clusters: 聚类数的对数值（以 2 为底），即 k = 2 ** log_nb_clusters。
    :return: 合适的 NumPy 整数类型，用于存储聚类索引。
    """
    # 由于聚类索引从零开始，因此考虑索引范围
    if log_nb_clusters <= 8:
        return np.uint8  # 使用 8 位无符号整数（0 到 255）
    elif log_nb_clusters <= 16:
        return np.uint16  # 使用 16 位无符号整数（0 到 65,535）
    elif log_nb_clusters <= 32:
        return np.uint32  # 使用 32 位无符号整数（0 到 4,294,967,295）
    else:
        return np.uint64  # 使用 64 位无符号整数（超大范围）

def squared_euclidean_dist(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    计算两个向量或向量数组之间的平方欧几里得距离。

    :param vec1: 第一个向量或向量数组，形状为 (d,) 或 (n, d)。
    :param vec2: 第二个向量或向量数组，形状为 (d,) 或 (n, d)。
    :return: 每个向量之间的平方欧几里得距离，结果为标量或数组。
    """
    # 计算平方欧几里得距离：对两个向量差的平方求和
    return np.sum(np.square(vec2 - vec1), axis=-1)
