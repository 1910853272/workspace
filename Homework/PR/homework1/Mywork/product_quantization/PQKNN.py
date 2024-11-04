import multiprocessing
from typing import Callable
import numpy as np
from sklearn.cluster import KMeans

# 从其他模块导入的实用函数
from .util import log_nb_clusters_to_np_int_type, squared_euclidean_dist


class ProductQuantizationKNN:
    """
    使用乘积量化（Product Quantization, PQ）加速最近邻搜索的 k 近邻（k-NN）算法实现。
    该方法将特征向量分成多个部分，并使用 KMeans 聚类进行压缩，以加速距离计算。
    """

    def __init__(self, n: int, c: int):
        """
        初始化 PQKNN 对象。

        :param n: 将原始数据分成的子向量个数。
        :param c: 用于确定 KMeans 聚类数量的指数，即 k = 2**c。
        """
        self.n = n  # 子向量的个数
        self.k = 2 ** c  # 每个子向量的聚类数
        self.int_type = log_nb_clusters_to_np_int_type(c)  # 用于存储压缩数据的整数类型
        self.subvector_centroids = {}  # 用于存储每个分区的质心的字典
        self.random_state = 420  # KMeans 的随机种子，确保结果可复现

    def _get_data_partition(self, train_data, partition_idx):
        """
        根据分区索引获取训练数据的特定分区。

        :param train_data: 完整的训练数据。
        :param partition_idx: 要获取的分区索引。
        :return: 对应分区的训练数据切片。
        """
        partition_start = partition_idx * self.partition_size
        partition_end = (partition_idx + 1) * self.partition_size
        train_data_partition = train_data[:, partition_start:partition_end]
        return train_data_partition

    def _compress_partition(self, partition_idx: int, train_data_partition):
        """
        使用 KMeans 对数据分区进行压缩。

        :param partition_idx: 分区索引。
        :param train_data_partition: 要压缩的数据分区。
        :return: 分区索引、压缩后的数据和聚类质心。
        """
        km = KMeans(n_clusters=self.k, n_init=1, random_state=self.random_state)  # 使用 KMeans 聚类
        compressed_data_partition = km.fit_predict(train_data_partition).astype(self.int_type)
        partition_centroids = km.cluster_centers_
        return partition_idx, compressed_data_partition, partition_centroids

    def compress(self, train_data: np.ndarray, train_labels: np.ndarray):
        """
        使用产品量化对给定的训练数据进行压缩。

        :param train_data: 2D 数组，每一行代表一个训练样本。
        :param train_labels: 与训练样本对应的 1D 标签数组。
        """
        nb_samples = len(train_data)
        assert nb_samples == len(train_labels), "样本数量与标签数量不匹配。"
        self.train_labels = train_labels  # 存储标签
        self.compressed_data = np.empty(shape=(nb_samples, self.n), dtype=self.int_type)  # 初始化压缩数据

        d = len(train_data[0])  # 数据的维度
        self.partition_size = d // self.n  # 每个子向量分区的大小

        # 使用多进程并行化压缩各个分区
        with multiprocessing.Pool() as pool:
            params = [(partition_idx, self._get_data_partition(train_data, partition_idx)) for partition_idx in range(self.n)]
            kms = pool.starmap(self._compress_partition, params)  # 并行运行压缩
            # 存储压缩数据和质心
            for (partition_idx, compressed_data_partition, partition_centroids) in kms:
                self.compressed_data[:, partition_idx] = compressed_data_partition
                self.subvector_centroids[partition_idx] = partition_centroids

    def predict_single_sample(self, test_sample: np.ndarray, nearest_neighbors: int,
                              calc_dist: Callable[[np.ndarray, np.ndarray], np.ndarray] = squared_euclidean_dist):
        """
        使用产品量化 k-NN 算法预测单个样本的标签。

        :param test_sample: 要分类的 1D 样本数组。
        :param nearest_neighbors: k 近邻中考虑的邻居数。
        :param calc_dist: 计算距离的函数，默认为平方欧几里得距离。
        :return: 预测的标签。
        """
        assert hasattr(self, 'compressed_data') and hasattr(self, 'train_labels'), \
            "没有可用的压缩数据，无法进行 k-NN 搜索。"

        # 计算测试样本与每个分区质心之间的距离
        distances = np.empty(shape=(self.k, self.n), dtype=np.float64)
        for partition_idx in range(self.n):
            partition_start = partition_idx * self.partition_size
            partition_end = (partition_idx + 1) * self.partition_size
            test_sample_partition = test_sample[partition_start:partition_end]
            centroids_partition = self.subvector_centroids[partition_idx]
            distances[:, partition_idx] = calc_dist(test_sample_partition, centroids_partition)

        # 计算所有训练样本的近似距离
        nb_stored_samples = len(self.compressed_data)
        distance_sums = np.zeros(shape=nb_stored_samples)
        for partition_idx in range(self.n):
            distance_sums += distances[:, partition_idx][self.compressed_data[:, partition_idx]]

        # 找到 k 个最近的邻居
        indices = np.argpartition(distance_sums, nearest_neighbors)
        labels = self.train_labels[indices][:nearest_neighbors]
        unique_labels, counts = np.unique(labels, return_counts=True)

        # 确定出现频率最高的标签
        if len(unique_labels) == 1:
            return unique_labels[0]  # 只有一个唯一的标签

        # 按频率排序，遇到平局时根据距离打破
        sorted_idxs = np.argsort(counts)[::-1]
        unique_labels = unique_labels[sorted_idxs]
        counts = counts[sorted_idxs]

        if counts[0] != counts[1]:  # 无平局
            return unique_labels[0]

        # 如果出现平局，选择总距离最小的标签
        max_count = counts[0]
        idx = 0
        min_distance = float('inf')
        selected_label = None
        while idx < len(unique_labels) and counts[idx] == max_count:
            label = unique_labels[idx]
            label_indices = np.where(labels == label)
            summed_distance = np.sum(distance_sums[indices[label_indices]])
            if summed_distance < min_distance:
                selected_label = label
                min_distance = summed_distance
            idx += 1
        return selected_label

    def predict(self, test_data: np.ndarray, nearest_neighbors: int,
                calc_dist: Callable[[np.ndarray, np.ndarray], np.ndarray] = squared_euclidean_dist) -> np.ndarray:
        """
        使用产品量化 k-NN 算法预测一组测试样本的标签。

        :param test_data: 2D 数组，每一行是一个待分类的样本。
        :param nearest_neighbors: k 近邻中考虑的邻居数。
        :param calc_dist: 计算距离的函数，默认为平方欧几里得距离。
        :return: 预测的标签数组。
        """
        assert test_data.ndim == 2, "test_data 必须是一个 2D 数组。"
        # 对于大型数据集使用多进程
        if len(test_data) > 2000:
            with multiprocessing.Pool() as pool:
                params = [(test_sample, nearest_neighbors, calc_dist) for test_sample in test_data]
                preds = pool.starmap(self.predict_single_sample, params)
        else:
            preds = [self.predict_single_sample(row, nearest_neighbors, calc_dist) for row in test_data]
        return np.array(preds)
