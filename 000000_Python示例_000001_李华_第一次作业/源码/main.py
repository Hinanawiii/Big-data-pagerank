import os
import time
import heapq
import numpy as np
import gc
from threading import Thread
from queue import Queue
from collections import defaultdict

# 延迟导入，只在需要时导入
# from scipy import sparse
# import networkx as nx  
# import matplotlib.pyplot as plt

#程序默认的输入文件路径为"Data.txt"
#程序默认的输出文件路径为"实验结果/Res.txt"
#都基于当前工作目录，请根据需要修改


class PageRankBase:
    """PageRank算法的基类，定义通用方法和接口"""
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100):
        """
        初始化PageRank算法基类
        
        参数:
            data_path: 图数据文件路径
            teleport: 传送参数，表示按照链接随机浏览的概率
            epsilon: 收敛阈值
            max_iter: 最大迭代次数
        """
        self.data_path = data_path
        self.teleport = teleport
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.name = "基础PageRank"
        
    def load_graph(self):
        """从文件加载图数据（子类需实现）"""
        raise NotImplementedError("子类必须实现load_graph方法")
    
    def compute_pagerank(self):
        """计算PageRank值（子类需实现）"""
        raise NotImplementedError("子类必须实现compute_pagerank方法")
    
    def save_results(self, pr_values, output_file="Res.txt", top_n=100):
        """保存前top_n个节点的PageRank值"""
        # 检查输出路径是否包含目录
        output_dir = os.path.dirname(output_file)
        if output_dir:  # 如果有目录部分，确保目录存在
            os.makedirs(output_dir, exist_ok=True)
        
        top_nodes = heapq.nlargest(top_n, pr_values.items(), key=lambda x: x[1])
        
        with open(output_file, 'w') as f:
            for node, score in top_nodes:
                f.write(f"{node} {score:.10f}\n")
        
        print(f"前{top_n}个节点的PageRank值已保存至: {output_file}")
        
        print("\n前10个节点的PageRank值:")
        for i, (node, score) in enumerate(top_nodes[:min(10, len(top_nodes))], 1):
            print(f"{i}. 节点ID: {node}, PageRank值: {score:.10f}")
        
        return top_nodes


class BasicPageRank(PageRankBase):
    """使用基本邻接表实现的PageRank算法"""
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100):
        super().__init__(data_path, teleport, epsilon, max_iter)
        self.in_edges = defaultdict(list)  # 入度邻接表
        self.out_degrees = defaultdict(int)  # 出度表
        self.nodes = set()  # 所有节点
        self.name = "基础邻接表PageRank"
    
    def load_graph(self):
        """从文件加载图数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        
        print(f"正在加载数据: {self.data_path}")
        edges = set()  # 使用集合去重
        
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    edges.add((from_node, to_node))
        
        # 构建图结构
        for from_node, to_node in edges:
            self.in_edges[to_node].append(from_node)  # 入度表
            self.out_degrees[from_node] += 1  # 出度表
            self.nodes.add(from_node)
            self.nodes.add(to_node)
        
        # 计算节点总数
        self.n_nodes = len(self.nodes)
        print(f"数据加载完成，共有{len(edges)}条唯一边，{self.n_nodes}个节点")
    
    def compute_pagerank(self):
        """计算PageRank值"""
        print(f"\n===== {self.name} =====")
        print(f"开始计算PageRank，参数: teleport={self.teleport}, epsilon={self.epsilon}, max_iter={self.max_iter}")
        start_time = time.time()
        
        self.load_graph()
        
        # 初始化PageRank值
        pr_old = {node: 1.0 / self.n_nodes for node in self.nodes}
        
        # 迭代计算
        for iteration in range(1, self.max_iter + 1):
            # 处理dead ends导致的PR值流失
            pr_sum = sum(pr_old[node] for node in self.nodes if self.out_degrees[node] == 0)
            
            # 初始化新的PR值
            pr_new = {node: (1.0 - self.teleport) / self.n_nodes for node in self.nodes}
            
            # 计算每个节点的新PR值
            for node in self.nodes:
                # 来自入边贡献的PR值
                incoming_pr = 0
                for src in self.in_edges[node]:
                    if self.out_degrees[src] > 0:  # 只考虑有出度的节点
                        incoming_pr += pr_old[src] / self.out_degrees[src]
                
                # 应用teleport参数和处理dead-ends 和 蜘蛛陷阱
                pr_new[node] += self.teleport * (incoming_pr + pr_sum / self.n_nodes)
            
            # 归一化PR值
            pr_sum = sum(pr_new.values())
            for node in pr_new:
                pr_new[node] /= pr_sum
            
            # 计算最大变化值，用于判断收敛
            max_diff = max(abs(pr_new[node] - pr_old[node]) for node in self.nodes)
            
            # 打印迭代信息
            if iteration % 10 == 0 or iteration == 1:
                print(f"迭代 {iteration}: 最大变化={max_diff:.10f}")
            
            # 检查收敛条件
            if max_diff < self.epsilon:
                print(f"收敛于迭代 {iteration}，最大变化={max_diff:.10f}")
                break
            
            # 更新PR值
            pr_old = pr_new.copy()
        
        if iteration == self.max_iter:
            print(f"达到最大迭代次数 {self.max_iter}，最大变化={max_diff:.10f}")
        
        elapsed_time = time.time() - start_time
        print(f"计算完成，耗时 {elapsed_time:.2f} 秒")
        
        return pr_old


class MinimalMemoryPageRank(PageRankBase):
    """使用最小内存消耗的PageRank实现"""
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100):
        super().__init__(data_path, teleport, epsilon, max_iter)
        self.nodes = []
        self.node_map = {}
        self.n_nodes = 0
        self.out_degrees = None
        self.dangling_nodes = None
        self.name = "最小内存PageRank"
    
    def load_graph(self):
        """使用最小内存加载图数据"""
        print("加载图数据...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        
        node_set = set()
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node, to_node = int(parts[0]), int(parts[1])
                    node_set.add(from_node)
                    node_set.add(to_node)
        
        self.nodes = sorted(node_set)
        self.n_nodes = len(self.nodes)
        self.node_map = {node: idx for idx, node in enumerate(self.nodes)}
        
        self.out_degrees = np.zeros(self.n_nodes, dtype=np.int16)
        
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node, to_node = int(parts[0]), int(parts[1])
                    if from_node in self.node_map:
                        src_idx = self.node_map[from_node]
                        self.out_degrees[src_idx] += 1
        
        self.dangling_nodes = np.where(self.out_degrees == 0)[0]
        
        print(f"图数据加载完成: {self.n_nodes}个节点, 死端节点: {len(self.dangling_nodes)}个")
        
        del node_set
        gc.collect()
    
    def compute_pagerank(self):
        """使用最小内存计算PageRank"""
        print(f"\n===== {self.name} =====")
        print(f"计算PageRank: teleport={self.teleport}, epsilon={self.epsilon}")
        start_time = time.time()
        
        self.load_graph()
        
        v = np.ones(self.n_nodes, dtype=np.float32) / self.n_nodes
        v_new = np.zeros(self.n_nodes, dtype=np.float32)
        
        for iteration in range(1, self.max_iter + 1):
            dangling_sum = np.sum(v[self.dangling_nodes])
            dangling_contribution = self.teleport * dangling_sum / self.n_nodes
            
            v_new.fill((1.0 - self.teleport) / self.n_nodes + dangling_contribution)
            
            with open(self.data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        from_node, to_node = int(parts[0]), int(parts[1])
                        
                        if from_node in self.node_map and to_node in self.node_map:
                            src_idx = self.node_map[from_node]
                            dst_idx = self.node_map[to_node]
                            
                            if self.out_degrees[src_idx] > 0:
                                v_new[dst_idx] += self.teleport * v[src_idx] / self.out_degrees[src_idx]
            
            v_sum = np.sum(v_new)
            if v_sum > 0:
                v_new /= v_sum
            
            max_diff = np.max(np.abs(v_new - v))
            
            if iteration % 10 == 0 or iteration == 1:
                print(f"迭代 {iteration}: 最大变化={max_diff:.10f}")
            
            if max_diff < self.epsilon:
                print(f"收敛于迭代 {iteration}，最大变化={max_diff:.10f}")
                break
            
            v, v_new = v_new, v
        
        if iteration == self.max_iter:
            print(f"达到最大迭代次数 {self.max_iter}，最大变化={max_diff:.10f}")
        
        elapsed_time = time.time() - start_time
        print(f"计算完成，耗时 {elapsed_time:.2f} 秒")
        
        result = {self.nodes[i]: float(v[i]) for i in range(self.n_nodes)}
        return result


class SparseMatrixPageRank(PageRankBase):
    """使用稀疏矩阵实现的PageRank算法"""
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100):
        super().__init__(data_path, teleport, epsilon, max_iter)
        self.node_map = {}  # 节点ID到索引的映射
        self.reverse_map = {}  # 索引到节点ID的映射
        self.n_nodes = 0
        self.M = None  # 稀疏矩阵
        self.dangling_nodes = None  # 出度为0的节点
        self.name = "稀疏矩阵PageRank"
    
    def load_graph(self):
        from scipy import sparse
        """从文件加载图数据并构建稀疏矩阵"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        
        print(f"正在加载数据: {self.data_path}")
        
        # 第一次扫描：收集唯一节点
        unique_nodes = set()
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    unique_nodes.add(from_node)
                    unique_nodes.add(to_node)
        
        # 创建节点映射
        for idx, node_id in enumerate(sorted(unique_nodes)):
            self.node_map[node_id] = idx
            self.reverse_map[idx] = node_id
        
        self.n_nodes = len(unique_nodes)
        
        # 第二次扫描：计算出度和构建稀疏矩阵数据
        out_degrees = np.zeros(self.n_nodes, dtype=np.int32)
        rows, cols, data = [], [], []
        unique_edges = set()
        
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    src_idx = self.node_map[from_node]
                    dst_idx = self.node_map[to_node]
                    edge = (src_idx, dst_idx)
                    
                    if edge not in unique_edges:
                        unique_edges.add(edge)
                        out_degrees[src_idx] += 1
        
        # 构建稀疏矩阵数据
        for src_idx, dst_idx in unique_edges:
            if out_degrees[src_idx] > 0:
                rows.append(dst_idx)
                cols.append(src_idx)
                data.append(1.0 / out_degrees[src_idx])
        
        # 构建稀疏矩阵
        self.M = sparse.csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        
        # 标记死端节点（出度为0的节点）
        self.dangling_nodes = np.where(out_degrees == 0)[0]
        
        print(f"数据加载完成，共有{len(unique_edges)}条唯一边，{self.n_nodes}个节点")
        print(f"稀疏矩阵构建完成，包含{len(data)}个非零元素")
        print(f"死端节点数量: {len(self.dangling_nodes)}")
        
        del unique_nodes, unique_edges
        gc.collect()
    
    def compute_pagerank(self):
        """使用稀疏矩阵计算PageRank"""
        print(f"\n===== {self.name} =====")
        print(f"开始使用稀疏矩阵计算PageRank，参数: teleport={self.teleport}, epsilon={self.epsilon}")
        start_time = time.time()
        
        self.load_graph()
        
        # 初始化PageRank向量
        v = np.ones(self.n_nodes) / self.n_nodes
        
        # 迭代计算
        for iteration in range(1, self.max_iter + 1):
            # 计算死端贡献
            dangling_sum = sum(v[i] for i in self.dangling_nodes) / self.n_nodes
            
            # 计算新的PR值 (包含随机传送和死端处理)
            v_new = self.teleport * self.M.dot(v) + \
                   self.teleport * dangling_sum + \
                   (1.0 - self.teleport) / self.n_nodes
            
            # 归一化
            v_new = v_new / np.sum(v_new)
            
            # 计算误差
            err = np.linalg.norm(v_new - v, 1)
            
            # 打印迭代信息
            if iteration % 10 == 0 or iteration == 1:
                print(f"迭代 {iteration}: 误差={err:.10f}")
            
            # 检查收敛条件
            if err < self.epsilon:
                print(f"收敛于迭代 {iteration}，误差={err:.10f}")
                break
            
            # 更新向量
            v = v_new
        
        if iteration == self.max_iter:
            print(f"达到最大迭代次数 {self.max_iter}，误差={err:.10f}")
        
        elapsed_time = time.time() - start_time
        print(f"计算完成，耗时 {elapsed_time:.2f} 秒")
        
        # 转换回原始节点ID
        result = {self.reverse_map[i]: v[i] for i in range(self.n_nodes)}
        return result



class BlockMatrixPageRank(SparseMatrixPageRank):
    """使用分块矩阵优化的PageRank算法"""
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100, block_size=1000):
        super().__init__(data_path, teleport, epsilon, max_iter)
        self.block_size = block_size
        self.name = "分块矩阵PageRank"
    
    def compute_pagerank(self):
        """使用分块矩阵优化的PageRank计算"""
        print(f"\n===== {self.name} =====")
        print(f"开始使用分块矩阵计算PageRank，参数: teleport={self.teleport}, epsilon={self.epsilon}, block_size={self.block_size}")
        start_time = time.time()
        
        self.load_graph()
        
        # 初始化PageRank向量
        v = np.ones(self.n_nodes) / self.n_nodes
        
        # 确定块数
        num_blocks = (self.n_nodes + self.block_size - 1) // self.block_size
        
        # 迭代计算
        for iteration in range(1, self.max_iter + 1):
            # 计算死端贡献
            dangling_sum = sum(v[i] for i in self.dangling_nodes) / self.n_nodes
            
            # 初始化新向量（包含随机传送和死端处理）
            v_new = np.ones(self.n_nodes) * ((1.0 - self.teleport) / self.n_nodes + self.teleport * dangling_sum)
            
            # 分块计算矩阵乘法
            for block in range(num_blocks):
                start_idx = block * self.block_size
                end_idx = min((block + 1) * self.block_size, self.n_nodes)
                
                # 提取当前块的矩阵
                M_block = self.M[start_idx:end_idx, :]
                
                # 计算当前块的PR值贡献
                v_block = M_block.dot(v)
                
                # 更新相应位置
                v_new[start_idx:end_idx] += self.teleport * v_block
            
            # 归一化
            v_new = v_new / np.sum(v_new)
            
            # 计算误差
            err = np.linalg.norm(v_new - v, 1)
            
            # 打印迭代信息
            if iteration % 10 == 0 or iteration == 1:
                print(f"迭代 {iteration}: 误差={err:.10f}")
            
            # 检查收敛条件
            if err < self.epsilon:
                print(f"收敛于迭代 {iteration}，误差={err:.10f}")
                break
            
            # 更新向量
            v = v_new
        
        if iteration == self.max_iter:
            print(f"达到最大迭代次数 {self.max_iter}，误差={err:.10f}")
        
        elapsed_time = time.time() - start_time
        print(f"计算完成，耗时 {elapsed_time:.2f} 秒")
        
        # 转换回原始节点ID
        result = {self.reverse_map[i]: v[i] for i in range(self.n_nodes)}
        return result


class CSRPageRank(PageRankBase):
    """使用CSR格式和多线程实现的PageRank算法"""
    
    __slots__ = ('data_path', 'teleport', 'epsilon', 'max_iter', 'block_size', 
                 'num_threads', 'nodes', 'n_nodes', 'out_degrees', 
                 'dangling_nodes', 'matrix_data', 'matrix_indices', 
                 'matrix_indptr', 'matrix_shapes', 'name')
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100, 
                 block_size=3200, num_threads=None):
        super().__init__(data_path, teleport, epsilon, max_iter)
        self.block_size = block_size
        self.num_threads = min(3, num_threads or max(1, os.cpu_count() - 1))
        
        self.nodes = None
        self.n_nodes = 0
        self.out_degrees = None
        self.dangling_nodes = None
        self.matrix_data = []
        self.matrix_indices = []
        self.matrix_indptr = []
        self.matrix_shapes = []
        self.name = "CSR格式PageRank"
    
    def edge_generator(self):
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    yield int(parts[0]), int(parts[1])
    
    def load_graph(self):
        print("加载图数据...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        
        node_set = set()
        edge_count = 0
        
        for src_id, dst_id in self.edge_generator():
            node_set.add(src_id)
            node_set.add(dst_id)
            edge_count += 1
        
        self.nodes = np.array(sorted(node_set), dtype=np.uint16)
        self.n_nodes = len(self.nodes)
        node_map = {node: idx for idx, node in enumerate(self.nodes)}
        
        edge_dtype = np.dtype([
            ('src', np.uint16), 
            ('dst', np.uint16), 
            ('block_id', np.uint8)
        ])
        
        edges = np.zeros(edge_count, dtype=edge_dtype)
        out_counts = np.zeros(self.n_nodes, dtype=np.uint16)
        
        idx = 0
        for src_id, dst_id in self.edge_generator():
            src_idx = node_map.get(src_id)
            dst_idx = node_map.get(dst_id)
            
            if src_idx is not None and dst_idx is not None:
                out_counts[src_idx] += 1
                block_id = dst_idx // self.block_size
                
                edges[idx] = (src_idx, dst_idx, block_id)
                idx += 1
        
        max_out = np.max(out_counts)
        degree_dtype = (np.uint8 if max_out < 255 else 
                       np.uint16 if max_out < 65535 else np.uint32)
        
        self.out_degrees = out_counts.astype(degree_dtype)
        self.dangling_nodes = np.nonzero(self.out_degrees == 0)[0].astype(np.uint16)
        
        print(f"图数据加载完成: {self.n_nodes}个节点, {edge_count}条边")
        
        del node_set, node_map
        gc.collect()
        
        return edges
    
    def build_matrix_blocks(self, edges):
        print("构建矩阵块...")
        
        n_nodes, block_size = self.n_nodes, self.block_size
        num_blocks = (n_nodes + block_size - 1) // block_size
        
        for block_id in range(num_blocks):
            start_row = block_id * block_size
            end_row = min(start_row + block_size, n_nodes)
            block_height = end_row - start_row
            
            block_mask = (edges['block_id'] == block_id) & (self.out_degrees[edges['src']] > 0)
            block_edges = edges[block_mask]
            
            if len(block_edges) > 0:
                rows = block_edges['dst'] - start_row
                cols = block_edges['src']
                
                data = np.empty(len(rows), dtype=np.float32)
                np.divide(self.teleport, self.out_degrees[cols].astype(np.float32), out=data)
                
                sorted_idx = np.lexsort((cols, rows))
                sorted_rows = rows[sorted_idx]
                sorted_cols = cols[sorted_idx]
                sorted_data = data[sorted_idx]
                
                row_counts = np.bincount(sorted_rows, minlength=block_height)
                
                indptr = np.zeros(block_height + 1, dtype=np.int32)
                indptr[1:] = row_counts
                np.cumsum(indptr, out=indptr)
                
                self.matrix_data.append(sorted_data)
                self.matrix_indices.append(sorted_cols)
                self.matrix_indptr.append(indptr)
            else:
                self.matrix_data.append(np.array([], dtype=np.float32))
                self.matrix_indices.append(np.array([], dtype=np.int32))
                self.matrix_indptr.append(np.zeros(block_height + 1, dtype=np.int32))
            
            self.matrix_shapes.append((block_height, n_nodes))
            
            if block_id % 4 == 3:
                gc.collect()
        
        print(f"矩阵块构建完成: {num_blocks}个块")
        
        del edges
        gc.collect()
        
        return True
    
    def _worker_function(self, block_id, data, indices, indptr, shape, v, result, start_row, result_queue):
        if len(data) > 0:
            n_rows = shape[0]
            local_result = np.zeros(n_rows, dtype=np.float32)
            
            for i in range(n_rows):
                row_start, row_end = indptr[i], indptr[i+1]
                if row_start != row_end:
                    local_result[i] = np.dot(data[row_start:row_end], v[indices[row_start:row_end]])
            
            result[start_row:start_row+n_rows] += local_result
        
        result_queue.put(block_id)
    
    def compute_pagerank(self):
        print(f"\n===== {self.name} =====")
        print(f"计算PageRank: 线程数={self.num_threads}")
        start_time = time.time()
        
        edges = self.load_graph()
        self.build_matrix_blocks(edges)
        
        n_nodes = self.n_nodes
        v = np.ones(n_nodes, dtype=np.float32) / n_nodes
        v_new = np.zeros(n_nodes, dtype=np.float32)
        
        teleport = self.teleport
        base_score = (1.0 - teleport) / n_nodes
        dangling_nodes = self.dangling_nodes
        epsilon = self.epsilon
        block_size = self.block_size
        
        result_queue = Queue()
        
        non_empty_blocks = [(block_id, data, indices, indptr, shape) 
                           for block_id, (data, indices, indptr, shape) 
                           in enumerate(zip(self.matrix_data, self.matrix_indices, 
                                          self.matrix_indptr, self.matrix_shapes)) 
                           if len(data) > 0]
        
        for iteration in range(1, self.max_iter + 1):
            dangling_contribution = teleport * np.sum(v[dangling_nodes]) / n_nodes
            v_new.fill(base_score + dangling_contribution)
            
            active_threads = 0
            threads = []
            
            for block_id, data, indices, indptr, shape in non_empty_blocks:
                start_row = block_id * block_size
                
                t = Thread(
                    target=self._worker_function,
                    args=(block_id, data, indices, indptr, shape, v, v_new, start_row, result_queue)
                )
                threads.append(t)
                t.start()
                active_threads += 1
                
                if active_threads >= self.num_threads:
                    result_queue.get()
                    active_threads -= 1
            
            for _ in range(active_threads):
                result_queue.get()
            
            for t in threads:
                t.join()
            
            v_sum = np.sum(v_new)
            if v_sum > 0:
                v_new /= v_sum
            
            max_diff = np.max(np.abs(v_new - v))
            
            if iteration == 1 or iteration % 10 == 0:
                print(f"迭代 {iteration}: 最大变化={max_diff:.10f}")
            
            if max_diff < epsilon:
                print(f"收敛于迭代 {iteration}")
                break
            
            v, v_new = v_new, v
            v_new.fill(0.0)
            
            if iteration % 10 == 0:
                gc.collect()
        
        if iteration == self.max_iter:
            print(f"达到最大迭代次数 {self.max_iter}")
        
        elapsed_time = time.time() - start_time
        print(f"计算完成，耗时 {elapsed_time:.2f} 秒")
        
        # 转换回原始节点ID
        result = {int(self.nodes[i]): float(v[i]) for i in range(n_nodes)}
        
        self.matrix_data = []
        self.matrix_indices = []
        self.matrix_indptr = []
        self.matrix_shapes = []
        gc.collect()
        
        return result


class NetworkXPageRank(PageRankBase):
    """使用NetworkX库实现的PageRank算法，用于结果对比"""
    
    def __init__(self, data_path="Data.txt", teleport=0.85, epsilon=1e-8, max_iter=100):
        super().__init__(data_path, teleport, epsilon, max_iter)
        self.G = None
        self.name = "NetworkX PageRank"
    
    def load_graph(self):
        """从文件加载图数据到NetworkX图对象"""
        # 动态导入NetworkX，减少常驻内存
        import networkx as nx
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        
        print("加载图数据到NetworkX...")
        self.G = nx.DiGraph()
        
        # 批量读取和添加边，减少内存使用
        batch_size = 10000
        edges_batch = []
        
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    edges_batch.append((from_node, to_node))
                    
                    # 达到批处理阈值时添加边并清理
                    if len(edges_batch) >= batch_size:
                        self.G.add_edges_from(edges_batch)
                        edges_batch = []
                        gc.collect()
            
            # 添加最后一批边
            if edges_batch:
                self.G.add_edges_from(edges_batch)
        
        print(f"NetworkX图加载完成: {self.G.number_of_nodes()}个节点, {self.G.number_of_edges()}条边")
    
    def compute_pagerank(self):
        """使用NetworkX计算PageRank"""
        # 动态导入NetworkX，减少常驻内存
        import networkx as nx
        
        print(f"\n===== {self.name} =====")
        print(f"使用NetworkX计算PageRank，参数: teleport={self.teleport}, epsilon={self.epsilon}, max_iter={self.max_iter}")
        start_time = time.time()
        
        self.load_graph()
        
        # 使用NetworkX计算PageRank
        pr = nx.pagerank(
            self.G, 
            alpha=self.teleport, 
            tol=self.epsilon, 
            max_iter=self.max_iter,
            weight=None
        )
        
        elapsed_time = time.time() - start_time
        print(f"NetworkX计算完成，耗时 {elapsed_time:.2f} 秒")
        
        # 清理内存
        self.G = None
        gc.collect()
        
        return pr


def compare_results(results_dict, top_n=10):
    """比较不同算法的PageRank结果"""
    print("\n===== PageRank结果比较 =====")
    
    if len(results_dict) <= 1:
        print("只有一个算法结果，无需比较。")
        return
    
    # 比较top_n节点的重叠
    print("\n算法结果的Top节点重叠比较:")
    algorithm_names = list(results_dict.keys())
    
    # 存储每个算法的top节点，避免重复计算
    top_nodes_dict = {}
    for name, pr_values in results_dict.items():
        top_nodes_dict[name] = heapq.nlargest(top_n, pr_values.items(), key=lambda x: x[1])
    
    # 比较算法之间的结果差异
    for i in range(len(algorithm_names)):
        for j in range(i+1, len(algorithm_names)):
            name_i = algorithm_names[i]
            name_j = algorithm_names[j]
            
            # 获取top节点
            top_set_i = {node for node, _ in top_nodes_dict[name_i]}
            top_set_j = {node for node, _ in top_nodes_dict[name_j]}
            
            # 计算重叠
            overlap = len(top_set_i & top_set_j)
            
            print(f"{name_i} vs {name_j}: Top {top_n}节点重叠率 = {overlap}/{top_n} ({overlap/top_n:.2%})")
            
            # 显示差异节点
            if overlap < top_n:
                diff_i = top_set_i - top_set_j
                diff_j = top_set_j - top_set_i
                
                if diff_i:
                    print(f"  {name_i} 独有节点: {sorted(diff_i)[:5]}{' 等' if len(diff_i) > 5 else ''}")
                if diff_j:
                    print(f"  {name_j} 独有节点: {sorted(diff_j)[:5]}{' 等' if len(diff_j) > 5 else ''}")
    
    # 可视化比较 (可选)
    if len(results_dict) > 1:
        # 获取每个算法前5名的节点
        compare_nodes = set()
        for name in algorithm_names:
            top5 = [node for node, _ in top_nodes_dict[name][:5]]
            compare_nodes.update(top5)
        
        # 打印表格，显示前几个节点的PageRank值比较
        compare_nodes = sorted(list(compare_nodes))[:10]  # 最多显示10个节点
        
        # 打印表头
        header = "节点ID".ljust(10)
        for name in algorithm_names:
            header += f" | {name[:15].ljust(15)}"
        print("\n" + header)
        print("-" * (10 + 18 * len(algorithm_names)))
        
        # 打印每个节点的PageRank值
        for node in compare_nodes:
            row = str(node).ljust(10)
            for name in algorithm_names:
                # 获取节点的PageRank值和排名
                pr_value = results_dict[name].get(node, 0)
                rank = 0
                for idx, (n, _) in enumerate(top_nodes_dict[name]):
                    if n == node:
                        rank = idx + 1
                        break
                
                if rank > 0 and rank <= top_n:
                    row += f" | {pr_value:.8f} (#{rank})"
                else:
                    row += f" | {pr_value:.8f}     "
            print(row)
    
    # 如果需要可视化，则通过函数参数控制
    try:
        # 尝试导入matplotlib，如果失败则跳过可视化
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 设置中文字体
        try:
            # 尝试使用微软雅黑字体（Windows系统上常见的中文字体）
            font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题
        except:
            print("无法设置中文字体，某些中文可能无法正确显示")
        
        # 获取共同出现在所有算法top结果中的节点
        common_nodes = set.intersection(*[set(node for node, _ in top_nodes_dict[name][:min(5, top_n)])
                                       for name in algorithm_names])
        
        if common_nodes:
            # 选择绘制的节点
            plot_nodes = sorted(list(common_nodes))[:min(5, len(common_nodes))]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 为每个算法创建一组数据
            x = range(len(plot_nodes))
            width = 0.8 / len(algorithm_names)
            
            for i, name in enumerate(algorithm_names):
                values = [results_dict[name].get(node, 0) for node in plot_nodes]
                plt.bar([p + width*i for p in x], values, width, label=name)
            
            plt.xlabel('节点ID')
            plt.ylabel('PageRank值')
            plt.title('共同Top节点的PageRank值比较')
            plt.xticks([p + width*(len(algorithm_names)-1)/2 for p in x], plot_nodes)
            plt.legend()
            
            # 保存图表
            plt.tight_layout()
            plt.savefig('实验结果/pagerank_comparison.png')
            plt.close()  # 关闭图表释放内存
            print(f"PageRank值比较图表已保存为: pagerank_comparison.png")
            
        else:
            print("算法之间的top节点没有交集，无法生成比较图表")
    except ImportError:
        print("未安装matplotlib，跳过图表生成")


def compare_teleport_values(algorithm_class, data_path, teleport_values=[0.7, 0.85, 0.95],
                           epsilon=1e-8, max_iter=100, top_n=10):
    """比较不同teleport参数值的影响，使用更少的值以减少内存消耗"""
    print(f"\n===== 不同teleport参数值的影响 ({algorithm_class.__name__}) =====")
    
    # 确保输出目录存在
    os.makedirs("实验结果", exist_ok=True)
    
    results = {}
    exec_times = []
    
    for teleport in teleport_values:
        # 强制垃圾回收
        gc.collect()
        
        start_time = time.time()
        
        if algorithm_class.__name__ == "BlockMatrixPageRank":
            algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                      epsilon=epsilon, max_iter=max_iter, block_size=1000)
        elif algorithm_class.__name__ == "CSRPageRank":
            algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                      epsilon=epsilon, max_iter=max_iter, 
                                      block_size=3200, num_threads=None)
        else:
            algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                      epsilon=epsilon, max_iter=max_iter)
        
        # 使用算法名和teleport值作为标识
        algorithm.name = f"{algorithm_class.__name__} (t={teleport})"
        pr_values = algorithm.compute_pagerank()
        
        elapsed_time = time.time() - start_time
        exec_times.append(elapsed_time)
        
        # 只保存前top_n个节点以节省内存
        top_nodes = heapq.nlargest(top_n, pr_values.items(), key=lambda x: x[1])
        results[algorithm.name] = dict(top_nodes)
        
        # 保存结果到文件，以teleport值区分
        output_file = f"实验结果/Res_t{teleport}.txt"
        algorithm.save_results(pr_values, output_file=output_file)
        
        print(f"\n===== teleport={teleport} 的前{top_n}个节点 =====")
        for i, (node, score) in enumerate(top_nodes, 1):
            print(f"{i}. 节点ID: {node}, PageRank值: {score:.10f}")
        
        # 清理内存
        algorithm = None
        pr_values = None
        gc.collect()
    
    # 比较结果
    compare_results(results, top_n=top_n)
    
    # 如果需要绘图功能，才导入matplotlib
    try:
        import matplotlib.pyplot as plt
        
        # 绘制执行时间比较图
        plt.figure(figsize=(8, 5))
        plt.plot(teleport_values, exec_times, marker='o')
        plt.xlabel('Teleport Parameter')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time with Different Teleport Values')
        plt.grid(True)
        plt.savefig('实验结果/teleport_times.png')
        plt.close()  # 关闭图表释放内存
        print(f"Teleport参数执行时间比较图已保存为: 实验结果/teleport_times.png")
    except ImportError:
        print("未安装matplotlib，跳过图表生成")
    
    return results


def benchmark_algorithms(data_path, teleport=0.85, epsilon=1e-8, max_iter=100, selected_algorithms=None):
    """对比不同PageRank算法的性能和结果"""
    print("\n===== PageRank算法基准测试 =====")
    
    # 确保输出目录存在
    os.makedirs("实验结果", exist_ok=True)
    
    # 可供选择的算法类
    available_algorithms = {
        'basic': BasicPageRank,           # 基础实现
        'minimal': MinimalMemoryPageRank, # 最小内存实现
        'sparse': SparseMatrixPageRank,   # 稀疏矩阵实现
        'block': BlockMatrixPageRank,     # 分块矩阵实现
        'csr': CSRPageRank,               # CSR格式多线程实现
        'networkx': NetworkXPageRank      # NetworkX实现，用于比较结果
    }
    
    # 如果没有指定算法，默认使用所有算法
    if not selected_algorithms:
        selected_algorithms = list(available_algorithms.keys())
    
    # 过滤出选择的算法
    algorithm_classes = [available_algorithms[name] for name in selected_algorithms 
                         if name in available_algorithms]
    
    if not algorithm_classes:
        print("未选择任何有效算法。")
        return {}
    
    results = {}
    exec_times = []
    memory_usages = []
    
    # 尝试导入内存监控模块
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_monitor_available = True
    except ImportError:
        memory_monitor_available = False
        print("未安装psutil库，无法监控内存使用")
    
    for algorithm_class in algorithm_classes:
        # 强制垃圾回收
        gc.collect()
        
        # 记录初始内存使用
        if memory_monitor_available:
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        else:
            start_memory = 0
        
        start_time = time.time()
        
        # 创建算法实例
        if algorithm_class == BlockMatrixPageRank:
            algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                      epsilon=epsilon, max_iter=max_iter, block_size=1000)
        elif algorithm_class == CSRPageRank:
            algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                      epsilon=epsilon, max_iter=max_iter, 
                                      block_size=3200, num_threads=None)
        else:
            algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                      epsilon=epsilon, max_iter=max_iter)
        
        # 计算PageRank值
        pr_values = algorithm.compute_pagerank()
        
        elapsed_time = time.time() - start_time
        
        # 记录最终内存使用
        if memory_monitor_available:
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_usage = end_memory - start_memory
        else:
            memory_usage = 0
        
        # 收集性能数据
        exec_times.append(elapsed_time)
        memory_usages.append(memory_usage)
        
        # 保存结果
        results[algorithm.name] = pr_values
        
        # 不保存结果到文件，只显示统计信息
        print(f"\n算法: {algorithm.name}")
        print(f"执行时间: {elapsed_time:.2f} 秒")
        if memory_monitor_available:
            print(f"内存增长: {memory_usage:.2f} MB")
        
        # 打印前10个节点
        top_nodes = heapq.nlargest(10, pr_values.items(), key=lambda x: x[1])
        print("前10个节点:")
        for i, (node, score) in enumerate(top_nodes, 1):
            print(f"{i}. 节点ID: {node}, PageRank值: {score:.10f}")
        
        # 每次运行结束后清理内存
        algorithm = None
        gc.collect()
    
    # 比较性能，不详细比较PageRank值
    print("\n===== 算法性能比较 =====")
    print("算法\t\t执行时间(秒)\t内存增长(MB)")
    print("-" * 50)
    
    for i, algo_class in enumerate(algorithm_classes):
        algo_name = algo_class.__name__
        print(f"{algo_name:<15}\t{exec_times[i]:.2f}\t\t{memory_usages[i]:.2f}")
    
    # 如果需要绘图，才导入matplotlib
    if len(algorithm_classes) > 1:
        try:
            import matplotlib.pyplot as plt
            
            # 绘制执行时间比较图
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(algorithm_classes)), exec_times)
            plt.xlabel('Algorithm')
            plt.ylabel('Execution Time (seconds)')
            plt.title('PageRank Algorithm Execution Time Comparison')
            plt.xticks(range(len(algorithm_classes)), [algo.__name__ for algo in algorithm_classes], rotation=45)
            plt.tight_layout()
            plt.savefig('实验结果/algorithm_times.png')
            plt.close()  # 关闭图形以释放内存
            print("算法执行时间比较图已保存为: 实验结果/algorithm_times.png")
            
            # 如果有内存监控数据，绘制内存使用比较图
            if memory_monitor_available:
                plt.figure(figsize=(10, 5))
                plt.bar(range(len(algorithm_classes)), memory_usages)
                plt.xlabel('Algorithm')
                plt.ylabel('Memory Growth (MB)')
                plt.title('PageRank Algorithm Memory Usage Comparison')
                plt.xticks(range(len(algorithm_classes)), [algo.__name__ for algo in algorithm_classes], rotation=45)
                plt.tight_layout()
                plt.savefig('实验结果/algorithm_memory.png')
                plt.close()
                print("算法内存使用比较图已保存为: 实验结果/algorithm_memory.png")
        except ImportError:
            print("未安装matplotlib，跳过图表生成")
    
    return results

def verify_with_networkx(pr_values, data_path, teleport, epsilon, max_iter):
    """与NetworkX库的结果进行对比，验证算法准确性"""
    print("\n===== 与NetworkX结果对比验证 =====")
    
    # 动态导入NetworkX，减少常驻内存
    try:
        import networkx as nx
        
        # 加载图到NetworkX
        G = nx.DiGraph()
        edges = []
        
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    edges.append((from_node, to_node))
        
        G.add_edges_from(edges)
        
        # 使用NetworkX计算PageRank
        print("使用NetworkX计算PageRank进行结果验证...")
        nx_pr = nx.pagerank(
            G, 
            alpha=teleport, 
            tol=epsilon, 
            max_iter=max_iter,
            weight=None
        )
        
        # 比较前100个节点的准确性
        print("比较两种算法的结果...")
        top_pr = heapq.nlargest(100, pr_values.items(), key=lambda x: x[1])
        top_nx = heapq.nlargest(100, nx_pr.items(), key=lambda x: x[1])
        
        top_pr_nodes = {node for node, _ in top_pr}
        top_nx_nodes = {node for node, _ in top_nx}
        
        # 计算TOP100节点重叠率
        overlap = len(top_pr_nodes & top_nx_nodes)
        print(f"TOP100节点重叠率: {overlap}/100 ({overlap}%)")
        
        # 计算结果差异
        common_nodes = set(pr_values.keys()) & set(nx_pr.keys())
        if common_nodes:
            # 计算相对误差
            errors = []
            for node in common_nodes:
                pr_val = pr_values[node]
                nx_val = nx_pr[node]
                if nx_val > 0:  # 避免除以零
                    rel_error = abs(pr_val - nx_val) / nx_val
                    errors.append(rel_error)
            
            # 统计误差
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
            print(f"平均相对误差: {avg_error:.6f}")
            print(f"最大相对误差: {max_error:.6f}")
            
            # 检查是否符合要求
            if avg_error < 0.01 and max_error < 0.05:
                print("验证结果: 通过 ✓ (误差在可接受范围内)")
            else:
                print("验证结果: 误差较大，可能需要检查算法实现")
        
        # 清理内存
        G = None
        nx_pr = None
        gc.collect()
        
        return True
    
    except ImportError:
        print("无法导入NetworkX库，跳过结果验证")
        return False

def main():
    """主函数 - 内存优化版本"""
    # 设置参数
    data_path = "Data.txt"
    teleport = 0.85  # 传送参数
    epsilon = 1e-8   # 收敛阈值
    max_iter = 100   # 最大迭代次数
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 文件 {data_path} 不存在!")
        return 1
    
    # 命令行参数解析
    import sys
    import argparse
    
    # 显示帮助信息函数，节省内存
    def show_help():
        print("\nPageRank算法实现 - 内存优化版")
        print("\n用法: python pagerank.py [选项]")
        print("\n选项:")
        print("  --help, -h             显示此帮助信息")
        print("  --algorithm, -a ALGO   指定算法 (minimal, csr, basic, sparse, block, networkx)")
        print("  --teleport, -t VAL     设置teleport参数值 (默认: 0.85)")
        print("  --compare, -c          比较所有算法")
        print("  --teleport-compare, -tc 比较不同teleport值的影响")
        print("  --file, -f FILE        指定输入文件 (默认: Data.txt)")
        print("  --output, -o FILE      指定输出文件 (默认: 实验结果/Res.txt)")
        print("  --no-verify, -nv       不与NetworkX结果进行对比验证")
        print("\n默认使用最小内存算法 (MinimalMemoryPageRank)")
    
    # 简单解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            show_help()
            return 0
        
        # 解析参数
        algo_name = "minimal"  # 默认使用最小内存算法
        output_file = "实验结果/Res.txt"
        compare_mode = False
        teleport_compare = False
        verify_mode = False  # 默认不进行NetworkX验证
        
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            
            if arg in ["--algorithm", "-a"] and i+1 < len(sys.argv):
                algo_name = sys.argv[i+1]
                verify_mode
                i += 2
            elif arg in ["--teleport", "-t"] and i+1 < len(sys.argv):
                try:
                    teleport = float(sys.argv[i+1])
                    i += 2
                except ValueError:
                    print(f"错误: teleport值必须是浮点数: {sys.argv[i+1]}")
                    return 1
            elif arg in ["--compare", "-c"]:
                compare_mode = True
                i += 1
            elif arg in ["--teleport-compare", "-tc"]:
                teleport_compare = True
                i += 1
            elif arg in ["--file", "-f"] and i+1 < len(sys.argv):
                data_path = sys.argv[i+1]
                i += 2
            elif arg in ["--output", "-o"] and i+1 < len(sys.argv):
                output_file = sys.argv[i+1]
                i += 2
            elif arg in ["--no-verify", "-nv"]:
                verify_mode = False  # 用户可以选择关闭验证
                i += 1
            else:
                i += 1
        
        # 进行teleport参数比较
        if teleport_compare:
            # 强制垃圾回收
            gc.collect()
            compare_teleport_values(MinimalMemoryPageRank, data_path, 
                                   teleport_values=[0.7, 0.85, 0.95],
                                   epsilon=epsilon, max_iter=max_iter)
            return 0
        
        # 算法比较模式
        if compare_mode:
            # 强制垃圾回收
            gc.collect()
            # 比较所有算法，而不仅是minimal和csr
            benchmark_algorithms(data_path, teleport, epsilon, max_iter, 
                               selected_algorithms=['basic', 'minimal', 'sparse', 'block', 'csr', 'networkx'])
            return 0
        
        # 运行单个算法
        algorithms = {
            "basic": BasicPageRank,
            "minimal": MinimalMemoryPageRank,
            "sparse": SparseMatrixPageRank,
            "block": BlockMatrixPageRank,
            "csr": CSRPageRank,
            "networkx": NetworkXPageRank
        }
        
        if algo_name in algorithms:
            # 强制垃圾回收
            gc.collect()
            
            algorithm_class = algorithms[algo_name]
            
            print(f"使用 {algo_name} 算法计算PageRank")
            
            # 创建算法实例
            if algorithm_class == BlockMatrixPageRank:
                algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                         epsilon=epsilon, max_iter=max_iter, 
                                         block_size=1000)
            elif algorithm_class == CSRPageRank:
                algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                         epsilon=epsilon, max_iter=max_iter, 
                                         block_size=3200, num_threads=None)
            else:
                algorithm = algorithm_class(data_path=data_path, teleport=teleport, 
                                         epsilon=epsilon, max_iter=max_iter)
            
            # 计算PageRank
            pr_values = algorithm.compute_pagerank()
            
            # 保存结果
            algorithm.save_results(pr_values, output_file)
            
            # 与NetworkX进行结果对比验证
            if verify_mode and algo_name != 'networkx':
                # 如果算法不是NetworkX本身且未关闭验证，则进行验证
                verify_with_networkx(pr_values, data_path, teleport, epsilon, max_iter)
        else:
            print(f"未知的算法: {algo_name}")
            print("可用算法: basic, minimal, sparse, block, csr, networkx")
            
    else:
        # 默认运行：使用最小内存实现
        print("运行默认PageRank算法 (MinimalMemoryPageRank实现)")
        algorithm = MinimalMemoryPageRank(
            data_path=data_path, 
            teleport=teleport,
            epsilon=epsilon,
            max_iter=max_iter
        )
        
        #如果要使用教师提供的内存计算工具，请修改上面的算法类

        # 计算PageRank
        pr_values = algorithm.compute_pagerank()
        
        # 保存结果
        algorithm.save_results(pr_values, "实验结果/Res.txt")
        
        # 默认不与NetworkX对比验证
        # verify_with_networkx(pr_values, data_path, teleport, epsilon, max_iter)
    
    return 0


if __name__ == "__main__":
    main()