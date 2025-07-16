import torch
import numpy as np

class CustomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_views_per_polyp, drop_last):
        """
        sublist_indices: 每个view所属的polyp的索引
        batch_size: 每个批次的大小
        """
        self.sublist_indices = dataset.train_instance
        self.batch_size = batch_size
        self.num_polyp_per_batch = batch_size // num_views_per_polyp
        self.drop_last = drop_last
        # 获取所有polyp的索引
        self.unique_class_indices = np.unique(self.sublist_indices)

        assert len(self.sublist_indices) % num_views_per_polyp == 0, "每个polyp应该包含4个view"
        
        # 根据polyp索引将数据分组
        self.class_groups = {}
        for i, class_idx in enumerate(self.unique_class_indices):
            self.class_groups[class_idx] = np.where(self.sublist_indices == class_idx)[0]

    def __iter__(self):
        
        # 随机打乱大类的顺序
        random_class_indices = np.random.permutation(self.unique_class_indices)
        batch_indices = []  # 用于收集每个批次的索引
        total_indices = []
        # 遍历所有大类
        for class_idx in random_class_indices:
            # 将每个大类的所有元素加入当前批次
            batch_indices.extend(self.class_groups[class_idx])
            
            # 每满一个 batch_size 就生成一个批次
            if len(batch_indices) >= self.batch_size:
                # yield batch_indices
                total_indices.append(batch_indices)
                batch_indices = []  # 剩余的数据继续处理

        # 如果还有剩余数据（最后一个批次小于batch_size），则依然要返回
        if len(batch_indices) > 0 and not self.drop_last:
            # yield batch_indices
            total_indices.append(batch_indices)
        return iter(total_indices)
        
    def __len__(self):
        if self.drop_last:
            return len(self.sublist_indices) // self.batch_size
        else:
            return (len(self.sublist_indices) + self.batch_size - 1) // self.batch_size