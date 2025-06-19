import time
from collections import defaultdict

import torch
import torch_scatter
from einops import rearrange

class TimeRecorder:
    _instance = None

    def __init__(self):
        self.items = {}
        self.accumulations = defaultdict(list)
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"
        self.enabled = False

    def __new__(cls):
        # singleton
        if cls._instance is None:
            cls._instance = super(TimeRecorder, cls).__new__(cls)
        return cls._instance

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.items[name] = time.time()

    def end(self, name: str, accumulate: bool = False) -> float:
        if not self.enabled or name not in self.items:
            return
        torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        if accumulate:
            self.accumulations[name].append(delta)
        t = delta * self.time_scale
        print(f"{name}: {t:.2f}{self.time_unit}")

    def get_accumulation(self, name: str, average: bool = False) -> float:
        if not self.enabled or name not in self.accumulations:
            return
        acc = self.accumulations.pop(name)
        total = sum(acc)
        if average:
            t = total / len(acc) * self.time_scale
        else:
            t = total * self.time_scale
        print(f"{name} for {len(acc)} times: {t:.2f}{self.time_unit}")

tr = TimeRecorder()


def collate_fn_from_batch(batch):
    collated_batch = {}
    batch_size = batch['coords'].shape[0]
    collated_batch['batch_size'] = batch_size

    for key in batch.keys():
        if key == 'coords':
            # coords = [batch[key][i] for i in range(batch_size)]
            coords = []
            for i in range(batch_size):
                raw_coords_normalized = batch[key][i] - batch[key][i].min(0, keepdim=True).values
                coords.append(raw_coords_normalized)
            batch_coords = [
                torch.cat([torch.full((coord.shape[0], 1), i, dtype=coord.dtype, device=coord.device), coord], dim=1)
                for i, coord in enumerate(coords)]
            collated_batch[key] = torch.cat(batch_coords, dim=0)
        else:
            collated_batch[key] = torch.cat([batch[key][i] for i in range(batch_size)], dim=0)

    return collated_batch

def ravel_hash_torch(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, x.shape

    x = x - torch.min(x, dim=0)[0]
    x = x.to(torch.int64)
    xmax = torch.max(x, dim=0)[0].to(torch.int64) + 1

    h = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]

    return h

def voxelize_without_feature_pool(coords, voxel_size):
    # No Batch Dim !!! The batch dim is added in the collate_fn
    # Coords: [N, 1+3]
    coords = coords.detach()
    new_float_coord = torch.cat(
        [coords[:, 0].view(-1, 1), (coords[:, 1:]) / voxel_size], 1
    )
    # new_float_coord: [N, 1+3]
    new_int_coord = torch.floor(new_float_coord).int()

    # faster implementation
    tr.start('ravel_hash_torch')
    hash_idx = ravel_hash_torch(new_int_coord)
    tr.end('ravel_hash_torch')

    _, idx_query = torch.unique(hash_idx, sorted=False, return_inverse=True, )
    voxel_coords = torch_scatter.scatter_mean(new_int_coord, idx_query.long(), dim=0).detach()

    return voxel_coords, idx_query

def voxelize_without_feature_pool_with_point_new_coord(coords, voxel_size):
    # No Batch Dim !!! The batch dim is added in the collate_fn
    # Coords: [N, 1+3]
    coords = coords.detach()
    new_float_coord = torch.cat(
        [coords[:, 0].view(-1, 1), (coords[:, 1:]) / voxel_size], 1
    )
    # new_float_coord: [N, 1+3]
    new_int_coord = torch.floor(new_float_coord).int()

    # faster implementation
    tr.start('ravel_hash_torch')
    hash_idx = ravel_hash_torch(new_int_coord)
    tr.end('ravel_hash_torch')

    _, idx_query = torch.unique(hash_idx, sorted=False, return_inverse=True, )
    voxel_coords = torch_scatter.scatter_mean(new_int_coord, idx_query.long(), dim=0).detach()

    return voxel_coords, idx_query, new_float_coord

def voxelize_with_feature_pool(coords, feature, voxel_size):
    # No Batch Dim !!! The batch dim is added in the collate_fn
    # Coords: [N, 1+3]
    coords = coords.detach()
    new_float_coord = torch.cat(
        [coords[:, 0].view(-1, 1), (coords[:, 1:]) / voxel_size], 1
    )
    # new_float_coord: [N, 1+3]
    new_int_coord = torch.floor(new_float_coord).int()

    # faster implementation
    tr.start('ravel_hash_torch')
    hash_idx = ravel_hash_torch(new_int_coord)
    tr.end('ravel_hash_torch')

    # tr.start('unique_with_indices_torch')
    # # This process is very slow, although it's already faster than the previous implementation (i.e., in torchsparse)
    # # A possible speed-up is to do cache for idx and inverse_idx if the voxel level has been computed before
    # _, indices, idx_query = unique_with_indices_torch(hash_idx)
    # # assert torch.all(new_int_coord[indices][idx_query] == new_int_coord)
    # # Note that feature[indices][idx_query] != feature, because feature is still continuous
    # voxel_coords = new_int_coord[indices]
    # tr.end('unique_with_indices_torch')

    _, idx_query = torch.unique(hash_idx, sorted=False, return_inverse=True, )
    voxel_coords = torch_scatter.scatter_mean(new_int_coord, idx_query.long(), dim=0).detach()

    tr.start('torch_scatter.scatter_mean')
    voxel_feature_pool = torch_scatter.scatter_mean(feature, idx_query.long(), dim=0)
    tr.end('torch_scatter.scatter_mean')

    return voxel_coords, voxel_feature_pool, idx_query

def devoxelize_with_feature_nearest(feature, idx_query):
    return feature[idx_query]

def unique_with_indices_torch(x):
    assert x.dim() == 1

    x_sorted, sorted_indices = torch.sort(x)

    inverse_sorted_indices = torch.empty_like(sorted_indices)
    inverse_sorted_indices[sorted_indices] = torch.arange(sorted_indices.size(0), device=x.device)

    # st = time.time()
    unique_mask = torch.cat((torch.tensor([True], device=x.device), x_sorted[1:] != x_sorted[:-1]))
    # print('unique_mask:', time.time() - st)

    x_unique = x_sorted[unique_mask]
    inverse_mask = torch.searchsorted(x_unique, x_sorted)

    indices = sorted_indices[unique_mask]
    inverse_indices = inverse_mask[inverse_sorted_indices]

    return x_unique, indices, inverse_indices

