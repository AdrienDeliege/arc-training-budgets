import torch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import json
import random
import argparse
import itertools
import numpy as np
from itertools import combinations
IGNORE_INDEX = 10
PAD_INDEX = 11
MAX_SIZE = 30

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    For parallel training
    """
    inputs = torch.stack([item['inputs'] for item in batch], dim=0)
    attention = torch.stack([item['attention_mask'] for item in batch], dim=0)
    targets = torch.stack([item['targets'] for item in batch], dim=0)
    task_ids = torch.stack([item['task_id'] for item in batch], dim=0)
    target_shapes = torch.stack([item['target_shape'] for item in batch], dim=0)
    example_indices = torch.stack([item['example_index'] for item in batch], dim=0)
    offset = torch.stack([torch.tensor(item['offset']) for item in batch], dim=0)
    scale_factors = torch.stack([torch.tensor(item['scale_factor']) for item in batch], dim=0)
    task_names = [item['task_name'] for item in batch]
    raw_inputs = [item['raw_input'] for item in batch]
    raw_outputs = [item['raw_output'] for item in batch]
    collated = {'inputs': inputs, 'attention_mask': attention, 'targets': targets, 'task_ids': task_ids, 'target_shapes': target_shapes, 'example_indices': example_indices, 'task_names': task_names, 'raw_inputs': raw_inputs, 'raw_outputs': raw_outputs, 'offset': offset, 'scale_factors': scale_factors}
    if 'transform_id' in batch[0]:
        collated['transform_ids'] = torch.stack([item['transform_id'] for item in batch], dim=0)
    if 'color_permutation' in batch[0]:
        collated['color_permutations'] = torch.stack([item['color_permutation'] for item in batch], dim=0)
    if 'augmentation_index' in batch[0]:
        collated['augmentation_indices'] = torch.stack([item['augmentation_index'] for item in batch], dim=0)
    if 'base_task_id' in batch[0]:
        collated['base_task_ids'] = torch.stack([item['base_task_id'] for item in batch], dim=0)
    return collated

def pad_grid_with_translation(grid: List[List[int]], max_size: int, x_offset: int, y_offset: int, output_shape=True) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Do random translation and padding
    Returns:
        padded_tensor: (max_size, max_size) tensor with padding
        mask: (max_size, max_size) tensor, 1 for valid positions, 0 for padding
        height: height of original grid
        width: width of original grid
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if height > max_size - 2 or width > max_size - 2:
        raise ValueError(f'Grid size ({height}, {width}) exceeds configured max_size={max_size - 2}.')
    tensor = torch.full((max_size, max_size), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((max_size, max_size), dtype=torch.long)
    values = torch.tensor(grid, dtype=torch.long)
    tensor[y_offset:y_offset + height, x_offset:x_offset + width] = values
    mask[y_offset:y_offset + height, x_offset:x_offset + width] = 1
    if output_shape:
        tensor[y_offset:y_offset + height, x_offset + width] = PAD_INDEX
        tensor[y_offset + height, x_offset:x_offset + width + 1] = PAD_INDEX
        mask[y_offset:y_offset + height + 1, x_offset:x_offset + width + 1] = 1
    return (tensor, mask, height, width)

def resolution_augmentation(example, max_cur_x, max_cur_y, rng, img_size=60, max_scale_factor_cap: Optional[int]=None):
    """
    Do resolution augmentation by random scaling
    1. Randomly choose a scale factor
    2. Scale up the input and output grids
    3. Return the new example and scale factor
    """
    max_len = max(max_cur_x, max_cur_y)
    max_scale_factor = img_size // max_len
    if max_scale_factor_cap is not None and max_scale_factor_cap > 0:
        max_scale_factor = min(max_scale_factor, max_scale_factor_cap)
    max_scale_factor = max(1, max_scale_factor)
    scale_factor = rng.randint(1, max_scale_factor)
    new_example = {}
    new_example['input'] = np.repeat(np.repeat(example['input'], scale_factor, axis=0), scale_factor, axis=1).tolist()
    new_example['output'] = np.repeat(np.repeat(example['output'], scale_factor, axis=0), scale_factor, axis=1).tolist()
    return (new_example, scale_factor)

def get_num_on_the_fly_transforms(geometry_set: str) -> int:
    if geometry_set == 'dihedral':
        return 8
    if geometry_set == 'varc_basic':
        return 6
    raise ValueError(f'Unsupported geometry_set: {geometry_set}')

def apply_on_the_fly_geometry(grid: List[List[int]], transform_id: int) -> List[List[int]]:
    array = np.asarray(grid, dtype=np.int64)
    if transform_id == 0:
        transformed = array
    elif transform_id == 1:
        transformed = np.rot90(array, k=1)
    elif transform_id == 2:
        transformed = np.rot90(array, k=2)
    elif transform_id == 3:
        transformed = np.rot90(array, k=3)
    elif transform_id == 4:
        transformed = np.flipud(array)
    elif transform_id == 5:
        transformed = np.fliplr(array)
    elif transform_id == 6:
        transformed = array.T
    elif transform_id == 7:
        transformed = np.flipud(np.fliplr(array.T))
    else:
        raise ValueError(f'Unsupported transform_id: {transform_id}')
    return transformed.tolist()

def undo_on_the_fly_geometry(grid: List[List[int]], transform_id: int) -> List[List[int]]:
    array = np.asarray(grid, dtype=np.int64)
    if array.ndim < 2 or 0 in array.shape[:2]:
        return []
    if transform_id == 0:
        transformed = array
    elif transform_id == 1:
        transformed = np.rot90(array, k=3)
    elif transform_id == 2:
        transformed = np.rot90(array, k=2)
    elif transform_id == 3:
        transformed = np.rot90(array, k=1)
    elif transform_id == 4:
        transformed = np.flipud(array)
    elif transform_id == 5:
        transformed = np.fliplr(array)
    elif transform_id == 6:
        transformed = array.T
    elif transform_id == 7:
        transformed = np.flipud(np.fliplr(array.T))
    else:
        raise ValueError(f'Unsupported transform_id: {transform_id}')
    return transformed.tolist()

def sample_on_the_fly_color_permutation(rng: random.Random) -> List[int]:
    real_colors = list(range(1, 10))
    rng.shuffle(real_colors)
    return [0] + real_colors

def identity_on_the_fly_color_permutation() -> List[int]:
    return list(range(10))

def deterministic_on_the_fly_color_permutation(seed: int) -> List[int]:
    rng = random.Random(int(seed))
    return sample_on_the_fly_color_permutation(rng)

def apply_on_the_fly_color_permutation(grid: List[List[int]], color_permutation: List[int]) -> List[List[int]]:
    array = np.asarray(grid, dtype=np.int64)
    mapped = array.copy()
    for (source_color, target_color) in enumerate(color_permutation):
        mapped[array == source_color] = target_color
    return mapped.tolist()

def invert_on_the_fly_color_permutation(color_permutation: List[int]) -> Dict[int, int]:
    return {int(target): int(source) for (source, target) in enumerate(color_permutation)}

def apply_on_the_fly_augmentation(example: Dict[str, Any], transform_id: int, color_permutation: List[int]) -> Dict[str, Any]:
    augmented: Dict[str, Any] = {}
    augmented['input'] = apply_on_the_fly_color_permutation(apply_on_the_fly_geometry(example['input'], transform_id), color_permutation)
    if 'output' in example:
        augmented['output'] = apply_on_the_fly_color_permutation(apply_on_the_fly_geometry(example['output'], transform_id), color_permutation)
    return augmented

class ARCDataset(Dataset):

    def __init__(self, root: Path, split: str, subset: str='train', max_size: int=32, task_lookup: Optional[Dict[str, int]]=None, max_scale_factor: Optional[int]=None) -> None:
        if subset not in {'train', 'test'}:
            raise ValueError("subset must be 'train' or 'test'.")
        self.rng = random.Random(42)
        self.root = Path(root)
        self.max_size = max_size
        self.subset = subset
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.task_lookup: Dict[str, int] = dict(task_lookup) if task_lookup is not None else {}
        split_dir = self.root / 'data' / split
        split_file = split_dir if split_dir.is_file() else split_dir.with_suffix('.json')
        if split_file.is_file():
            files = [split_file]
            split_dir_for_message = split_file.parent
        else:
            files = [path for path in sorted(split_dir.rglob('*.json')) if not path.name.startswith('._')]
            split_dir_for_message = split_dir
        examples_key = 'train' if subset == 'train' else 'test'
        self.translation_enabled = True
        self.fix_scale_factor = 2
        self.resolution_enabled = True
        self.max_scale_factor = max_scale_factor
        print('Start loading data...')
        for file_path in files:
            task_name = file_path.stem
            if task_lookup is None:
                if task_name in self.task_lookup:
                    task_index = self.task_lookup[task_name]
                else:
                    task_index = len(self.task_lookup)
                    self.task_lookup[task_name] = task_index
            else:
                if task_name not in self.task_lookup:
                    continue
                task_index = self.task_lookup[task_name]
            with file_path.open('r') as fh:
                task_data = json.load(fh)
            examples = task_data.get(examples_key, [])
            for (example_index, example) in enumerate(examples):
                max_cur_y = len(example['input'])
                max_cur_x = len(example['input'][0])
                if 'output' in example:
                    max_cur_y = max(max_cur_y, len(example['output']))
                    max_cur_x = max(max_cur_x, len(example['output'][0]))
                if max_cur_y > MAX_SIZE or max_cur_x > MAX_SIZE:
                    continue
                self.samples.append({'example': example, 'task_index': task_index, 'task_name': task_name, 'example_index': example_index})
        if not self.samples:
            raise RuntimeError(f"No samples found for split '{split}' subset '{subset}' under {split_dir_for_message}.")
        print('Finish loading!!')
        self.num_tasks = len(self.task_lookup)
        self.num_base_tasks = self.num_tasks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cur_batch = self.samples[idx]
        processed_random_translation_batch = self.process_per_example(example=cur_batch['example'], task_index=cur_batch['task_index'], task_name=cur_batch['task_name'], example_index=cur_batch['example_index'], rng=self.rng, if_translation=self.translation_enabled)
        return processed_random_translation_batch

    def enable_translation(self):
        self.translation_enabled = True

    def disable_resolution_augmentation(self, fix_scale_factor=2):
        self.fix_scale_factor = fix_scale_factor
        self.resolution_enabled = False

    def enable_resolution_augmentation(self):
        self.resolution_enabled = True

    def disable_translation(self):
        self.translation_enabled = False

    def process_per_example(self, example, task_index, task_name, example_index, rng, if_translation=True):
        """
        Conduct random translation and resolution augmentation for each example
        1. Randomly scale up the input and output grids
        2. Randomly translate the grids within the max_size
        3. Pad the grids to max_size
        4. Return the processed tensors and other info
        Returns:
            A dictionary containing:
                inputs: (max_size, max_size) tensor of input grid with padding
                attention_mask: (max_size, max_size) tensor, 1 for valid positions, 0 for padding
                targets: (max_size, max_size) tensor of output grid with padding
                task_id: tensor of task index
                task_name: task name string
                example_index: tensor of example index
                target_shape: tensor of (height, width) of original output grid
                raw_input: original input grid
                raw_output: original output grid
                offset: (x_offset, y_offset) used for translation
                scale_factor: scale factor used for resolution augmentation
        """
        max_cur_y = len(example['input'])
        max_cur_x = len(example['input'][0])
        if 'output' in example:
            max_cur_y = max(max_cur_y, len(example['output']))
            max_cur_x = max(max_cur_x, len(example['output'][0]))
        max_img_size = self.max_size - 2
        max_size = self.max_size
        if self.resolution_enabled:
            (example, scale_factor) = resolution_augmentation(example, max_cur_x, max_cur_y, rng, img_size=max_img_size, max_scale_factor_cap=self.max_scale_factor)
        else:
            scale_factor = self.fix_scale_factor
            new_example = {}
            new_example['input'] = np.repeat(np.repeat(example['input'], scale_factor, axis=0), scale_factor, axis=1).tolist()
            new_example['output'] = np.repeat(np.repeat(example['output'], scale_factor, axis=0), scale_factor, axis=1).tolist()
            example = new_example
        max_cur_x = max_cur_x * scale_factor
        max_cur_y = max_cur_y * scale_factor
        if if_translation:
            x_offset = rng.randint(1, max_img_size - max_cur_x) if max_img_size > max_cur_x else 1
            y_offset = rng.randint(1, max_img_size - max_cur_y) if max_img_size > max_cur_y else 1
        else:
            x_offset = 1
            y_offset = 1
        (input_grid, input_mask, _, _) = pad_grid_with_translation(example['input'], max_size, x_offset, y_offset, output_shape=False)
        if 'output' in example:
            (target_grid, target_mask, target_h, target_w) = pad_grid_with_translation(example['output'], max_size, x_offset, y_offset, output_shape=True)
        else:
            target_grid = torch.full((max_size, max_size), IGNORE_INDEX, dtype=torch.long)
            target_mask = torch.zeros((max_size, max_size), dtype=torch.long)
            target_h = 0
            target_w = 0
        target_grid = target_grid.clone()
        target_grid[target_mask == 0] = IGNORE_INDEX
        raw_input = example.get('input', [])
        raw_output = example.get('output') if 'output' in example else None
        return {'inputs': input_grid, 'attention_mask': input_mask, 'targets': target_grid, 'task_id': torch.tensor(task_index, dtype=torch.long), 'task_name': task_name, 'example_index': torch.tensor(example_index, dtype=torch.long), 'target_shape': torch.tensor([target_h, target_w], dtype=torch.long), 'raw_input': raw_input, 'raw_output': raw_output, 'offset': (x_offset, y_offset), 'scale_factor': scale_factor}

class OnTheFlyAugmentedARCDataset(ARCDataset):

    def __init__(self, root: Path, split: str, subset: str='train', max_size: int=32, task_lookup: Optional[Dict[str, int]]=None, virtual_variants_per_epoch: int=1, eval_augmentations: int=1, geometry_set: str='varc_basic', augmentation_schedule: str='random', augmentation_seed: int=12345, task_id_mode: str='base', max_scale_factor: Optional[int]=None) -> None:
        if task_id_mode not in {'base', 'per_variant'}:
            raise ValueError(f'Unsupported task_id_mode: {task_id_mode}')
        loader_task_lookup = None if task_id_mode == 'per_variant' else task_lookup
        super().__init__(root, split, subset=subset, max_size=max_size, task_lookup=loader_task_lookup, max_scale_factor=max_scale_factor)
        self.base_num_samples = len(self.samples)
        self.virtual_variants_per_epoch = max(1, int(virtual_variants_per_epoch))
        self.eval_augmentations = max(1, int(eval_augmentations))
        self.geometry_set = geometry_set
        self.num_transforms = get_num_on_the_fly_transforms(geometry_set)
        if augmentation_schedule not in {'random', 'fixed'}:
            raise ValueError(f'Unsupported augmentation_schedule: {augmentation_schedule}')
        self.augmentation_schedule = augmentation_schedule
        self.augmentation_seed = int(augmentation_seed)
        self.num_aug_task_tokens = max(self.virtual_variants_per_epoch, self.eval_augmentations)
        self.task_id_mode = task_id_mode
        self.base_task_lookup = dict(self.task_lookup)
        self.num_base_tasks = len(self.base_task_lookup)
        if self.task_id_mode == 'per_variant':
            self.task_lookup = dict(task_lookup) if task_lookup is not None else self._build_per_variant_task_lookup()
            self.num_tasks = len(self.task_lookup)

    @staticmethod
    def augmentation_task_name(task_name: str, augmentation_index: int) -> str:
        return f'{task_name}__aug{augmentation_index:04d}'

    def _build_per_variant_task_lookup(self) -> Dict[str, int]:
        base_task_names = sorted(self.base_task_lookup, key=lambda task_name: self.base_task_lookup[task_name])
        task_lookup: Dict[str, int] = {}
        for base_task_name in base_task_names:
            for augmentation_index in range(self.num_aug_task_tokens):
                task_lookup[self.augmentation_task_name(base_task_name, augmentation_index)] = len(task_lookup)
        return task_lookup

    def __len__(self) -> int:
        multiplier = self.virtual_variants_per_epoch if self.subset == 'train' else self.eval_augmentations
        return self.base_num_samples * multiplier

    def _fixed_augmentation(self, augmentation_index: int) -> Tuple[int, List[int], int]:
        if augmentation_index == 0:
            return (0, identity_on_the_fly_color_permutation(), 0)
        non_identity_transforms = max(1, self.num_transforms - 1)
        transform_group = (augmentation_index - 1) // 10
        transform_id = 1 + transform_group % non_identity_transforms
        color_variant = (augmentation_index - 1) % 10
        if color_variant == 0:
            color_permutation = identity_on_the_fly_color_permutation()
        else:
            color_permutation = deterministic_on_the_fly_color_permutation(self.augmentation_seed + transform_group * 1000003 + color_variant * 1048583)
        return (transform_id, color_permutation, augmentation_index)

    def _sample_train_augmentation(self, augmentation_index: int) -> Tuple[int, List[int], int]:
        if self.augmentation_schedule == 'fixed':
            return self._fixed_augmentation(augmentation_index)
        transform_id = self.rng.randrange(self.num_transforms)
        color_permutation = sample_on_the_fly_color_permutation(self.rng)
        return (transform_id, color_permutation, augmentation_index)

    def _sample_eval_augmentation(self, augmentation_index: int) -> Tuple[int, List[int], int]:
        if self.augmentation_schedule == 'fixed':
            return self._fixed_augmentation(augmentation_index)
        if augmentation_index == 0:
            return (0, identity_on_the_fly_color_permutation(), 0)
        rng = random.Random(self.augmentation_seed + augmentation_index * 1000003 + self.num_transforms * 997)
        transform_id = rng.randrange(self.num_transforms)
        color_permutation = sample_on_the_fly_color_permutation(rng)
        return (transform_id, color_permutation, augmentation_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = idx % self.base_num_samples
        augmentation_index = idx // self.base_num_samples
        cur_batch = self.samples[base_idx]
        if self.subset == 'train':
            (transform_id, color_permutation, stored_aug_index) = self._sample_train_augmentation(augmentation_index)
        else:
            (transform_id, color_permutation, stored_aug_index) = self._sample_eval_augmentation(augmentation_index)
        augmented_example = apply_on_the_fly_augmentation(cur_batch['example'], transform_id, color_permutation)
        processed = self.process_per_example(example=augmented_example, task_index=cur_batch['task_index'], task_name=cur_batch['task_name'], example_index=cur_batch['example_index'], rng=self.rng, if_translation=self.translation_enabled)
        processed['base_task_id'] = torch.tensor(cur_batch['task_index'], dtype=torch.long)
        if self.task_id_mode == 'per_variant':
            augmentation_task_name = self.augmentation_task_name(cur_batch['task_name'], stored_aug_index)
            if augmentation_task_name not in self.task_lookup:
                raise KeyError(f'Missing task id for fixed augmentation slot {augmentation_task_name!r}.')
            processed['task_id'] = torch.tensor(self.task_lookup[augmentation_task_name], dtype=torch.long)
        processed['transform_id'] = torch.tensor(transform_id, dtype=torch.long)
        processed['color_permutation'] = torch.tensor(color_permutation, dtype=torch.long)
        processed['augmentation_index'] = torch.tensor(stored_aug_index, dtype=torch.long)
        return processed

def _use_on_the_fly_augmentations(args: argparse.Namespace) -> bool:
    return bool(getattr(args, 'on_the_fly_augmentations', False) or getattr(args, 'ttt_on_the_fly_augmentations', False))

def _augmentation_arg(args: argparse.Namespace, generic_name: str, ttt_name: str, default: Any) -> Any:
    generic_value = getattr(args, generic_name, None)
    if generic_value is not None:
        return generic_value
    return getattr(args, ttt_name, default)

def build_dataloaders(args: argparse.Namespace, *, distributed: bool=False, rank: int=0, world_size: int=1):
    root = Path(args.data_root)
    extra_roots: Optional[List[Path]] = None
    extra_limit: Optional[int] = None
    train_split = getattr(args, 'train_split', 'training')
    use_on_the_fly_augmentations = _use_on_the_fly_augmentations(args)
    virtual_variants_per_epoch = _augmentation_arg(args, 'virtual_variants_per_epoch', 'ttt_virtual_variants_per_epoch', 1)
    eval_augmentations = _augmentation_arg(args, 'eval_augmentations', 'ttt_eval_augmentations', 1)
    geometry_set = _augmentation_arg(args, 'geometry_set', 'ttt_geometry_set', 'dihedral')
    augmentation_schedule = _augmentation_arg(args, 'augmentation_schedule', 'ttt_augmentation_schedule', 'random')
    augmentation_seed = _augmentation_arg(args, 'augmentation_seed', 'ttt_augmentation_seed', 12345)
    augmentation_task_id_mode = _augmentation_arg(args, 'augmentation_task_id_mode', 'ttt_augmentation_task_id_mode', 'base')
    if use_on_the_fly_augmentations:
        train_dataset = OnTheFlyAugmentedARCDataset(root, train_split, subset='train', max_size=args.image_size, max_scale_factor=getattr(args, 'max_scale_factor', None), virtual_variants_per_epoch=virtual_variants_per_epoch, eval_augmentations=eval_augmentations, geometry_set=geometry_set, augmentation_schedule=augmentation_schedule, augmentation_seed=augmentation_seed, task_id_mode=augmentation_task_id_mode)
    else:
        train_dataset = ARCDataset(root, train_split, subset='train', max_size=args.image_size, max_scale_factor=getattr(args, 'max_scale_factor', None))
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn)
    eval_loader = None
    eval_sampler = None
    if args.eval_split:
        eval_subset = getattr(args, 'eval_subset', 'test')
        if use_on_the_fly_augmentations:
            eval_dataset = OnTheFlyAugmentedARCDataset(root, args.eval_split, subset=eval_subset, max_size=args.image_size, max_scale_factor=getattr(args, 'max_scale_factor', None), task_lookup=train_dataset.task_lookup, virtual_variants_per_epoch=virtual_variants_per_epoch, eval_augmentations=eval_augmentations, geometry_set=geometry_set, augmentation_schedule=augmentation_schedule, augmentation_seed=augmentation_seed, task_id_mode=augmentation_task_id_mode)
        else:
            eval_dataset = ARCDataset(root, args.eval_split, subset=eval_subset, max_size=args.image_size, max_scale_factor=getattr(args, 'max_scale_factor', None), task_lookup=train_dataset.task_lookup)
        if distributed:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, sampler=eval_sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        eval_dataset = None
    return (train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler)
