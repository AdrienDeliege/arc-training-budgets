# https://github.com/ekinakyurek/marc
import copy
from typing import List, Optional, Tuple
import numpy as np
from numpy.random import RandomState
from .arc import Example, Grid, Task

class Augmenter:
    share_rng: bool = False

    def __repr__(self):
        return str(self)

    def __call__(self, grid: Grid, rng: RandomState=None) -> Grid:
        return self.apply_to_grid(grid, rng=rng)

    def apply_to_grid(self, grid: Grid, rng: RandomState=None) -> Grid:
        raise NotImplementedError()

    def apply_to_example(self, example: Example, rng: RandomState=None, to_input: bool=True, to_output: bool=True) -> Example:
        input = self.apply_to_grid(example.input, rng=rng) if to_input else example.input
        output = self.apply_to_grid(example.output, rng=rng) if to_output else example.output
        if example.cot is not None:
            cot = copy.deepcopy(example.cot)
            if not np.array_equal(cot[-1], output):
                cot.append(output)
        else:
            cot = None
        return Example(input, output, cot)

    def apply_to_task(self, task: Task, rng: RandomState=None, share_rng=False, **kwargs) -> Task:
        if self.share_rng:
            train_rngs = [copy.deepcopy(rng) for i in range(len(task.train_examples))]
            test_rng = copy.deepcopy(rng)
        else:
            train_rngs = [rng for _ in range(len(task.train_examples))]
            test_rng = rng
        return Task(train_examples=[self.apply_to_example(example, rng=rng_i, **kwargs) for (example, rng_i) in zip(task.train_examples, train_rngs)], test_example=self.apply_to_example(task.test_example, rng=test_rng, **kwargs), name=task.name)

class Rotate(Augmenter):

    def __init__(self, angle: int):
        assert angle in [0, 90, 180, 270]
        self.angle = angle

    def __str__(self):
        return f'Rotate({self.angle})'

    def apply_to_grid(self, grid: Grid, rng: RandomState=None) -> Grid:
        del rng
        if self.angle == 90:
            return np.rot90(grid, k=1)
        elif self.angle == 180:
            return np.rot90(grid, k=2)
        elif self.angle == 270:
            return np.rot90(grid, k=3)
        else:
            raise ValueError('Invalid angle')

class PermuteColors(Augmenter):
    share_rng = True
    color_mapper = None

    def __str__(self):
        return 'PermuteColors()'

    def apply_to_task(self, task: Task, use_test_output: bool=True, rng: RandomState=None, share_rng=False, **kwargs) -> Task:
        colors = []
        for example in task.train_examples:
            colors += example.input.flatten().tolist()
            colors += example.output.flatten().tolist()
        colors += task.test_example.input.flatten().tolist()
        if use_test_output:
            colors += task.test_example.output.flatten().tolist()
        colors = set(colors)
        colors = colors - {0}
        remaining_colors = list(set(list(range(1, 10))) - colors)
        colors = list(colors)
        rng.shuffle(remaining_colors)
        permuted_colors = rng.permutation(colors).tolist()
        color_map = {0: 0}
        for color in colors:
            if color in color_map:
                continue
            if len(remaining_colors) > 0:
                new_color = remaining_colors.pop()
            else:
                new_color = permuted_colors.pop()
                color_map[new_color] = color
                if color in permuted_colors:
                    permuted_colors.remove(color)
            color_map[color] = new_color
        self._color_map = color_map

        def color_mapper(color: int) -> int:
            return color_map.get(color, color)
        self.color_mapper = np.vectorize(color_mapper)
        return Task(train_examples=[self.apply_to_example(example, rng=rng, **kwargs) for example in task.train_examples], test_example=self.apply_to_example(task.test_example, rng=rng, **kwargs), name=task.name)

    def apply_to_grid(self, grid: Grid, rng: RandomState=None) -> Grid:
        return self.color_mapper(grid)

class PermuteColorswithMap(Augmenter):

    def __init__(self, color_map):
        self.color_map = color_map

        def color_mapper(color: int) -> int:
            return color_map.get(color, color)
        self.color_mapper = np.vectorize(color_mapper)

    def apply_to_grid(self, grid: Grid, rng: RandomState=None) -> Grid:
        return self.color_mapper(grid)

    def __repr__(self):
        return f'PermuteColorswithMap({self.color_map})'

    def __str__(self):
        return self.__repr__()

class Flip(Augmenter):

    def __init__(self, axis: int):
        assert axis in [0, 1]
        self.axis = axis

    def __str__(self):
        return f'Flip({self.axis})'

    def apply_to_grid(self, grid: Grid, rng: RandomState=None) -> Grid:
        del rng
        if self.axis == 0:
            return np.flipud(grid)
        elif self.axis == 1:
            return np.fliplr(grid)
        else:
            raise ValueError('Invalid axis')
