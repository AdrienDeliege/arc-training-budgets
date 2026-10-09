# ARC task structures from https://github.com/ekinakyurek/marc
import dataclasses
import glob
import json
import os
from typing import List, Optional
import numpy as np
Grid = np.ndarray

def to_tuple(arr):
    return tuple((tuple([int(e) for e in row]) for row in arr))

def to_list(arr):
    return [[int(e) for e in row] for row in arr]

@dataclasses.dataclass
class Example:
    """
    class to represent an example
    """
    input: Grid
    output: Grid
    cot: Optional[List[Grid]] = None

    def input_size(self) -> int:
        """return the size of the input grid"""
        return self.input.size

    def output_size(self) -> int:
        """return the size of the output grid"""
        return self.output.size

    def size(self) -> int:
        """return the size of the example"""
        return max(self.input_size(), self.output_size())

    def __hash__(self) -> int:
        return hash((self.input.tobytes(), self.output.tobytes()))

    def __repr__(self) -> str:
        return f'Example(input={self.input}, output={self.output})'

    def serialize(self) -> dict:
        example = {'input': self.input.tolist(), 'output': self.output.tolist()}
        if self.cot:
            example['cot'] = [cot.tolist() for cot in self.cot]
        return example

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Example):
            return NotImplemented
        return np.array_equal(self.input, other.input) and np.array_equal(self.output, other.output)

    @classmethod
    def deserialize(cls, data: dict, test: bool=False) -> 'Example':
        input = np.array(data['input'])
        if test:
            output = input.copy()
        elif 'output' in data:
            output = np.array(data['output'])
        else:
            output = input.copy()
        cot = None
        if 'cot' in data:
            cot = [np.array(c) for c in data['cot']]
        return cls(input, output, cot)

@dataclasses.dataclass
class Task:
    """
    A class to represent a task
    """
    test_example: Example
    train_examples: List[Example] = dataclasses.field(default_factory=list)
    name: str = ''

    def size(self) -> int:
        """return the size of the task"""
        return max([example.size() for example in self.train_examples])

    def max_height(self) -> int:
        max_x = 0
        for example in self.train_examples:
            (x, _) = example.input.shape
            max_x = max(max_x, x)
            (x, _) = example.output.shape
            max_x = max(max_x, x)
        (x, _) = self.test_example.input.shape
        max_x = max(max_x, x)
        (x, _) = self.test_example.output.shape
        max_x = max(max_x, x)
        return max_x

    def max_width(self) -> int:
        max_y = 0
        for example in self.train_examples:
            (_, y) = example.input.shape
            max_y = max(max_y, y)
            (_, y) = example.output.shape
            max_y = max(max_y, y)
        (_, y) = self.test_example.input.shape
        max_y = max(max_y, y)
        (_, y) = self.test_example.output.shape
        max_y = max(max_y, y)
        return max_y

    def __repr__(self) -> str:
        return f'Task(train={self.train_examples}, test={self.test_example})'

    def serialize(self) -> dict:
        return {'train': [train.serialize() for train in self.train_examples], 'test': [self.test_example.serialize()], 'name': self.name}

    def __hash__(self) -> int:
        return hash((tuple((train for train in self.train_examples)), self.test_example))

    @classmethod
    def deserialize(cls, data: dict, test: bool=False) -> 'Task':
        assert len(data['test']) == 1, 'Only one test example is allowed'
        train = [Example.deserialize(train) for train in data['train']]
        test = Example.deserialize(data['test'][0], test=test)
        return cls(train_examples=train, test_example=test, name=data.get('name', ''))

    @classmethod
    def read_tasks_from_dict(cls, data: dict, test: bool=False) -> List['Task']:
        tasks = []
        for test_data in data['test']:
            task = cls.deserialize({'train': data['train'], 'test': [test_data], 'name': data.get('name', '')}, test=test)
            tasks.append(task)
        return tasks

    def entropy(self) -> float:
        """return the entropy of the outputs"""
        outputs = [example.output.flatten() for example in self.train_examples]
        outputs.append(self.test_example.output.flatten())
        vocabulary = np.unique(np.concatenate(outputs)).tolist()
        max_output_length = max([len(output) for output in outputs])
        probs = np.zeros((len(vocabulary), max_output_length))
        for (i, output) in enumerate(outputs):
            for (j, value) in enumerate(output):
                index_of_value = vocabulary.index(value)
                probs[index_of_value, j] += 1
        probs = probs / probs.sum(axis=0)
        entropy = -np.sum(probs * np.log(probs + 1e-09), axis=0)
        return np.mean(entropy)
