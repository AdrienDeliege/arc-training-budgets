"""Checkpoint portability checks; run in the TRM environment with PyTorch 2.7+."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None and hasattr(torch.nn, 'Buffer'), 'Requires PyTorch 2.7+ environment')
class CheckpointTests(unittest.TestCase):
    def test_compiled_and_eager_checkpoint_keys_and_embedding_resize(self):
        # Exercise the actual loader without importing the CUDA training runtime.
        source = ast.parse((ROOT / 'vendor/trm/pretrain.py').read_text())
        loader = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'load_checkpoint')
        namespace = {'torch': torch, 'nn': nn, 'PretrainConfig': object}
        exec(compile(ast.Module(body=[loader], type_ignores=[]), 'TRM checkpoint loader', 'exec'), namespace)

        class Embedding(nn.Module):
            def __init__(self, count):
                super().__init__()
                self.register_buffer('weights', torch.zeros(count, 4))

        class Model(nn.Module):
            def __init__(self, count):
                super().__init__()
                self.inner = nn.Module()
                self.inner.puzzle_emb = Embedding(count)
                self.projection = nn.Linear(4, 4)

        class LossHead(nn.Module):
            def __init__(self, count):
                super().__init__()
                self.model = Model(count)

        original = LossHead(7)
        original.model.inner.puzzle_emb.weights.copy_(torch.arange(28).reshape(7, 4))
        original_state = original.state_dict()
        for source_compiled in [False, True]:
            for target_compiled in [False, True]:
                for count in [3, 7]:
                    with self.subTest(source_compiled=source_compiled, target_compiled=target_compiled, count=count):
                        state = {('_orig_mod.' if source_compiled else '') + k: v.clone() for k, v in original_state.items()}
                        destination = LossHead(count)
                        if target_compiled:
                            destination = torch.compile(destination)
                        with patch.object(torch, 'load', return_value=state):
                            namespace['load_checkpoint'](destination, SimpleNamespace(load_checkpoint='test-checkpoint.pt'))
                        loaded = {k.removeprefix('_orig_mod.'): v for k, v in destination.state_dict().items()}
                        for key, value in original_state.items():
                            expected = value.mean(0, keepdim=True).expand(count, -1) if key.endswith('puzzle_emb.weights') and count != 7 else value
                            self.assertTrue(torch.equal(loaded[key], expected), key)


if __name__ == '__main__':
    unittest.main()
