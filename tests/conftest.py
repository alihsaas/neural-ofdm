import importlib.util

TORCH_TESTS = ["test_neural_models.py", "test_attention_models.py"]

collect_ignore = [] if importlib.util.find_spec("torch") else TORCH_TESTS
