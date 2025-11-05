# tests/test_search.py
import numpy as np
from src.search import cosine_similarity, _to_numpy

# test_cosine_similarity_identical function tests the cosine similarity of two identical vectors.
# We create two identical vectors and test the cosine similarity.
# We assert the cosine similarity is 1.0.
def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

# test_cosine_similarity_orthogonal function tests the cosine similarity of two orthogonal vectors.
# We create two orthogonal vectors and test the cosine similarity.
# We assert the cosine similarity is 0.0.
def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-6

# test_to_numpy_valid_and_invalid function tests the _to_numpy function with valid and invalid vectors.
# We create a valid vector and test the _to_numpy function.
# We assert the shape of the vector is 3.
# We assert the _to_numpy function returns None for an empty list.
# We assert the _to_numpy function returns None for a None value.
def test_to_numpy_valid_and_invalid():
    v = [1,2,3]
    arr = _to_numpy(v)
    assert arr.shape[0] == 3
    assert _to_numpy([]) is None
    assert _to_numpy(None) is None
