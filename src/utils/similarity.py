"""
Unified similarity utilities for text and embedding comparisons.

This module provides standardized similarity calculations used across
the research agent for deduplication, clustering, and comparison tasks.
"""

import numpy as np
from typing import List, Union


def cosine_similarity(
    vec1: Union[np.ndarray, List[float]],
    vec2: Union[np.ndarray, List[List[float]], np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate cosine similarity between vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    resulting in a value between -1 (opposite) and 1 (identical).
    We clip to [0, 1] range for use as similarity score.

    Args:
        vec1: Single embedding vector (1D array or list)
        vec2: Single vector (1D) or array of vectors (2D)

    Returns:
        Similarity score(s) in range [0.0, 1.0]
        - Single float if vec2 is 1D
        - Array of floats if vec2 is 2D

    Examples:
        >>> vec_a = [1, 2, 3]
        >>> vec_b = [1, 2, 3]
        >>> cosine_similarity(vec_a, vec_b)
        1.0

        >>> vec_c = [[1, 2, 3], [4, 5, 6]]
        >>> cosine_similarity(vec_a, vec_c)
        array([1.0, 0.974...])
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Normalize vec1
    vec1_norm = vec1 / np.linalg.norm(vec1)

    # Handle single vector vs array of vectors
    if vec2.ndim == 1:
        # Single vector comparison
        vec2_norm = vec2 / np.linalg.norm(vec2)
        similarity = np.dot(vec1_norm, vec2_norm)
    else:
        # Array of vectors (2D matrix)
        vec2_norms = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        similarity = np.dot(vec2_norms, vec1_norm)

    # Clamp to [0, 1] range (negative similarities become 0)
    return np.clip(similarity, 0.0, 1.0)
