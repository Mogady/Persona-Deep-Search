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


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Jaccard similarity is the size of the intersection divided by
    the size of the union of the sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Similarity score in range [0.0, 1.0]
        - 0.0 means no overlap
        - 1.0 means identical sets

    Examples:
        >>> set1 = {"apple", "banana", "orange"}
        >>> set2 = {"banana", "orange", "grape"}
        >>> jaccard_similarity(set1, set2)
        0.5  # 2 items in common out of 4 total unique items
    """
    if not set1 and not set2:
        return 1.0  # Both empty = identical

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def text_overlap_similarity(text1: str, text2: str, case_sensitive: bool = False) -> float:
    """
    Calculate word-level overlap similarity between two texts.

    Uses Jaccard similarity on word sets.

    Args:
        text1: First text string
        text2: Second text string
        case_sensitive: Whether to preserve case (default: False)

    Returns:
        Similarity score in range [0.0, 1.0]

    Examples:
        >>> text1 = "the quick brown fox"
        >>> text2 = "the lazy brown dog"
        >>> text_overlap_similarity(text1, text2)
        0.4  # 2 words in common (the, brown) out of 5 unique words
    """
    if not text1 or not text2:
        return 0.0

    # Convert to word sets
    if not case_sensitive:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
    else:
        words1 = set(text1.split())
        words2 = set(text2.split())

    return jaccard_similarity(words1, words2)
