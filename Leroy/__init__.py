"""
Leroy's GMM Segmentation Module

This module provides Gaussian Mixture Model based segmentation for saliva detection.
"""

from .leroy_gmm import (
    segment_image_gmm,
    fit_gmm_best_of_n,
    fit_gmm_single,
    gmm_to_probability_map,
    identify_foreground_component,
    load_image,
    load_ground_truth,
    preprocess_image,
    evaluate_performance,
    probability_to_mask,
    cleanup_mask
)

__all__ = [
    'segment_image_gmm',
    'fit_gmm_best_of_n',
    'fit_gmm_single',
    'gmm_to_probability_map',
    'identify_foreground_component',
    'load_image',
    'load_ground_truth',
    'preprocess_image',
    'evaluate_performance',
    'probability_to_mask',
    'cleanup_mask'
]

