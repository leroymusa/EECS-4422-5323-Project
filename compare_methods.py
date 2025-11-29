"""
Comparison Script: K-Means vs GMM vs GrabCut
=============================================
Generates side-by-side comparison of all segmentation methods.

This script:
1. Runs Amely's k-means (imports from unsupervised.py)
2. Runs Leroy's GMM (imports from leroy_gmm.py)
3. Runs GrabCut (GMM + Graph Cuts - spatially regularized)
4. Compares all methods on the same images
5. Generates a summary table for the report
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from other modules
from leroy_gmm import (
    load_image, load_ground_truth, preprocess_image,
    fit_gmm_best_of_n, identify_foreground_component,
    gmm_to_probability_map, probability_to_mask, cleanup_mask,
    evaluate_performance
)

# Import Amely's k-means if available
try:
    from amely_unsupervised_work.unsupervised import segment_kmeans_rgb
    AMELY_AVAILABLE = True
except ImportError:
    AMELY_AVAILABLE = False
    print("Warning: Amely's unsupervised.py not found, k-means comparison disabled")


# =============================================================================
# GRABCUT IMPLEMENTATION
# =============================================================================

def segment_grabcut(image: np.ndarray,
                    initial_mask: Optional[np.ndarray] = None,
                    n_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    GrabCut segmentation (GMM + Graph Cut internally).
    
    GrabCut uses:
    - Two GMMs (foreground and background)
    - Graph cut optimization for spatial smoothness
    
    Args:
        image: BGR image
        initial_mask: Optional initialization mask (from k-means or GMM)
        n_iterations: Number of GrabCut iterations
    
    Returns:
        Final mask and probability-like confidence map
    """
    h, w = image.shape[:2]
    
    if initial_mask is None:
        # Use Otsu to get initial seed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert to GrabCut mask format
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:] = cv2.GC_PR_BGD  # Probable background
        mask[otsu_mask == 255] = cv2.GC_PR_FGD  # Probable foreground
    else:
        # Use provided mask as initialization
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:] = cv2.GC_PR_BGD
        mask[initial_mask == 255] = cv2.GC_PR_FGD
    
    # Allocate GMM arrays (required by OpenCV)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    # Run GrabCut with mask initialization
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 
                    n_iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        print(f"GrabCut error: {e}")
        return initial_mask if initial_mask is not None else np.zeros((h, w), dtype=np.uint8), np.zeros((h, w))
    
    # Convert GrabCut mask to binary
    # GC_FGD (1) or GC_PR_FGD (3) -> foreground
    final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Create a pseudo-confidence map based on mask values
    confidence = np.zeros((h, w), dtype=np.float32)
    confidence[mask == cv2.GC_FGD] = 1.0       # Definite foreground
    confidence[mask == cv2.GC_PR_FGD] = 0.75   # Probable foreground
    confidence[mask == cv2.GC_PR_BGD] = 0.25   # Probable background
    confidence[mask == cv2.GC_BGD] = 0.0       # Definite background
    
    return final_mask, confidence


# =============================================================================
# COMPARISON PIPELINE
# =============================================================================

def compare_methods_single_image(filepath: str,
                                 k_kmeans: int = 3,
                                 n_gmm_components: int = 2,
                                 n_gmm_runs: int = 10,
                                 save_plot: bool = True,
                                 verbose: bool = True) -> Dict:
    """
    Run all methods on a single image and compare results.
    """
    results = {
        'filename': os.path.basename(filepath),
        'kmeans': None,
        'gmm': None,
        'grabcut': None,
        'grabcut_gmm_init': None  # GrabCut initialized with GMM
    }
    
    # Load image
    original = load_image(filepath)
    if original is None:
        return results
    
    # Load ground truth
    base_name = os.path.splitext(filepath)[0]
    base_name_gt = base_name.replace("Raw_images", "Binary_masks")
    mask_path = base_name_gt + ".tif"
    if not os.path.exists(mask_path):
        mask_path = base_name_gt + ".png"
    gt_mask = load_ground_truth(mask_path, original.shape)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing methods on: {os.path.basename(filepath)}")
        print(f"{'='*70}")
    
    # Preprocess
    processed = preprocess_image(original, use_bilateral=True)
    
    # --- METHOD 1: K-MEANS ---
    if AMELY_AVAILABLE:
        if verbose:
            print("\n[1/4] Running K-Means...")
        try:
            kmeans_mask_raw, centers, labels = segment_kmeans_rgb(processed, k=k_kmeans)
            kmeans_mask = cleanup_mask(kmeans_mask_raw)
            
            if gt_mask is not None:
                acc, prec, rec, dice = evaluate_performance(kmeans_mask, gt_mask)
                results['kmeans'] = {
                    'accuracy': acc, 'precision': prec, 'recall': rec, 'dice': dice,
                    'mask': kmeans_mask, 'centers': centers
                }
                if verbose:
                    print(f"  K-Means (k={k_kmeans}): Dice={dice:.3f}")
        except Exception as e:
            print(f"  K-Means failed: {e}")
    
    # --- METHOD 2: GMM ---
    if verbose:
        print("\n[2/4] Running GMM...")
    pixels = processed.reshape(-1, 3).astype(np.float64)
    
    # Use k-means centers as initialization if available
    kmeans_init = None
    if results['kmeans'] is not None and results['kmeans']['centers'] is not None:
        centers = results['kmeans']['centers']
        if len(centers) == n_gmm_components:
            kmeans_init = centers
    
    best_gmm, best_ll, all_lls = fit_gmm_best_of_n(
        pixels, n_components=n_gmm_components, n_runs=n_gmm_runs,
        kmeans_centers=kmeans_init, verbose=False
    )
    
    if best_gmm is not None:
        fg_comp = identify_foreground_component(best_gmm, method='brightest')
        prob_map = gmm_to_probability_map(best_gmm, processed, fg_comp)
        gmm_mask_raw = probability_to_mask(prob_map, threshold=0.5)
        gmm_mask = cleanup_mask(gmm_mask_raw)
        
        if gt_mask is not None:
            acc, prec, rec, dice = evaluate_performance(gmm_mask, gt_mask)
            results['gmm'] = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'dice': dice,
                'mask': gmm_mask, 'prob_map': prob_map, 'log_likelihood': best_ll,
                'gmm_model': best_gmm, 'fg_component': fg_comp
            }
            if verbose:
                print(f"  GMM (n={n_gmm_components}): Dice={dice:.3f}, LL={best_ll:.0f}")
    
    # --- METHOD 3: GRABCUT (Otsu init) ---
    if verbose:
        print("\n[3/4] Running GrabCut (Otsu init)...")
    grabcut_mask, grabcut_conf = segment_grabcut(processed, initial_mask=None)
    grabcut_mask = cleanup_mask(grabcut_mask)
    
    if gt_mask is not None:
        acc, prec, rec, dice = evaluate_performance(grabcut_mask, gt_mask)
        results['grabcut'] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'dice': dice,
            'mask': grabcut_mask, 'confidence': grabcut_conf
        }
        if verbose:
            print(f"  GrabCut (Otsu): Dice={dice:.3f}")
    
    # --- METHOD 4: GRABCUT (GMM init) ---
    if results['gmm'] is not None:
        if verbose:
            print("\n[4/4] Running GrabCut (GMM init)...")
        gmm_init_mask = results['gmm']['mask']
        grabcut_gmm_mask, grabcut_gmm_conf = segment_grabcut(processed, initial_mask=gmm_init_mask)
        grabcut_gmm_mask = cleanup_mask(grabcut_gmm_mask)
        
        if gt_mask is not None:
            acc, prec, rec, dice = evaluate_performance(grabcut_gmm_mask, gt_mask)
            results['grabcut_gmm_init'] = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'dice': dice,
                'mask': grabcut_gmm_mask, 'confidence': grabcut_gmm_conf
            }
            if verbose:
                print(f"  GrabCut (GMM): Dice={dice:.3f}")
    
    # --- VISUALIZATION ---
    if save_plot:
        save_dir = os.path.join(os.path.dirname(filepath), '..', 'comparison_results')
        os.makedirs(save_dir, exist_ok=True)
        
        filename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        save_path = os.path.join(save_dir, f"{filename_no_ext}_comparison.png")
        
        visualize_comparison(
            original, gt_mask, results,
            os.path.basename(filepath), save_path
        )
    
    return results


def visualize_comparison(original: np.ndarray,
                        gt_mask: Optional[np.ndarray],
                        results: Dict,
                        title: str,
                        save_path: str):
    """Create a side-by-side comparison visualization."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original, GT, K-Means, GMM
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original: {title}")
    axes[0, 0].axis('off')
    
    if gt_mask is not None:
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis('off')
    
    if results['kmeans'] is not None:
        axes[0, 2].imshow(results['kmeans']['mask'], cmap='gray')
        axes[0, 2].set_title(f"K-Means\nDice={results['kmeans']['dice']:.3f}")
    else:
        axes[0, 2].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20)
    axes[0, 2].axis('off')
    
    if results['gmm'] is not None:
        axes[0, 3].imshow(results['gmm']['mask'], cmap='gray')
        axes[0, 3].set_title(f"GMM\nDice={results['gmm']['dice']:.3f}")
    else:
        axes[0, 3].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20)
    axes[0, 3].axis('off')
    
    # Row 2: GMM Probability, GrabCut (Otsu), GrabCut (GMM), Winner overlay
    if results['gmm'] is not None and 'prob_map' in results['gmm']:
        im = axes[1, 0].imshow(results['gmm']['prob_map'], cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title("GMM P(foreground)")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].axis('off')
    
    if results['grabcut'] is not None:
        axes[1, 1].imshow(results['grabcut']['mask'], cmap='gray')
        axes[1, 1].set_title(f"GrabCut (Otsu)\nDice={results['grabcut']['dice']:.3f}")
    axes[1, 1].axis('off')
    
    if results['grabcut_gmm_init'] is not None:
        axes[1, 2].imshow(results['grabcut_gmm_init']['mask'], cmap='gray')
        axes[1, 2].set_title(f"GrabCut (GMM init)\nDice={results['grabcut_gmm_init']['dice']:.3f}")
    axes[1, 2].axis('off')
    
    # Find best method
    methods = ['kmeans', 'gmm', 'grabcut', 'grabcut_gmm_init']
    method_names = ['K-Means', 'GMM', 'GrabCut', 'GrabCut+GMM']
    best_dice = 0
    best_method = None
    best_mask = None
    
    for method, name in zip(methods, method_names):
        if results[method] is not None:
            if results[method]['dice'] > best_dice:
                best_dice = results[method]['dice']
                best_method = name
                best_mask = results[method]['mask']
    
    if best_mask is not None:
        # Create overlay
        overlay = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
        mask_3ch = cv2.merge([best_mask, np.zeros_like(best_mask), np.zeros_like(best_mask)])
        overlay = cv2.addWeighted(overlay, 0.7, mask_3ch.astype(np.uint8), 0.3, 0)
        
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title(f"Best: {best_method}\nDice={best_dice:.3f}")
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison to: {save_path}")


def print_comparison_table(all_results: List[Dict]):
    """Print a summary table comparing all methods."""
    print("\n" + "=" * 100)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 100)
    
    header = f"{'IMAGE':<12} | {'K-MEANS':<12} | {'GMM':<12} | {'GRABCUT':<12} | {'GRABCUT+GMM':<12} | {'BEST':<15}"
    print(header)
    print("-" * 100)
    
    method_sums = {'kmeans': 0, 'gmm': 0, 'grabcut': 0, 'grabcut_gmm_init': 0}
    method_counts = {'kmeans': 0, 'gmm': 0, 'grabcut': 0, 'grabcut_gmm_init': 0}
    wins = {'kmeans': 0, 'gmm': 0, 'grabcut': 0, 'grabcut_gmm_init': 0}
    
    for res in all_results:
        row_values = {}
        best_dice = 0
        best_method = None
        
        for method in ['kmeans', 'gmm', 'grabcut', 'grabcut_gmm_init']:
            if res[method] is not None:
                dice = res[method]['dice']
                row_values[method] = f"{dice:.3f}"
                method_sums[method] += dice
                method_counts[method] += 1
                
                if dice > best_dice:
                    best_dice = dice
                    best_method = method
            else:
                row_values[method] = "N/A"
        
        if best_method:
            wins[best_method] += 1
        
        method_labels = {'kmeans': 'K-Means', 'gmm': 'GMM', 'grabcut': 'GrabCut', 'grabcut_gmm_init': 'GC+GMM'}
        best_label = method_labels.get(best_method, 'N/A')
        
        print(f"{res['filename']:<12} | {row_values['kmeans']:<12} | {row_values['gmm']:<12} | "
              f"{row_values['grabcut']:<12} | {row_values['grabcut_gmm_init']:<12} | {best_label:<15}")
    
    # Averages
    print("-" * 100)
    avg_row = "AVERAGE      |"
    for method in ['kmeans', 'gmm', 'grabcut', 'grabcut_gmm_init']:
        if method_counts[method] > 0:
            avg = method_sums[method] / method_counts[method]
            avg_row += f" {avg:<12.3f}|"
        else:
            avg_row += f" {'N/A':<12}|"
    print(avg_row)
    
    # Win counts
    print("-" * 100)
    wins_row = "WINS         |"
    for method in ['kmeans', 'gmm', 'grabcut', 'grabcut_gmm_init']:
        wins_row += f" {wins[method]:<12}|"
    print(wins_row)
    print("=" * 100)


def main():
    """Run comparison on all images."""
    files = [f'Saliva_Segmentation_dataset/Raw_images/{i}.tif' for i in range(1, 11)]
    
    print("\n" + "=" * 70)
    print("SEGMENTATION METHOD COMPARISON")
    print("K-Means vs GMM vs GrabCut")
    print("=" * 70)
    
    all_results = []
    
    for filepath in files:
        result = compare_methods_single_image(
            filepath,
            k_kmeans=3,
            n_gmm_components=2,
            n_gmm_runs=10,
            save_plot=True,
            verbose=True
        )
        all_results.append(result)
    
    # Print final comparison
    print_comparison_table(all_results)
    
    print("\n\nResults saved to: Saliva_Segmentation_dataset/comparison_results/")


if __name__ == "__main__":
    main()

