"""
Leroy's GMM-based Segmentation Pipeline (No OpenCV)
====================================================
Probabilistic segmentation using Gaussian Mixture Models.

Key features (from Prof's office hour notes):
- Fits GMM on RGB pixel features
- Runs multiple initializations to find best local optimum
- Uses pseudo-log-likelihood to select best model
- Outputs probability maps (not just hard labels)
- "Don't have to worry about foreground/background, because you get a 
   probability of foreground/background instead of a classification"

This plugs into Amely's k-means work by:
1. Accepting k-means centers as initialization seeds
2. Comparing GMM results with k-means results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from typing import Tuple, Optional, List, Dict

try:
    from amely_unsupervised_work.unsupervised import segment_kmeans_rgb
    HAS_AMELY_KMEANS = True
except ImportError:
    HAS_AMELY_KMEANS = False

# Image loading - try tifffile first, fallback to matplotlib
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# Output directory (root-level so results appear in the repo root as requested)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'leroy_gmm_results')

# =============================================================================
# IMAGE LOADING (No OpenCV needed)
# =============================================================================

def load_image(filepath: str, target_width: int = 800) -> Optional[np.ndarray]:
    """Load and resize image using matplotlib/tifffile."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    file_ext = os.path.splitext(filepath)[1].lower()
    img = None

    try:
        if file_ext in ['.tif', '.tiff']:
            if HAS_TIFFFILE:
                img = tifffile.imread(filepath)
            else:
                img = plt.imread(filepath)
        else:
            img = plt.imread(filepath)
        
        # Handle different image formats
        if img is None:
            return None
            
        # Convert to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        
        # Ensure 3 channels (RGB)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # RGBA -> RGB
            img = img[:, :, :3]
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    if img is None:
        return None

    # Resize if needed
    h, w = img.shape[:2]
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        # Simple resize using numpy
        img = resize_image(img, (new_h, target_width))
    
    return img


def resize_image(img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """Simple image resize using scipy zoom."""
    zoom_factors = (new_shape[0] / img.shape[0], 
                    new_shape[1] / img.shape[1], 
                    1)
    return ndimage.zoom(img, zoom_factors, order=1).astype(np.uint8)


def load_ground_truth(filepath: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load and resize ground truth mask."""
    if not os.path.exists(filepath):
        return None
    
    try:
        if HAS_TIFFFILE and filepath.endswith(('.tif', '.tiff')):
            gt = tifffile.imread(filepath)
        else:
            gt = plt.imread(filepath)
        
        # Convert to grayscale if needed
        if len(gt.shape) == 3:
            gt = np.mean(gt, axis=2)
        
        # Resize to match target
        if gt.shape[:2] != target_shape[:2]:
            zoom_factors = (target_shape[0] / gt.shape[0], 
                           target_shape[1] / gt.shape[1])
            gt = ndimage.zoom(gt, zoom_factors, order=0)
        
        # Binarize
        threshold = gt.max() / 2
        gt = (gt > threshold).astype(np.uint8) * 255
        
        return gt
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


# =============================================================================
# PREPROCESSING (Spatial Filtering - using scipy)
# =============================================================================

def preprocess_image(image: np.ndarray, 
                     use_gaussian: bool = True,
                     sigma: float = 1.0) -> np.ndarray:
    """
    Apply spatial filtering to reduce noise.
    Uses Gaussian blur (scipy) instead of bilateral filter.
    """
    if not use_gaussian:
        return image.copy()
    
    # Apply Gaussian filter to each channel
    processed = np.zeros_like(image)
    for i in range(3):
        processed[:, :, i] = ndimage.gaussian_filter(image[:, :, i], sigma=sigma)
    
    return processed


# =============================================================================
# GMM CORE FUNCTIONS (The heart of what Prof wants!)
# =============================================================================

def fit_gmm_single(pixels: np.ndarray, 
                   n_components: int = 2,
                   covariance_type: str = 'full',
                   random_state: int = 42,
                   means_init: Optional[np.ndarray] = None) -> Tuple[GaussianMixture, float]:
    """
    Fit a single GMM and return the model + log-likelihood.
    
    From Prof's notes:
    "Leroy: once data is fitted, evaluate the pseudo-log likelihood"
    
    Args:
        pixels: N x 3 array of RGB values
        n_components: Number of Gaussian components (2 = fg/bg, 3 = fg/bg/highlight)
        covariance_type: 'full', 'tied', 'diag', or 'spherical'
        random_state: For reproducibility
        means_init: Optional initial means (e.g., from k-means centers)
    
    Returns:
        Fitted GMM model and total pseudo-log-likelihood
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=1,  # Single init here; we handle multiple runs externally
        random_state=random_state,
        means_init=means_init,
        max_iter=200,
        tol=1e-4
    )
    
    gmm.fit(pixels)
    
    # PSEUDO-LOG-LIKELIHOOD: average log-likelihood * number of samples
    # Called "pseudo" because we're using the same data to fit and evaluate
    avg_log_likelihood = gmm.score(pixels)  # This is per-sample average
    total_log_likelihood = avg_log_likelihood * pixels.shape[0]
    
    return gmm, total_log_likelihood


def fit_gmm_best_of_n(pixels: np.ndarray,
                      n_components: int = 2,
                      n_runs: int = 10,
                      covariance_type: str = 'full',
                      kmeans_centers: Optional[np.ndarray] = None,
                      verbose: bool = True) -> Tuple[GaussianMixture, float, List[float]]:
    """
    Run GMM fitting multiple times and select the best by log-likelihood.
    
    From Prof's notes:
    "Only guaranteed to converge to local optimum"
    "How do you choose the local optimum? Compute the probability of each 
     solution, and choose the most probable solution."
    
    Args:
        pixels: N x 3 array of RGB values
        n_components: Number of components
        n_runs: Number of random initializations to try
        covariance_type: Covariance structure
        kmeans_centers: Optional k-means centers from Amely to use as one initialization
        verbose: Print progress
    
    Returns:
        Best GMM model, its log-likelihood, and list of all log-likelihoods
    """
    best_gmm = None
    best_ll = -np.inf
    all_lls = []
    
    for i in range(n_runs):
        # For the first run, use k-means centers if provided
        means_init = None
        if i == 0 and kmeans_centers is not None:
            if len(kmeans_centers) == n_components:
                means_init = kmeans_centers.astype(np.float64)
                if verbose:
                    print(f"  Run {i+1}/{n_runs}: Using k-means centers as initialization")
        
        try:
            gmm, ll = fit_gmm_single(
                pixels, 
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=i * 42,  # Different seed each run
                means_init=means_init
            )
            all_lls.append(ll)
            
            if ll > best_ll:
                best_ll = ll
                best_gmm = gmm
                if verbose:
                    print(f"  Run {i+1}/{n_runs}: LL = {ll:.2f} (NEW BEST)")
            elif verbose:
                print(f"  Run {i+1}/{n_runs}: LL = {ll:.2f}")
                
        except Exception as e:
            if verbose:
                print(f"  Run {i+1}/{n_runs}: Failed - {e}")
            all_lls.append(-np.inf)
    
    return best_gmm, best_ll, all_lls


def identify_foreground_component(gmm: GaussianMixture, 
                                  method: str = 'brightest') -> int:
    """
    Determine which GMM component corresponds to saliva (foreground).
    
    From Prof's notes:
    "Assume biggest distance is foreground" (or brightest)
    
    Methods:
        'brightest': Component with highest mean luminance (saliva glows)
        'distance': Component furthest from origin (most saturated/bright)
    """
    means = gmm.means_  # Shape: (n_components, 3) in RGB order
    
    if method == 'brightest':
        # Compute luminance: Y = 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * means[:, 0] + 0.587 * means[:, 1] + 0.114 * means[:, 2]
        return int(np.argmax(luminance))
    
    elif method == 'distance':
        # Distance from origin (black) in RGB space
        distances = np.linalg.norm(means, axis=1)
        return int(np.argmax(distances))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def gmm_to_probability_map(gmm: GaussianMixture, 
                           image: np.ndarray,
                           foreground_component: int) -> np.ndarray:
    """
    Convert GMM to a probability map P(foreground | pixel).
    
    From Prof's notes:
    "Probabilistic model now gives a lot per data point you try to fit"
    "Don't have to worry about foreground/background, because you get a 
     probability of foreground/background instead of a classification"
    
    Returns:
        Probability map with same height/width as image, values in [0, 1]
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float64)
    
    # Get posterior probabilities for all components
    probs = gmm.predict_proba(pixels)  # Shape: (N, n_components)
    
    # Extract foreground probability
    fg_prob = probs[:, foreground_component]
    
    return fg_prob.reshape(h, w)


def probability_to_mask(prob_map: np.ndarray, 
                        threshold: float = 0.5) -> np.ndarray:
    """
    Convert probability map to binary mask.
    
    Args:
        prob_map: 2D array of P(foreground)
        threshold: Probability threshold for foreground classification
    
    Returns:
        Binary mask (0 or 255)
    """
    mask = (prob_map > threshold).astype(np.uint8) * 255
    return mask


# =============================================================================
# POST-PROCESSING (Spatial Regularization - using scipy)
# =============================================================================

def cleanup_mask(mask: np.ndarray,
                 iterations: int = 1) -> np.ndarray:
    """Apply morphological operations to clean up the mask using scipy."""
    # Create a circular structuring element
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, 2)  # Make it larger
    
    # Opening (remove small bright spots)
    cleaned = ndimage.binary_opening(mask > 0, structure=struct, iterations=iterations)
    
    # Closing (fill small holes)
    cleaned = ndimage.binary_closing(cleaned, structure=struct, iterations=iterations)
    
    return (cleaned * 255).astype(np.uint8)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_performance(pred_mask: np.ndarray, 
                        true_mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute accuracy, precision, recall, and Dice score."""
    p = (pred_mask.flatten() > 127).astype(int)
    t = (true_mask.flatten() > 127).astype(int)
    
    TP = np.sum((p == 1) & (t == 1))
    TN = np.sum((p == 0) & (t == 0))
    FP = np.sum((p == 1) & (t == 0))
    FN = np.sum((p == 0) & (t == 1))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    
    return accuracy, precision, recall, dice


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_gmm_results(original: np.ndarray,
                          prob_map: np.ndarray,
                          final_mask: np.ndarray,
                          gt_mask: Optional[np.ndarray],
                          gmm: GaussianMixture,
                          fg_component: int,
                          log_likelihood: float,
                          title: str,
                          save_path: Optional[str] = None):
    """
    Comprehensive visualization of GMM segmentation results.
    
    Shows:
    1. Original image
    2. Marginal projections with GMM means
    3. 3D RGB scatter with GMM centers
    4. Probability heatmap
    5. Final binary mask
    6. Ground truth (if available)
    """
    fig = plt.figure(figsize=(18, 12))
    
    means = gmm.means_
    n_components = len(means)
    
    # 1. Original Image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original)
    ax1.set_title(f"Input: {title}")
    ax1.axis('off')
    
    # 2. Marginal Projections (Histograms with GMM means)
    ax2 = fig.add_subplot(2, 3, 2)
    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist, bins = np.histogram(original[:, :, i].flatten(), bins=256, range=(0, 256))
        ax2.plot(bins[:-1], hist, color=color, alpha=0.6, label=name)
        
        # Mark GMM means
        for cid, mean in enumerate(means):
            style = '-' if cid == fg_component else '--'
            alpha = 0.8 if cid == fg_component else 0.4
            ax2.axvline(x=mean[i], color=color, linestyle=style, alpha=alpha, linewidth=2)
    
    ax2.set_title(f"Marginal Projections\nSolid=Foreground, Dashed=Background")
    ax2.set_xlim([0, 256])
    ax2.legend()
    
    # 3. 3D Scatter Plot with GMM Centers
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Downsample pixels
    flat_img = original.reshape(-1, 3)
    sample_size = 3000
    if len(flat_img) > sample_size:
        indices = np.random.choice(len(flat_img), sample_size, replace=False)
    else:
        indices = np.arange(len(flat_img))
    
    sample_pixels = flat_img[indices]
    sample_colors = sample_pixels / 255.0
    
    ax3.scatter(sample_pixels[:, 0], sample_pixels[:, 1], sample_pixels[:, 2],
                c=sample_colors, marker='.', s=1, alpha=0.3)
    
    # Plot GMM centers
    for i, mean in enumerate(means):
        center_rgb = mean / 255.0
        label = "Spit (FG)" if i == fg_component else f"BG-{i}"
        marker = 'X' if i == fg_component else 'o'
        size = 300 if i == fg_component else 150
        ax3.scatter(mean[0], mean[1], mean[2],
                    c=[center_rgb], marker=marker, s=size, 
                    edgecolor='black', linewidth=2, label=label)
    
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_zlabel('Blue')
    ax3.set_title(f"GMM Components (n={n_components})\nLL={log_likelihood:.0f}")
    ax3.legend()
    
    # 4. Probability Heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax4.set_title("P(Foreground | pixel)")
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. Final Mask
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(final_mask, cmap='gray')
    ax5.set_title("GMM Segmentation Mask")
    ax5.axis('off')
    
    # 6. Ground Truth
    if gt_mask is not None:
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(gt_mask, cmap='gray')
        ax6.set_title("Ground Truth")
        ax6.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def segment_image_gmm(filepath: str,
                      n_components: int = 2,
                      n_runs: int = 10,
                      prob_threshold: float = 0.5,
                      use_gaussian: bool = True,
                      kmeans_centers: Optional[np.ndarray] = None,
                      use_kmeans_seed: bool = True,
                      save_plot: bool = True,
                      verbose: bool = True) -> Optional[Dict]:
    """
    Full GMM segmentation pipeline for a single image.
    
    This implements Prof's advice:
    1. Fit GMM on pixel colors
    2. Run multiple initializations (n_runs)
    3. Use pseudo-log-likelihood to select best model
    4. Get P(foreground | pixel) probabilities
    5. Choose most probable solution
    
    Args:
        filepath: Path to input image
        n_components: Number of GMM components (2 or 3 recommended)
        n_runs: Number of GMM initializations to try
        prob_threshold: Threshold for P(foreground) -> binary mask
        use_gaussian: Apply Gaussian filter preprocessing
        kmeans_centers: Optional centers from Amely's k-means (for initialization)
        save_plot: Save visualization to file
        verbose: Print progress
    
    Returns:
        Dictionary with metrics, mask, probability map, and GMM model
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(filepath)}")
        print(f"{'='*60}")
    
    # 1. Load image
    original_img = load_image(filepath)
    if original_img is None:
        return None
    
    # 2. Load ground truth
    base_name = os.path.splitext(filepath)[0]
    base_name_gt = base_name.replace("Raw_images", "Binary_masks")
    mask_path = base_name_gt + ".tif"
    if not os.path.exists(mask_path):
        mask_path = base_name_gt + ".png"
    gt_mask = load_ground_truth(mask_path, original_img.shape)
    
    # 3. Preprocess (spatial filtering)
    if verbose:
        print(f"Preprocessing: gaussian={use_gaussian}")
    processed = preprocess_image(original_img, use_gaussian=use_gaussian)
    
    # 4. Prepare pixel features
    pixels = processed.reshape(-1, 3).astype(np.float64)
    
    # 5. Fit GMM with multiple runs (Prof's key advice!)
    if verbose:
        print(f"Fitting GMM (n_components={n_components}, n_runs={n_runs})...")
    
    init_centers = kmeans_centers
    if init_centers is None and use_kmeans_seed and HAS_AMELY_KMEANS:
        try:
            _, centers, _ = segment_kmeans_rgb(processed, k=n_components)
            init_centers = centers.astype(np.float64)
            if verbose:
                print("  Using Amely's k-means centers to seed GMM.")
        except Exception as e:
            if verbose:
                print(f"  K-means seed unavailable: {e}")

    best_gmm, best_ll, all_lls = fit_gmm_best_of_n(
        pixels,
        n_components=n_components,
        n_runs=n_runs,
        kmeans_centers=init_centers,
        verbose=verbose
    )
    
    if best_gmm is None:
        print("ERROR: All GMM fits failed!")
        return None
    
    if verbose:
        print(f"\nBest Log-Likelihood: {best_ll:.2f}")
        print(f"LL Range: [{min(all_lls):.2f}, {max(all_lls):.2f}]")
        print(f"LL Std Dev: {np.std(all_lls):.2f}")
    
    # 6. Identify foreground component
    fg_component = identify_foreground_component(best_gmm, method='brightest')
    if verbose:
        print(f"Foreground component: {fg_component} (brightest)")
        print(f"GMM Means (RGB):")
        for i, mean in enumerate(best_gmm.means_):
            marker = " <-- FOREGROUND" if i == fg_component else ""
            print(f"  Component {i}: [{mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}]{marker}")
    
    # 7. Generate probability map (Prof's key point: probabilities not just labels!)
    prob_map = gmm_to_probability_map(best_gmm, processed, fg_component)
    
    # 8. Convert to binary mask
    raw_mask = probability_to_mask(prob_map, threshold=prob_threshold)
    
    # 9. Apply morphological cleanup
    final_mask = cleanup_mask(raw_mask)
    
    # 10. Evaluate
    metrics = None
    if gt_mask is not None:
        acc, prec, rec, dice = evaluate_performance(final_mask, gt_mask)
        metrics = {
            'filename': os.path.basename(filepath),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'dice': dice,
            'log_likelihood': best_ll
        }
        if verbose:
            print(f"\nMetrics:")
            print(f"  Accuracy:  {acc:.3f}")
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall:    {rec:.3f}")
            print(f"  Dice:      {dice:.3f}")
    else:
        metrics = {
            'filename': os.path.basename(filepath),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'dice': 0.0,
            'log_likelihood': best_ll
        }
    
    # 11. Visualize
    if save_plot:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        
        filename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        save_path = os.path.join(OUTPUT_ROOT, f"{filename_no_ext}_gmm_n{n_components}.png")
        
        visualize_gmm_results(
            original_img, prob_map, final_mask, gt_mask,
            best_gmm, fg_component, best_ll,
            os.path.basename(filepath), save_path
        )
    
    return {
        'metrics': metrics,
        'mask': final_mask,
        'prob_map': prob_map,
        'gmm': best_gmm,
        'fg_component': fg_component,
        'log_likelihood': best_ll,
        'all_lls': all_lls
    }


def print_results_table(all_results: List[Dict], n_components: int):
    """Print a formatted results table."""
    print("\n" + "=" * 85)
    print(f"GMM RESULTS (n_components={n_components})")
    print("-" * 85)
    print(f"{'IMAGE':<15} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'DICE':<10} | {'LOG-LIK':<12}")
    print("-" * 85)
    
    acc_sum, prec_sum, rec_sum, dice_sum, ll_sum = 0, 0, 0, 0, 0
    count = 0
    
    for result in all_results:
        m = result['metrics']
        print(f"{m['filename']:<15} | {m['accuracy']:<10.3f} | {m['precision']:<10.3f} | "
              f"{m['recall']:<10.3f} | {m['dice']:<10.3f} | {m['log_likelihood']:<12.0f}")
        
        if m['accuracy'] > 0:
            acc_sum += m['accuracy']
            prec_sum += m['precision']
            rec_sum += m['recall']
            dice_sum += m['dice']
            ll_sum += m['log_likelihood']
            count += 1
    
    print("-" * 85)
    if count > 0:
        print(f"{'AVERAGE':<15} | {acc_sum/count:<10.3f} | {prec_sum/count:<10.3f} | "
              f"{rec_sum/count:<10.3f} | {dice_sum/count:<10.3f} | {ll_sum/count:<12.0f}")
    print("=" * 85)


def main():
    """
    Main experiment: Run GMM segmentation on all images with different configs.
    
    This implements the full pipeline from Prof's office hours:
    - Fit GMM on RGB pixels
    - Run multiple initializations
    - Select best by pseudo-log-likelihood
    - Report metrics per condition
    """
    # --- CONFIG ---
    files_to_process = [f'Saliva_Segmentation_dataset/Raw_images/{i}.tif' for i in range(1, 11)]
    component_experiments = [3, 4]  # Try 3 and 4 components
    n_runs = 10  # Number of GMM initializations per image
    
    SAVE_PLOTS = True
    VERBOSE = True
    
    print("\n" + "=" * 85)
    print("LEROY'S GMM SEGMENTATION PIPELINE")
    print("=" * 85)
    print("Following Prof's advice:")
    print("  - Fit GMM on pixel colors")
    print("  - Run multiple initializations")
    print("  - Use pseudo-log-likelihood to select best model")
    print("  - Get P(foreground | pixel) probabilities")
    print("=" * 85)
    print(f"Images: {len(files_to_process)}")
    print(f"Component configs: {component_experiments}")
    print(f"GMM runs per image: {n_runs}")
    print("=" * 85)
    
    for n_components in component_experiments:
        print(f"\n\n>>> RUNNING EXPERIMENT: n_components = {n_components} <<<")
        all_results = []
        
        for filepath in files_to_process:
            result = segment_image_gmm(
                filepath,
                n_components=n_components,
                n_runs=n_runs,
                prob_threshold=0.5,
                use_gaussian=True,
                save_plot=SAVE_PLOTS,
                verbose=VERBOSE
            )
            if result:
                all_results.append(result)
        
        # Print summary table
        print_results_table(all_results, n_components)
    
    print("\n\nExperiment complete!")
    print(f"Results saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
