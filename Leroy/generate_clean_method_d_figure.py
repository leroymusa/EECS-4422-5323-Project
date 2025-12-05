#!/usr/bin/env python3
"""
Generate a clean, publication-quality figure for Method D (GMM).
Shows ONLY: Original image, probability maps, and binary masks for K=3 and K=4.
Clean layout without extra plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys

# Add path to import leroy_gmm functions
sys.path.append(os.path.dirname(__file__))

try:
    from leroy_gmm import (
        load_image, fit_gmm_best_of_n, identify_foreground_component,
        gmm_to_probability_map, cleanup_mask
    )
    HAS_GMM = True
except ImportError:
    print("Warning: Could not import leroy_gmm functions.")
    HAS_GMM = False

# Professional styling with LARGE fonts
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'axes.linewidth': 2.0,
})

COLORS = {
    'primary': '#2C3E50',
    'accent1': '#3498DB',
    'accent2': '#E74C3C',
}

def create_clean_method_d_figure():
    """Create a clean figure with just essential visualizations."""
    
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'leroy_gmm_results')
    output_dir = results_dir
    
    # Try to load actual data
    img_num = 1
    project_root = os.path.dirname(base_dir)
    img_path = os.path.join(project_root, 
                           'Saliva_Segmentation_dataset', 'Raw_images', f'{img_num}.tif')
    
    # Create VERY LARGE figure for readability
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25,
                          left=0.04, right=0.98, top=0.94, bottom=0.06)
    
    # Try to generate from actual GMM data
    use_gmm_data = False
    if HAS_GMM and os.path.exists(img_path):
        try:
            original = load_image(img_path)
            if original is not None:
                pixels = original.reshape(-1, 3) / 255.0
                h, w = original.shape[:2]
                
                print("Fitting GMM for K=3...")
                gmm_k3, pll_k3, _ = fit_gmm_best_of_n(pixels, n_components=3, n_runs=10, verbose=False)
                fg_comp_k3 = identify_foreground_component(gmm_k3)
                prob_map_k3 = gmm_k3.predict_proba(pixels)[:, fg_comp_k3].reshape(h, w)
                mask_k3 = (prob_map_k3 > 0.5).astype(float)
                mask_k3 = cleanup_mask(mask_k3)
                
                print("Fitting GMM for K=4...")
                gmm_k4, pll_k4, _ = fit_gmm_best_of_n(pixels, n_components=4, n_runs=10, verbose=False)
                fg_comp_k4 = identify_foreground_component(gmm_k4)
                prob_map_k4 = gmm_k4.predict_proba(pixels)[:, fg_comp_k4].reshape(h, w)
                mask_k4 = (prob_map_k4 > 0.5).astype(float)
                mask_k4 = cleanup_mask(mask_k4)
                
                use_gmm_data = True
                
                # Row 1: K=3 results
                # Original
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(original)
                ax1.set_title('(a) Original Image', fontsize=22, fontweight='bold', pad=20)
                ax1.axis('off')
                
                # K=3 Probability Map
                ax2 = fig.add_subplot(gs[0, 1])
                im2 = ax2.imshow(prob_map_k3, cmap='hot', vmin=0, vmax=1)
                ax2.set_title('(b) K=3: Probability Map', fontsize=22, fontweight='bold', pad=20)
                ax2.axis('off')
                cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                cbar2.ax.tick_params(labelsize=16)
                cbar2.set_label('P(Foreground)', fontsize=18, fontweight='bold')
                
                # K=3 Binary Mask
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(mask_k3, cmap='gray', vmin=0, vmax=1)
                ax3.set_title('(c) K=3: Binary Mask', fontsize=22, fontweight='bold', pad=20)
                ax3.axis('off')
                
                # Row 2: K=4 results
                # Original (same)
                ax4 = fig.add_subplot(gs[1, 0])
                ax4.imshow(original)
                ax4.set_title('(d) Original Image', fontsize=22, fontweight='bold', pad=20)
                ax4.axis('off')
                
                # K=4 Probability Map
                ax5 = fig.add_subplot(gs[1, 1])
                im5 = ax5.imshow(prob_map_k4, cmap='hot', vmin=0, vmax=1)
                ax5.set_title('(e) K=4: Probability Map', fontsize=22, fontweight='bold', pad=20)
                ax5.axis('off')
                cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
                cbar5.ax.tick_params(labelsize=16)
                cbar5.set_label('P(Foreground)', fontsize=18, fontweight='bold')
                
                # K=4 Binary Mask
                ax6 = fig.add_subplot(gs[1, 2])
                ax6.imshow(mask_k4, cmap='gray', vmin=0, vmax=1)
                ax6.set_title('(f) K=4: Binary Mask', fontsize=22, fontweight='bold', pad=20)
                ax6.axis('off')
                
        except Exception as e:
            print(f"Error generating from GMM: {e}")
            use_gmm_data = False
    
    # Fallback: use existing composite images but extract/show cleaner
    if not use_gmm_data:
        k3_path = os.path.join(results_dir, f'{img_num}_gmm_n3.png')
        k4_path = os.path.join(results_dir, f'{img_num}_gmm_n4.png')
        
        if os.path.exists(k3_path) and os.path.exists(k4_path):
            k3_img = np.array(Image.open(k3_path))
            k4_img = np.array(Image.open(k4_path))
            
            # Show full composites but larger
            ax1 = fig.add_subplot(gs[0, :])
            ax1.imshow(k3_img)
            ax1.set_title('(a) K=3: GMM Segmentation Results', fontsize=22, fontweight='bold', pad=20)
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[1, :])
            ax2.imshow(k4_img)
            ax2.set_title('(b) K=4: GMM Segmentation Results', fontsize=22, fontweight='bold', pad=20)
            ax2.axis('off')
        else:
            # Final fallback
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'GMM Visualization\n(Load images to generate)', 
                   ha='center', va='center', fontsize=28,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.axis('off')
    
    plt.suptitle('Method D: Probabilistic Gaussian Mixture Model Segmentation', 
                fontsize=26, fontweight='bold', y=0.98)
    
    # Save at HIGH resolution
    output_path = os.path.join(output_dir, 'method_d_clean_figure.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.1)
    print(f"Saved clean figure to: {output_path}")
    
    output_path_pdf = os.path.join(output_dir, 'method_d_clean_figure.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.1)
    print(f"Saved PDF to: {output_path_pdf}")
    
    plt.close()

if __name__ == '__main__':
    create_clean_method_d_figure()
