#!/usr/bin/env python3
"""
Generate high-quality standalone mask images and 3D plots for Method D.
Publication-ready visualizations with professional styling.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))

from leroy_gmm import (
    load_image, fit_gmm_best_of_n, identify_foreground_component,
    cleanup_mask
)

# Professional styling with high-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Professional color palette
COLORS = {
    'fg': '#E74C3C',      # Red for foreground
    'bg': '#3498DB',      # Blue for background
    'bg2': '#95A5A6',     # Gray for background 2
    'bg3': '#34495E',     # Dark gray for background 3
    'text': '#2C3E50',    # Dark text
    'grid': '#ECF0F1',    # Light grid
}

def create_standalone_mask(img_num, n_components, output_dir):
    """Create a high-quality standalone mask image."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(project_root, 
                           'Saliva_Segmentation_dataset', 'Raw_images', f'{img_num}.tif')
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    original = load_image(img_path)
    if original is None:
        return
    
    pixels = original.reshape(-1, 3) / 255.0
    h, w = original.shape[:2]
    
    print(f"Fitting GMM for K={n_components}...")
    gmm, pll, _ = fit_gmm_best_of_n(pixels, n_components=n_components, n_runs=10, verbose=False)
    fg_comp = identify_foreground_component(gmm)
    prob_map = gmm.predict_proba(pixels)[:, fg_comp].reshape(h, w)
    mask = (prob_map > 0.5).astype(float)
    mask = cleanup_mask(mask)
    
    # Create high-quality figure with side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), facecolor='white')
    
    # Left: Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=13, fontweight='bold', pad=8, color=COLORS['text'])
    axes[0].axis('off')
    
    # Right: Binary mask with better contrast
    im = axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    axes[1].set_title(f'K={n_components}: Segmentation Mask', fontsize=13, fontweight='bold', pad=8, color=COLORS['text'])
    axes[1].axis('off')
    
    # Add subtle border
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout(pad=0.5)
    
    output_path = os.path.join(output_dir, f'method_d_mask_k{n_components}.pdf')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.05, dpi=400)
    print(f"Saved mask to: {output_path}")
    plt.close()

def create_standalone_3d_plot(img_num, n_components, output_dir):
    """Create a high-quality standalone 3D RGB scatter plot."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(project_root, 
                           'Saliva_Segmentation_dataset', 'Raw_images', f'{img_num}.tif')
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    original = load_image(img_path)
    if original is None:
        return
    
    pixels = original.reshape(-1, 3)
    h, w = original.shape[:2]
    
    print(f"Fitting GMM for K={n_components} (3D plot)...")
    gmm, pll, _ = fit_gmm_best_of_n(pixels / 255.0, n_components=n_components, n_runs=10, verbose=False)
    fg_comp = identify_foreground_component(gmm)
    means = gmm.means_ * 255.0  # Convert back to 0-255 range
    
    # Downsample pixels for cleaner visualization
    sample_size = 2000
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
    else:
        indices = np.arange(len(pixels))
    
    sample_pixels = pixels[indices]
    sample_colors = sample_pixels / 255.0
    
    # Create high-quality 3D figure
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sampled pixels with better rendering
    ax.scatter(sample_pixels[:, 0], sample_pixels[:, 1], sample_pixels[:, 2],
               c=sample_colors, marker='.', s=0.8, alpha=0.4, edgecolors='none', rasterized=True)
    
    # Define colors for components
    bg_colors = [COLORS['bg'], COLORS['bg2'], COLORS['bg3'], '#9B59B6']
    
    # Plot GMM centers with professional styling
    for i, mean in enumerate(means):
        center_rgb = mean / 255.0
        if i == fg_comp:
            label = "Saliva (FG)"
            marker = 'X'
            size = 250
            color = COLORS['fg']
            edgecolor = 'black'
            linewidth = 2.5
        else:
            label = f"BG-{i}"
            marker = 'o'
            size = 120
            color = bg_colors[i % len(bg_colors)]
            edgecolor = 'white'
            linewidth = 2
        
        ax.scatter(mean[0], mean[1], mean[2],
                  c=[color], marker=marker, s=size,
                  edgecolor=edgecolor, linewidth=linewidth, 
                  label=label, zorder=10)
    
    # Professional axis styling - MUCH MORE padding for Blue z-axis
    ax.set_xlabel('Red', fontsize=12, fontweight='bold', labelpad=5, color=COLORS['text'])
    ax.set_ylabel('Green', fontsize=12, fontweight='bold', labelpad=5, color=COLORS['text'])
    ax.set_zlabel('Blue', fontsize=12, fontweight='bold', labelpad=25, color=COLORS['text'])  # MUCH MORE padding for Blue
    
    # Format title
    ax.set_title(f'K={n_components}: RGB Space\nPLL={pll:.0f}', 
                fontsize=13, fontweight='bold', pad=10, color=COLORS['text'])
    
    # Professional legend
    legend = ax.legend(loc='upper left', fontsize=9, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.95,
                      edgecolor='gray', facecolor='white')
    legend.get_frame().set_linewidth(1.0)
    
    # Set equal aspect ratio
    max_range = np.array([sample_pixels[:, 0].max() - sample_pixels[:, 0].min(),
                         sample_pixels[:, 1].max() - sample_pixels[:, 1].min(),
                         sample_pixels[:, 2].max() - sample_pixels[:, 2].min()]).max() / 2.0
    mid_x = (sample_pixels[:, 0].max() + sample_pixels[:, 0].min()) * 0.5
    mid_y = (sample_pixels[:, 1].max() + sample_pixels[:, 1].min()) * 0.5
    mid_z = (sample_pixels[:, 2].max() + sample_pixels[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Professional grid and background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Better viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Adjust layout with MUCH MORE padding for Blue label
    plt.tight_layout(pad=3.0)
    
    # Adjust subplot parameters to give more space for z-axis label
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    output_path = os.path.join(output_dir, f'method_d_3d_k{n_components}.pdf')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.4, dpi=400)  # MUCH MORE padding for Blue label
    print(f"Saved 3D plot to: {output_path}")
    plt.close()

def main():
    """Generate all standalone images."""
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'leroy_gmm_results')
    os.makedirs(output_dir, exist_ok=True)
    
    img_num = 1
    
    # Generate masks
    print("\n=== Generating High-Quality Masks ===")
    create_standalone_mask(img_num, 3, output_dir)
    create_standalone_mask(img_num, 4, output_dir)
    
    # Generate 3D plots
    print("\n=== Generating High-Quality 3D Plots ===")
    create_standalone_3d_plot(img_num, 3, output_dir)
    create_standalone_3d_plot(img_num, 4, output_dir)
    
    print("\n✓ All high-quality images generated!")

if __name__ == '__main__':
    main()
