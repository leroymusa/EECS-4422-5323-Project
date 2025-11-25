import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os

# --- LIBRARY IMPORTS ---
try:
    import rawpy
except ImportError:
    rawpy = None
try:
    import tifffile
except ImportError:
    tifffile = None

def load_image(filepath, target_width=800):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    file_ext = os.path.splitext(filepath)[1].lower()
    img = None

    if file_ext == '.arw':
        if rawpy:
            try:
                with rawpy.imread(filepath) as raw:
                    img = raw.postprocess()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"rawpy error: {e}")
        elif tifffile:
            try:
                raw_data = tifffile.imread(filepath)
                if raw_data.dtype == np.uint16:
                    raw_data = (raw_data / 256).astype(np.uint8)
                img = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2BGR)
            except Exception as e:
                print(f"tifffile error: {e}")
    else:
        img = cv2.imread(filepath)

    if img is None: return None

    h, w = img.shape[:2]
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (target_width, new_h))
    return img

def load_ground_truth(filepath, target_shape):
    if not os.path.exists(filepath): return None
    gt = cv2.imread(filepath, 0) 
    if gt is None: return None
    gt = cv2.resize(gt, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    return gt

def mask_out_distractors(image, regions):
    masked_img = image.copy()
    for (x, y, w, h) in regions:
        cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return masked_img

def preprocess_image_rgb(image):
    # --- STEP 1: RESTORED FILTERING ---
    # We are NOT doing Gamma/CLAHE (Contrast Boosting) because it hurts separation.
    # We ARE doing Blurring to remove sensor noise and smooth the mask.
    
    # Median Blur: Good for removing hot pixels (salt-and-pepper)
    processed = cv2.medianBlur(image, 5)
    
    # Gaussian Blur: Good for smoothing color transitions
    processed = cv2.GaussianBlur(processed, (5, 5), 0)
    
    return processed

def segment_kmeans_rgb(image, k=3):
    # k=3 splits data into: [Shadows], [Tissue/Tray], [Bright Spit]
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    
    # Determine Spit Cluster by max brightness
    luminance = 0.114 * centers[:, 0] + 0.587 * centers[:, 1] + 0.299 * centers[:, 2]
    spit_cluster_index = np.argsort(luminance)[-1]
    
    mask = (labels.flatten() == spit_cluster_index).reshape(image.shape[:2]).astype(np.uint8) * 255
    return mask, centers, labels

def cleanup_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    # Tiny bit of cleanup to remove single hot pixels
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned

def evaluate_performance(pred_mask, true_mask):
    p = pred_mask.flatten() / 255
    t = true_mask.flatten() / 255
    TP = np.sum((p == 1) & (t == 1))
    TN = np.sum((p == 0) & (t == 0))
    FP = np.sum((p == 1) & (t == 0))
    FN = np.sum((p == 0) & (t == 1))
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return accuracy, precision, recall

def process_single_image(filepath, crop_config, blackout_regions):
    print(f"\n--- Processing: {filepath} ---")
    original_img = load_image(filepath)

    if original_img is None: return

    # 1. Crop
    do_crop, coords = crop_config
    if do_crop:
        y1, y2, x1, x2 = coords
        original_img = original_img[y1:y2, x1:x2]

    # 2. Blackout
    if blackout_regions:
        original_img = mask_out_distractors(original_img, blackout_regions)

    # 3. Ground Truth
    base_name = os.path.splitext(filepath)[0]
    mask_path = base_name + ".tif" 
    gt_mask = load_ground_truth(mask_path, original_img.shape)

    # 4. Pipeline
    processed_rgb = preprocess_image_rgb(original_img)
    # UPDATED: k=3 to separate Spit from Light Background
    kmeans_mask, centers, labels = segment_kmeans_rgb(processed_rgb, k=3)
    final_mask = cleanup_mask(kmeans_mask)
    
    # 5. Metrics
    metrics_text = "Coverage: {:.2f}%".format(np.count_nonzero(final_mask)/final_mask.size*100)
    if gt_mask is not None:
        acc, prec, rec = evaluate_performance(final_mask, gt_mask)
        metrics_text += f"\nAcc: {acc:.3f}\nPrec: {prec:.3f}\nRec: {rec:.3f}"
        print(f"METRICS: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")
    else:
        print("No ground truth mask found.")

    # 6. Visualize
    visualize_results(original_img, processed_rgb, final_mask, gt_mask, centers, filepath, metrics_text)

def visualize_results(original, processed, mask, gt_mask, centers, title, metrics_text):
    luminance = 0.114 * centers[:, 0] + 0.587 * centers[:, 1] + 0.299 * centers[:, 2]
    spit_idx = np.argsort(luminance)[-1]
    
    # ADDED: Logic to determine columns based on GT and add one for 3D plot
    cols_base = 4 if gt_mask is not None else 3
    cols = cols_base + 1
    
    fig = plt.figure(figsize=(5*cols, 6))

    # Input
    ax1 = fig.add_subplot(1, cols, 1)
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Input\n{os.path.basename(title)}")
    ax1.axis('on')

    # Processed
    ax2 = fig.add_subplot(1, cols, 2)
    ax2.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    ax2.set_title("Blurred (Median+Gauss)")
    ax2.axis('off')

    # Prediction
    ax3 = fig.add_subplot(1, cols, 3)
    ax3.imshow(mask, cmap='gray')
    ax3.set_title("Predicted Mask (k=3)")
    plt.text(10, mask.shape[0] - 20, metrics_text, color='white', 
             fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    ax3.axis('off')

    # Ground Truth
    next_plot_idx = 4
    if gt_mask is not None:
        ax4 = fig.add_subplot(1, cols, 4)
        ax4.imshow(gt_mask, cmap='gray')
        ax4.set_title("Ground Truth")
        ax4.axis('off')
        next_plot_idx = 5

    # 3D Scatter
    ax5 = fig.add_subplot(1, cols, next_plot_idx, projection='3d')
    rows, c, _ = processed.shape
    sample_size = 5000
    flat_img = processed.reshape(-1, 3)
    if rows * c > sample_size: indices = np.random.choice(rows * c, sample_size, replace=False)
    else: indices = np.arange(rows * c)
    sample_pixels = flat_img[indices]
    sample_colors = sample_pixels[:, [2, 1, 0]] / 255.0
    
    ax5.scatter(sample_pixels[:, 2], sample_pixels[:, 1], sample_pixels[:, 0], 
                  c=sample_colors, marker='.', s=1, alpha=0.3)
    for i in range(len(centers)):
        center_rgb = centers[i][::-1] / 255.0
        label = "Spit" if i == spit_idx else f"C{i}"
        ax5.scatter(centers[i][2], centers[i][1], centers[i][0], 
                    c=[center_rgb], marker='X', s=200, edgecolor='black', label=label)
    ax5.set_title(f"3D Clusters")
    ax5.legend()

    plt.tight_layout()
    plt.show()

def main():
    files_to_process = ['amely_unsupervised_work/1.ARW', 'amely_unsupervised_work/2.ARW', 'amely_unsupervised_work/3.ARW', 'amely_unsupervised_work/4.ARW', 'amely_unsupervised_work/5.ARW'] 
    DO_CROP = False
    CROP_COORDS = (200, 600, 200, 600) 
    BLACKOUT_REGIONS = [] 

    for filename in files_to_process:
        process_single_image(filename, (DO_CROP, CROP_COORDS), BLACKOUT_REGIONS)

if __name__ == "__main__":
    main()