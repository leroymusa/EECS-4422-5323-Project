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
    # --- NO FILTERING (Best results for this dataset) ---
    return image.copy()

def segment_kmeans_rgb(image, k):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # '10' is number of random initializations (Stability)
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
    
    # --- DICE SCORE (F1 Score) ---
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    
    return accuracy, precision, recall, dice

def process_single_image(filepath, k, save_plot=True):
    original_img = load_image(filepath)
    if original_img is None: return None

    # 1. Load Ground Truth
    base_name = os.path.splitext(filepath)[0]
    base_name_gt = base_name.replace("Raw_images", "Binary_masks")
    mask_path = base_name_gt + ".tif"
    if not os.path.exists(mask_path): mask_path = base_name_gt + ".png"
    gt_mask = load_ground_truth(mask_path, original_img.shape)

    # 2. Pipeline
    processed_rgb = preprocess_image_rgb(original_img)
    kmeans_mask, centers, labels = segment_kmeans_rgb(processed_rgb, k=k)
    final_mask = cleanup_mask(kmeans_mask)
    
    # 3. Metrics
    metrics = None
    if gt_mask is not None:
        acc, prec, rec, dice = evaluate_performance(final_mask, gt_mask)
        metrics = (os.path.basename(filepath), acc, prec, rec, dice)
    else:
        metrics = (os.path.basename(filepath), 0.0, 0.0, 0.0, 0.0)

    # 4. Visualize & Save
    if save_plot:
        # Determine save path relative to input file
        input_dir = os.path.dirname(filepath)
        save_dir = os.path.join(input_dir, 'pictures')
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        save_filename = f"{filename_no_ext}_k{k}.png"
        save_path = os.path.join(save_dir, save_filename)
        
        visualize_results(original_img, processed_rgb, final_mask, gt_mask, centers, k, os.path.basename(filepath), save_path)
    
    return metrics

def visualize_results(original, processed, mask, gt_mask, centers, k, title, save_path=None):
    luminance = 0.114 * centers[:, 0] + 0.587 * centers[:, 1] + 0.299 * centers[:, 2]
    spit_idx = np.argsort(luminance)[-1]

    fig = plt.figure(figsize=(18, 10))

    # 1. Original
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Input: {title}")
    plt.axis('off')

    # 2. Histogram (Marginal Projections)
    plt.subplot(2, 3, 2)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([original], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.6)
        for cid, center in enumerate(centers):
            style = '--' if cid == spit_idx else ':'
            plt.axvline(x=center[i], color=color, linestyle=style, alpha=0.5)
    plt.title(f"Marginal Projections (Hist) k={k}\nDashed=Spit Center")
    plt.xlim([0, 256])

    # 3. 3D Scatter Plot (Restored)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Downsample pixels for speed
    rows, cols, _ = processed.shape
    sample_size = 5000
    flat_img = processed.reshape(-1, 3)
    if rows * cols > sample_size:
        indices = np.random.choice(rows * cols, sample_size, replace=False)
    else:
        indices = np.arange(rows * cols)
    
    sample_pixels = flat_img[indices]
    sample_colors = sample_pixels[:, [2, 1, 0]] / 255.0 # BGR to RGB
    
    ax3.scatter(sample_pixels[:, 2], sample_pixels[:, 1], sample_pixels[:, 0], 
                  c=sample_colors, marker='.', s=1, alpha=0.3)
    
    # Plot Centers
    for i in range(len(centers)):
        center_rgb = centers[i][::-1] / 255.0
        label = "Spit" if i == spit_idx else f"C{i}"
        ax3.scatter(centers[i][2], centers[i][1], centers[i][0], 
                    c=[center_rgb], marker='X', s=200, edgecolor='black', label=label)
    
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_zlabel('Blue')
    ax3.set_title(f"3D Cluster Cloud")
    ax3.legend()

    # 4. Prediction
    plt.subplot(2, 3, 4)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Prediction (k={k})")
    plt.axis('off')

    # 5. Ground Truth
    if gt_mask is not None:
        plt.subplot(2, 3, 5)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path)
        plt.close(fig) # Close figure to free memory
    else:
        plt.show()

def main():
    # --- CONFIG ---
    files_to_process = [f'Saliva_Segmentation_dataset/Raw_images/{i}.tif' for i in range(1, 11)]
    k_experiments = [2, 3, 4, 5]
    
    # Set to True to save plots to 'pictures' subfolder
    SAVE_PLOTS = True 

    for k in k_experiments:
        print(f"\n\n>>> RUNNING EXPERIMENT: K = {k} <<<")
        all_metrics = []
        
        for filename in files_to_process:
            result = process_single_image(filename, k=k, save_plot=SAVE_PLOTS)
            if result:
                all_metrics.append(result)

        # --- PRINT TABLE ---
        print("\n" + "="*75)
        print(f"RESULTS FOR K={k}")
        print("-" * 75)
        print(f"{'IMAGE':<20} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'DICE':<10}")
        print("-" * 75)
        
        acc_sum, prec_sum, rec_sum, dice_sum = 0, 0, 0, 0
        count = 0

        for name, acc, prec, rec, dice in all_metrics:
            print(f"{name:<20} | {acc:<10.3f} | {prec:<10.3f} | {rec:<10.3f} | {dice:<10.3f}")
            if acc > 0:
                acc_sum += acc
                prec_sum += prec
                rec_sum += rec
                dice_sum += dice
                count += 1
                
        print("-" * 75)
        if count > 0:
            print(f"{'AVERAGE':<20} | {acc_sum/count:<10.3f} | {prec_sum/count:<10.3f} | {rec_sum/count:<10.3f} | {dice_sum/count:<10.3f}")
        print("="*75)

if __name__ == "__main__":
    main()