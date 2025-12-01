import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def load_image_and_mask(image_path, mask_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, 0)            # grayscale mask (0 or 255)
    # mask = cv2.rotate(mask, cv2.ROTATE_180)
    mask= cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)       # convert to 0/1
    return img, mask

def extract_pixels(img, mask):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    saliva_pixels = np.column_stack((
        R[mask==1], 
        G[mask==1], 
        B[mask==1]
    ))
    
    bg_pixels = np.column_stack((
        R[mask==0],
        G[mask==0],
        B[mask==0]
    ))
    
    return saliva_pixels, bg_pixels

def plot_histograms(saliva, bg):
    colors = ["Red", "Green", "Blue"]
    
    for i, color in enumerate(colors):
        plt.figure()
        plt.hist(saliva[:,i], bins=50, label=f"Saliva {color}")
        plt.hist(bg[:,i], bins=50, alpha=0.5, label=f"Background {color}")
        plt.title(f"{color} Channel Histogram")
        plt.legend()
        plt.savefig(f"Histograms/{color}_hist.png")

def gaussian(x, mean, std):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-(x-mean)**2/(2*std**2))

def plot_gaussian_distributions(saliva, bg):
    channels = ["Red", "Green", "Blue"]
    for i, channel in enumerate(channels):
        saliva_vals = saliva[:, i]
        bg_vals     = bg[:, i]

        mean_s = saliva_vals.mean()
        std_s  = saliva_vals.std()
        mean_b = bg_vals.mean()
        std_b  = bg_vals.std()

        # print(f"{channel} Channel:")
        # print(f"   Saliva mean={mean_s:.2f}, std={std_s:.2f}")
        # print(f"   Background mean={mean_b:.2f}, std={std_b:.2f}")

        # histogram
        plt.figure(figsize=(8,4))
        # gaussian x-range
        x = np.linspace(0, 255, 500)
        g_s = gaussian(x, mean_s, std_s) * len(saliva)
        g_b = gaussian(x, mean_b, std_b) * len(bg)

        plt.plot(x, g_s, 'r-', linewidth=2, label="Saliva Gaussian")
        plt.plot(x, g_b, 'b-', linewidth=2, label="Background Gaussian")

        plt.title(f"{channel} — Gaussian fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Histograms/Gaussian2_{channel}.png", dpi=300)

def preprocess_image_rgb(image):
    # # Median Blur: Good for removing hot pixels (salt-and-pepper)
    # processed = cv2.medianBlur(image, 3)
    # processed = cv2.GaussianBlur(processed, (5, 5), 1)

    # # Gaussian Blur: Good for smoothing color transitions
    processed = cv2.GaussianBlur(image, (5, 5), 1)
    return processed

def postprocess_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    # fill small holes inside saliva blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # remove small noise blobs
    return mask


def segment_with_thresholds(img, thresholds):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    mask_pred = (
        (R > thresholds["R"]) |
        (G > thresholds["G"]) |
        (B > thresholds["B"])
    ).astype(np.uint8)
    
    return mask_pred

def compute_thresholds(saliva, bg):
    thresholds = {}
    channels = ["R","G","B"]
    
    for i, ch in enumerate(channels):
        mean_saliva = saliva[:,i].mean()
        mean_bg = bg[:,i].mean()
        thresholds[ch] = (mean_saliva + mean_bg) / 2
    
    return thresholds

def find_optimal_threshold(channel_saliva, channel_bg):
    X = np.concatenate([channel_saliva, channel_bg])
    y_mask = np.concatenate([np.ones(len(channel_saliva)), np.zeros(len(channel_bg))])
    
    thresholds = np.unique(X)
    best_t = None
    best_score = -1

    for t in thresholds:
        y_pred = (X > t).astype(int)
        precision, recall, accuracy, dice = evaluate(y_pred, y_mask)
        # score = 2 * precision * recall / (precision + recall + 1e-8)        # F1 score
        score = dice
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score

def evaluate(pred, mask):
    TP = np.sum((pred==1) & (mask==1))
    TN = np.sum((pred==0) & (mask==0))
    FP = np.sum((pred==1) & (mask==0))
    FN = np.sum((pred==0) & (mask==1))
    
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    dice = 2*TP/(2*TP + FP + FN)
    return precision, recall, accuracy, dice

def write_summary(all_metrics):
    print("="*60)
    print(f"{'IMAGE':<20} | {'PRECISION':<10} | {'RECALL':<10} | {'ACCURACY':<10} | {'DICE':<10}")
    print("-" * 60)
    
    prec_sum, rec_sum, acc_sum, dice_sum = 0, 0, 0, 0
    count = 0
    for name, prec, rec, acc, dice in all_metrics:
        # Only count if GT existed (non-zero stats)
        # Use a small epsilon check or just check if acc > 0
        print(f"{name:<20} | {prec:<10.3f} | {rec:<10.3f} | {acc:<10.3f} | {dice:<10.3f}")
        if acc > 0:
            acc_sum += acc
            prec_sum += prec
            rec_sum += rec
            dice_sum += dice
            count += 1 

    print("-" * 60)
    if count > 0:
        print(f"{'AVERAGE':<20} | {prec_sum/count:<10.3f} | {rec_sum/count:<10.3f} | {acc_sum/count:<10.3f} | {dice_sum/count:<10.3f}")
        print("="*60 + "\n")

def evaluate_result(image_paths, mask_paths, thresholds):
    metrics = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        img, mask = load_image_and_mask(image_path, mask_path)
        img = preprocess_image_rgb(img)
        pred_mask = segment_with_thresholds(img, thresholds)
        pred_mask = postprocess_mask(pred_mask)
        precision, recall, accuracy, dice = evaluate(pred_mask, mask)
        metrics.append([os.path.basename(mask_path), precision, recall, accuracy, dice])
    return metrics

def do_testing(test_image_paths, test_mask_paths, thresholds):
    ind = 9
    for test_image_path, test_mask_path in zip(test_image_paths, test_mask_paths):
        test_img, test_mask = load_image_and_mask(test_image_path, test_mask_path)
        test_img = preprocess_image_rgb(test_img)
        pred_mask = segment_with_thresholds(test_img, thresholds)
        pred_mask = postprocess_mask(pred_mask)
        precision, recall, accuracy, dice = evaluate(pred_mask, test_mask)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(test_img)
        plt.title(f"Original Image {test_image_path}")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.subplot(1,3,3)
        plt.axis('off')  # remove axis lines
        plt.imshow(np.ones_like(pred_mask) * 255, cmap='gray', vmin=0, vmax=255)
        plt.text(1000, 1000,  f"Accuracy : {accuracy:.3f}",  color='black', fontsize=12)
        plt.text(1000, 1500,  f"Precision: {precision:.3f}", color='black', fontsize=12)
        plt.text(1000, 2000,  f"Recall   : {recall:.3f}",    color='black', fontsize=12)
        plt.text(1000, 2500,  f"Dice     : {dice:.3f}",    color='black', fontsize=12)
        plt.savefig(f"output/optimal_dice_{ind}.png", dpi=300)
        ind += 1

def main():
    start = time.time()
    raw_images = [f'Raw_images/{i}.tif' for i in range(1, 11)]
    binary_masks = [f'Cleaned_binary_masks/{i}.tif' for i in range(1, 11)]

    train_image_paths = raw_images[:8]
    train_mask_paths = binary_masks[:8]

    test_image_paths = raw_images[-2:]
    test_mask_paths = binary_masks[-2:]

    # --- COLLECT TRAINING PIXELS ---
    all_saliva = []
    all_bg = []
    for img_path, mask_path in zip(train_image_paths, train_mask_paths):
        img, mask = load_image_and_mask(img_path, mask_path)
        img = preprocess_image_rgb(img)
        saliva, bg = extract_pixels(img, mask)
        all_saliva.append(saliva)
        all_bg.append(bg)
    all_saliva = np.vstack(all_saliva)
    all_bg     = np.vstack(all_bg)

    # --- PLOT HISTOGRAMS ---
    # plot_histograms(all_saliva, all_bg)
    # plot_gaussian_distributions(all_saliva, all_bg)
    
    # --- COMPUTE THRESHOLDS ---
    print("Find threshold")
    t_R, score_R = find_optimal_threshold(all_saliva[:,0], all_bg[:,0])
    t_G, score_G = find_optimal_threshold(all_saliva[:,1], all_bg[:,1])
    t_B, score_B = find_optimal_threshold(all_saliva[:,2], all_bg[:,2])

    print("Optimal thresholds:")
    print("R:", t_R, "score:", score_R)
    print("G:", t_G, "score:", score_G)
    print("B:", t_B, "score:", score_B)
    print(f"Runtime: {(time.time()-start):.3f}s")

    # --- APPLY TO TEST MATERIAL ---
    thresholds = {"R": t_R, "G": t_G, "B": t_B}
    do_testing(test_image_paths, test_mask_paths, thresholds)

    # --- PRINT SUMMARY TABLE ---
    training_metrics = evaluate_result(train_image_paths, train_mask_paths, thresholds)
    print("EVALUATE TRAINING SET:")
    write_summary(training_metrics)
    testing_metrics = evaluate_result(test_image_paths, test_mask_paths, thresholds)
    print("EVALUATE TESTING SET:")
    write_summary(testing_metrics)
    print(f"Runtime: {(time.time()-start):.3f}s")

if __name__ == "__main__":
    main()