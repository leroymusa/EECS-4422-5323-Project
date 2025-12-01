import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

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
def preprocess_image_rgb(image):
    # # Median Blur: Good for removing hot pixels (salt-and-pepper)
    # processed = cv2.medianBlur(image, 3)

    # # Gaussian Blur: Good for smoothing color transitions
    processed = cv2.GaussianBlur(image, (5, 5), 1)
    return processed

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

# After you have all_saliva and all_bg as (N×3) arrays
df_saliva = pd.DataFrame(all_saliva[:100000], columns=['R', 'G', 'B'])  # subsample if too big
df_bg     = pd.DataFrame(all_bg[:100000],     columns=['R', 'G', 'B'])

corr_saliva = df_saliva.corr()
corr_bg     = df_bg.corr()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.heatmap(corr_saliva, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Saliva pixels')

plt.subplot(1,2,2)
sns.heatmap(corr_bg, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Background pixels')
plt.tight_layout()
plt.savefig('Histograms/rgb_correlation_heatmap.png', dpi=300)
plt.show()