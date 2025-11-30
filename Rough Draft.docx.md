# Rough draft EECS 5323/4422

## Team 2 – Image Segmentation

# 1\. Introduction

## 1.1 Problem Definition

### Hospital infection

### Saliva as a vector for spreading infection

### Prevention methods

#### *Face masks*

#### *Face shields*

#### *Ambient cleaning/sterilization*

##### Automated cleaning using computer vision

## 1.2 Literature Review

### Medical image segmentation

#### *Definition*

#### *Techniques used*

#### *Challenges*

### Gap

## 1.3 Contributions 

# 2\. Method

## 2.1 Dataset Description

- 10 UV-C images from saliva over different substrates, both wet and dry.  
- Substrates: Aluminum, Steel, Whitewood, PVC and Wood.  
- 10 Ground Truth images representing binary masks already done on this set.  
- All Ground Truth images carry 4 foreground regions roughly centralized  
- When using machine learning methods, the first 8 images were considered part of the Training Set, and the remaining 2 images composed the Test Set.

## 2.2 Methods used

### Method A

	Method A is based on MATLAB and its computer vision libraries. The idea behind this method is to create a binary mask that identifies saliva spots with high recall, using a simple and fast technique that utilizes resources already present in MATLAB. This method will convert the images to grayscale, enhance their contrast, train a model to create binary masks, and apply what it has learned to the test set.

Pre-processing

			Luminance conversion

	The raw images, heavily red due to the UV-C setting, were imported to MATLAB and converted to luminance (black and white). The recommended ITU-R BT.601 weights to each colour channel (0.299\*Red, 0.587\*Green and 0.114\*Blue) were applied, yielding a visually correct grayscale image.

			CLAHE

	After this conversion, the images were treated with Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance their contrast and facilitate the identification of a suitable luminance threshold. This method was used to obtain a locally optimized histogram equalization, while avoiding noise amplification.

Ground Truth evaluation

	After treating the input image, the code turns to the corresponding ground truth file. It performs an analysis of the number of white pixels the ground truth possesses and divides this quantity by the number of white areas to obtain a plausible area range to search for when segmenting the input image. This step helps filter out elements with similar luminances but that are either\>50% too small (noise) or \>50% too large (other image objects).

		Training

			Otsu bias tuning

	Otsu’s thresholding technique automatically classifies pixels in a luminance image as foreground (white) or background (black) depending on a threshold. Its mathematical formulation calculates the threshold that better minimizes the intra-class variance between the pixels in each class. Although it is a great method for images that generate a bimodal histogram (as those present in the dataset), it can suffer from sources with uneven lighting and end up underestimating the foreground pixels. To compensate for this effect, Method A employs a set of optimizations to produce masks that more accurately recreate the ground truth. The first one involves training a bias factor to be combined with the original threshold. This bias lowers or increases Otsu’s threshold, resulting in more or fewer white pixels in the binary mask. After this phase, a plausibility filter is used to maintain on the binary mask only the regions that have a plausible area as the saliva patches, and are roughly centralized. Their distance from the geometric center of the four ground truth saliva patches is measured, and closer regions are rewarded. After this filtering, the bias that maximizes the Dice Similarity Coefficient (DICE) between all binary masks from the training set and their respective ground truths is chosen. The DICE coefficient was chosen to combine precision and recall into a single, simple, and comparable measurement across methods.

		Testing

Otsu thresholding				

Area filter

Centrality filter

Morphological cleanup

		Segmented images

### Method B

### Method C

Method C looks at using k-means to distinguish between saliva pixels and background pixels. 

#### *Pre-processing* 

- Did not pre-process images with Gaussian/Median due to an improvement with no pre-processing   
- All tables below are with k=3   
- Below table: *With pre-processing* 

| Image | Accuracy | Precision | Recall | DICE Score |
| :---- | :---- | :---- | :---- | :---- |
| 1.tif | 0.996 | 0.911 | 0.899 | 0.905 |
| 2.tif | 0.838 | 0.089 | 0.821 | 0.161 |
| 3.tif | 0.990 | 0.796 | 0.950 | 0.866 |
| 4.tif | 0.959 | 0.428 | 0.896 | 0.579 |
| 5.tif | 0.813 | 0.094 | 0.994 | 0.172 |
| 6.tif | 0.778 | 0.104 | 0.984 | 0.188 |
| 7.tif | 0.987 | 0.670 | 0.920 | 0.775 |
| 8.tif | 0.947 | 0.377 | 0.979 | 0.544 |
| 9.tif | 0.980 | 0.549 | 0.989 | 0.706 |
| 10.tif | 0.962 | 0.352 | 0.833 | 0.495 |
| **AVERAGE** | **0.925** | **0.437** | **0.926** | **0.539** |

- Below table: *Without pre-processing* (better result) 

| Image | Accuracy | Precision | Recall | DICE Score |
| :---- | :---- | :---- | :---- | :---- |
| 1.tif | 0.997 | 0.937 | 0.896 | 0.916 |
| 2.tif | 0.844 | 0.092 | 0.820 | 0.165 |
| 3.tif | 0.990 | 0.813 | 0.949 | 0.876 |
| 4.tif | 0.960 | 0.432 | 0.895 | 0.583 |
| 5.tif | 0.818 | 0.096 | 0.995 | 0.175 |
| 6.tif | 0.779 | 0.105 | 0.985 | 0.190 |
| 7.tif | 0.987 | 0.671 | 0.923 | 0.777 |
| 8.tif | 0.949 | 0.387 | 0.977 | 0.554 |
| 9.tif | 0.981 | 0.552 | 0.989 | 0.709 |
| 10.tif | 0.963 | 0.358 | 0.820 | 0.498 |
| **AVERAGE** | **0.927** | **0.444** | **0.925** | **0.544** |

#### *K-means* 

- Every pixel has RGB value   
- K-means is run 10 times, best result is kept   
  - Best result determined through ‘compactness’ metric \- whichever cluster is smallest   
- Centre of each cluster is found   
  - To determine which cluster is spit, perceived luminance is used to compare brightness\!   
    - 0.114\*Blue \+ 0.587\*Green \+ 0.299\*Red  
  - Brightest centre belongs to spit cluster   
- Then, all pixels belonging to the spit cluster in the original image are assigned a value of 255, all other values get 0 (creates mask) 

#### *Post-processing* 

- Morphological filters (close, open) to remove small white spots and small dark spots 

#### *Discussion* 

- K-means cluster 2, 3, 4, 5 were tested   
  - 2 performs worst at precision   
  - 3 is a good balance   
  - 4 is also a good balance, but starts to lose recall   
  - 5 loses too much recall 

### Method D

Method D builds a Gaussian Mixture Model (GMM) in RGB space to separate saliva from background. Each pixel (R,G,B) is treated as a sample from a mixture of K Gaussians, with K = 3 or 4 matching the most competitive clusterings from Method C. The EM algorithm (via `sklearn.mixture.GaussianMixture`) is run multiple times per image, so that different initializations and the extra components can be evaluated by likelihood rather than by a single hard assignment.

#### Pre-processing

- Pixels remain in RGB space, and the images are reshaped into long pixel arrays so the GMM learns color distributions directly.
- Cluster centers extracted from the k-means pipeline provide candidate initial means, ensuring the probabilistic model starts near plausible saliva/background separations.
- No additional smoothing or filtering is applied prior to fitting, keeping the preprocessing comparable to the other methods.

#### Training and model selection

- For each image and each value of K, n_runs = 10 independent GMM fits are executed. The first few runs are seeded with k-means centers, while later runs use random seeds to explore other local optima.
- Each run produces a pseudo-log-likelihood equal to the sum over all pixels of log p(x_i | Theta) on the training data. The model with the highest pseudo-log-likelihood is chosen, following the strategy to “select the most probable solution” when EM only guarantees convergence to a local maximum.
- Posterior probabilities P(z_i = k | x_i, Theta*) are used to create a foreground probability map. The brightest component (luminance 0.299 R + 0.587 G + 0.114 B) is designated as saliva.

#### Mask construction and evaluation

- The foreground probability map is thresholded at 0.5 and then subjected to morphological opening and closing to remove noise and fill small holes.
- The resulting mask is evaluated against the ground truth using accuracy, precision, recall, and Dice. Per-image metrics and pseudo-log-likelihoods are tracked for the report, and visualizations are exported for reference.

This GMM procedure turns the hard k-means clusters from Method C into a soft segmentation model, quantifies uncertainty with posterior probabilities, and makes model selection explicit through pseudo-log-likelihood comparisons.

# 3\. Results

## Cross-comparison of precision/recall

| Method | K | Accuracy | Precision | Recall | Dice |
| :---- | :----: | :----: | :----: | :----: | :----: |
| Method C (k-means, no pre-processing) | 3 | 0.927 | 0.444 | 0.925 | 0.544 |
| Method D (GMM) | 3 | 0.930 | 0.282 | 0.992 | 0.432 |
| Method D (GMM) | 4 | 0.959 | 0.419 | 0.964 | 0.564 |

The probabilistic GMM favors recall more than the hard k-means solution, especially for K = 3, because the model absorbs a larger share of the bright saliva pixels that the thresholded clusters sometimes treat as background. Increasing to K = 4 regains precision and raises Dice beyond the k-means baseline while keeping recall high. The best pseudo-log-likelihood per image also improves for K = 4 (≈ -2.77 million vs. -2.92 million), suggesting that the extra component better explains the color variation in the scene.

# 4\. Discussion

# 5\. Conclusion

# 6\. References