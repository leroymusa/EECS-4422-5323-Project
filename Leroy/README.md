## Leroy's GMM Segmentation Work

This folder contains all of Leroy's Gaussian Mixture Model (GMM) based segmentation implementation **for the saliva segmentation project**. This README explains, in detail, **how the method works mathematically**, how it **builds on Amely's k-means**, and **how to interpret the results**.

---

### 1. Files in this directory

- **`leroy_gmm.py`**: Main GMM segmentation pipeline.
- **`leroy_gmm_results/`**: All visualization figures for each image and each number of components.
- **`results.txt`**: Text summary of quantitative metrics (accuracy, precision, recall, Dice, log-likelihood) for each image and each number of components (3 and 4).

---

### 2. Mathematical model

#### 2.1 Pixel feature space

Each pixel in an image is represented in **RGB space** as a 3D vector
$$
\mathbf{x} = (R, G, B)^\top \in \mathbb{R}^3.
$$

We assume that each pixel belongs to one of $K$ latent classes (mixture components) such as saliva foreground, background surfaces, or highlights. Let the latent class be
$$
z \in \{1, \dots, K\}.
$$

#### 2.2 Gaussian Mixture Model (GMM)

We model the RGB distribution of all pixels in an image as a **mixture of Gaussians**
\[
p(\mathbf{x} \mid \Theta)
  = \sum_{k=1}^{K} \pi_k \,\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
\]
$$
p(\mathbf{x} \mid \Theta)
  = \sum_{k=1}^{K} \pi_k \,\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
$$
where
- $K = n_{\text{components}} \in \{3, 4\}$ in our experiments,
- $\pi_k$ are mixture weights (priors), $\pi_k \ge 0, \ \sum_k \pi_k = 1$,
- $\boldsymbol{\mu}_k \in \mathbb{R}^3$ are mean RGB vectors,
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{3\times 3}$ are full covariance matrices,
- $\Theta = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$ are all GMM parameters.

#### 2.3 Log-likelihood and pseudo-log-likelihood

Given all image pixels \(\{\mathbf{x}_i\}_{i=1}^N\), the log-likelihood is
$$
\mathcal{L}(\Theta) = \sum_{i=1}^{N} \log p(\mathbf{x}_i \mid \Theta).
$$

Following Prof's notes, we evaluate the **pseudo-log-likelihood** on the **same data used to fit the model**:
$$
\text{PLL}(\Theta)
  = \sum_{i=1}^{N} \log p(\mathbf{x}_i \mid \Theta)
  = N \cdot \underbrace{\frac{1}{N} \sum_{i=1}^N \log p(\mathbf{x}_i \mid \Theta)}_{\text{average log-likelihood per pixel}}.
$$

In the code (`fit_gmm_single`):
- `gmm.score(pixels)` returns the average log-likelihood per pixel;
- we multiply by `pixels.shape[0]` to obtain the total pseudo-log-likelihood.

This exactly matches the office-hour description: “once data is fitted, evaluate the pseudo-log likelihood,” acknowledging that we fit and evaluate on the same data.

#### 2.4 EM algorithm (via `GaussianMixture`)

The parameters \(\Theta\) are estimated via the **Expectation–Maximization (EM)** algorithm:

1. **E-step** (responsibilities):
   $$
   \gamma_{ik}
     = p(z_i = k \mid \mathbf{x}_i, \Theta^{(t)})
     = \frac{\pi_k^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}
            {\sum_j \pi_j^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}.
   $$

2. **M-step** (parameter update):
   $$
   \pi_k^{(t+1)} = \frac{1}{N}\sum_i \gamma_{ik}, \quad
   \boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}, \quad
   \boldsymbol{\Sigma}_k^{(t+1)} = \frac{\sum_i \gamma_{ik} (\mathbf{x}_i-\boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i-\boldsymbol{\mu}_k^{(t+1)})^\top}{\sum_i \gamma_{ik}}.
   $$

We **do not** implement EM by hand; instead we use `sklearn.mixture.GaussianMixture`, which performs EM internally. Our contribution is how we:
- choose the number of components,
- initialize EM using Amely’s k‑means results,
- run multiple initializations, and
- select the best model via pseudo-log-likelihood.

---

### 3. Relationship to Amely's k-means

#### 3.1 Amely's k-means model

Amely performs k-means clustering in RGB space. For a chosen $K$, k-means finds centers
$$
\{\mathbf{c}_k\}_{k=1}^K
$$
by minimizing:
$$
\sum_{k=1}^{K} \sum_{i \in C_k} \lVert \mathbf{x}_i - \mathbf{c}_k \rVert^2.
$$

Each pixel is assigned to the **nearest** center (hard assignment), which gives a **binary segmentation mask** but **no probabilities**.

Amely explores \(K = 2,3,4,5\), reporting metrics (accuracy, precision, recall, Dice) in `amely_unsupervised_work/results.txt`. Her best balanced performance occurs at \(K=4\), with \(K=3\) also giving good results.

#### 3.2 Using Amely's solutions to seed GMM

Prof’s instruction:
- “Try to pass off all k-means solutions to Leroy so he can run in multiple conditions.”

Our code follows this by:

1. **Running Amely’s `segment_kmeans_rgb`** for the same \(K\) (3 or 4) on each image.
2. Extracting the **RGB centers** \(\{\mathbf{c}_k^{(r)}\}\) from several independent k‑means runs (different random starts).
3. Providing these centers as **initial means** \(\boldsymbol{\mu}_k^{(0)}\) to the GMM in multiple EM runs.

In `fit_gmm_best_of_n`:
- `kmeans_centers_list` stores all center sets we obtained from Amely’s method.
- For the first few GMM runs, `means_init` is set from these centers.
- Subsequent runs use purely random initializations.

This transfers Amely’s **discrete cluster structure** into a **probabilistic mixture** and then lets EM refine it.

---

### 4. Multiple initializations and selection by pseudo-log-likelihood

EM only guarantees convergence to a **local** optimum of the log-likelihood. From office hours:
- “Only guaranteed to converge to local optimum.”
- “How do you choose the local optimum? Compute the probability of each solution, and choose the most probable solution.”

Implementation in our code:

1. For each image and each $K\in\{3,4\}$, we run **$n_\text{runs} = 10$** independent GMM fits (some seeded by k‑means, some random).
2. For each run $r$ we compute the pseudo-log-likelihood $\text{PLL}(\Theta_r)$.
3. We pick the **best run**:
   $$
   r^* = \arg\max_r \text{PLL}(\Theta_r),
   $$
   and use $\Theta_{r^*}$ for the rest of the pipeline.

This directly implements the “choose the most probable solution” idea.

---

### 5. From GMM to probability maps and masks

#### 5.1 Posterior probabilities

For the selected model $\Theta^*$, EM gives responsibilities $\gamma_{ik} = p(z_i = k \mid \mathbf{x}_i,\Theta^*)$. For each pixel:
$$
P(z_i = k \mid \mathbf{x}_i, \Theta^*) = \gamma_{ik}.
$$

In `gmm_to_probability_map`, we call `gmm.predict_proba(pixels)` to get all $\gamma_{ik}$, then extract the **foreground component** (see below), giving a probability map
$$
P_{\text{FG}}(\mathbf{x}_i) = p(z_i = k_{\text{fg}} \mid \mathbf{x}_i, \Theta^*).
$$

This is exactly what Prof meant by *“Don’t have to worry about foreground/background, because you get a probability of foreground/background instead of a classification.”*

#### 5.2 Choosing the saliva component (foreground)

Saliva under UV is **bright**. For each component mean $\boldsymbol{\mu}_k = (R_k, G_k, B_k)$, we compute luminance
$$
Y_k = 0.299 R_k + 0.587 G_k + 0.114 B_k.
$$

We choose
$$
k_{\text{fg}} = \arg\max_k Y_k,
$$
i.e., the **brightest component** in luminance. In code, this is implemented in `identify_foreground_component` using NumPy.

#### 5.3 From probability map to binary mask

We threshold the probability map at $T = 0.5$:
$$
M(i) =
  \begin{cases}
    255, & P_{\text{FG}}(\mathbf{x}_i) > 0.5, \\
    0,   & \text{otherwise}.
  \end{cases}
$$

Then we apply **morphological opening and closing** (`cleanup_mask`) to remove small noisy blobs and fill small holes, giving a clean saliva mask.

---

### 6. Evaluation against ground truth

We compare the final mask $M$ to the binary ground-truth mask $G$. Defining:
- $\text{TP} = |\{i : M(i)=1, G(i)=1\}|$,
- $\text{TN} = |\{i : M(i)=0, G(i)=0\}|$,
- $\text{FP} = |\{i : M(i)=1, G(i)=0\}|$,
- $\text{FN} = |\{i : M(i)=0, G(i)=1\}|$,

we compute:
$$
\text{Accuracy} = \frac{\text{TP}+\text{TN}}{\text{TP}+\text{TN}+\text{FP}+\text{FN}}, \quad
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}},
$$
$$
\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}, \quad
\text{Dice} = \frac{2\,\text{TP}}{2\,\text{TP}+\text{FP}+\text{FN}}.
$$

These are computed in `evaluate_performance` and written to `results.txt` for each image and each \(K\), along with the best pseudo-log-likelihood.

---

### 7. Experimental setup and interpretation

#### 7.1 Setup

- Images: `Saliva_Segmentation_dataset/Raw_images/1.tif` … `10.tif`.
- Ground truth: corresponding masks in `Saliva_Segmentation_dataset/Binary_masks/`.
- Components: $K = 3$ and $K = 4$ (chosen based on Amely’s results).
- For each image and each $K$: 10 GMM runs with different initializations, best selected by pseudo-log-likelihood.

#### 7.2 Connection to Amely’s k-means results

From `amely_unsupervised_work/results.txt`:
- $K=3$: first “good” segmentation (average Dice ≈ 0.544).
- $K=4$: best balanced result (Dice ≈ 0.601, Precision ≈ 0.569, Recall ≈ 0.817).
- $K=5$: slightly higher Dice (≈ 0.627) but significantly lower recall (≈ 0.713), so too conservative.

Our GMM experiments with $K=3$ and $K=4$ are designed to:
1. **Match** these promising k‑means settings.
2. **See if a probabilistic model can improve** segmentation quality at the same \(K\).
3. Provide **pseudo-log-likelihood curves** to discuss model fit in the report.

#### 7.3 Overall meaning for the unsupervised method

The final unsupervised pipeline across both students is:

1. **Amely (k-means)**:
   - Explore different values of $K$.
   - Evaluate segmentation performance (accuracy, precision, recall, Dice).
   - Identify that $K=3$ and $K=4$ are the most meaningful choices.

2. **Leroy (GMM)**:
   - Treat pixel colors as samples from a **probabilistic mixture model**.
   - Use Amely’s k‑means centers as informative initializations.
   - Run EM multiple times and choose the model with highest pseudo-log-likelihood.
   - Produce **probability maps** and final masks.
   - Evaluate again with accuracy/precision/recall/Dice and compare to Amely’s k‑means.

This demonstrates a complete, theoretically grounded **unsupervised segmentation framework**:
- Start from k-means (simple, fast, hard clusters),
- Upgrade to GMM (full covariance, probabilities, model selection),
- Validate empirically using the labeled masks.

---

### 8. How to run

From the project root:

```bash
python Leroy/leroy_gmm.py
```

This will:
- Process all 10 images.
- Run GMM with 3 and 4 components.
- Save figures to `Leroy/leroy_gmm_results/`.
- Save text results to `Leroy/results.txt`.

Use these outputs directly in the report to compare k-means vs. GMM and to explain how the probabilistic mixture model improves (or behaves differently from) the original k‑means segmentation.

