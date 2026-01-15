# Evaluation Metrics for Defect Image Generation

**Project:** Siemens Energy - Bottle Defect Detection using Stable Diffusion  
**Task 7:** Automated Quantitative Evaluation Metrics  
**Date:** January 2026

---

## Overview

This document outlines quantitative metrics for evaluating the quality of synthetically generated defect images from the LoRA-tuned Stable Diffusion model. The evaluation framework combines general image quality metrics with domain-specific defect detection criteria.

---

## 1. General Image Quality Metrics

### 1.1 Fréchet Inception Distance (FID)

**Purpose:** Measures the distance between real and generated image distributions in feature space.

**Definition:**

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

Where:
- $\mu_r, \mu_g$ = mean of real and generated image features
- $\Sigma_r, \Sigma_g$ = covariance matrices of real and generated features
- Features extracted from InceptionV3 pool3 layer (2048-dim)

**Interpretation:**
- **FID < 30:** Excellent quality (comparable to real images)
- **FID 30-50:** Good quality (suitable for few-shot scenarios)
- **FID 50-100:** Acceptable for data augmentation
- **FID > 100:** Poor quality, significant distribution mismatch

**Note:** FID requires $\ge50$ samples per class for stable estimates. Since our dataset has only $\approx20$ samples per class, we'll experience higher FID variance. To mitigate this, generate 100 synthetic samples per class (keeping one side stable) and consider computing FID on all 83 samples combined rather than per-class for better stability. 

### 1.2 Inception Score (IS)

**Purpose:** Evaluates image quality and diversity using a pretrained classifier.

**Definition:**

$$\text{IS} = \exp(\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))])$$

Where:
- $p(y|x)$ = conditional class distribution from InceptionV3
- $p(y)$ = marginal class distribution
- $D_{KL}$ = Kullback-Leibler divergence

**Interpretation:**
- **Higher IS = Better:** More confident predictions + diverse samples
- **IS > 3.0:** Good for multi-object datasets (faces, animals, vehicles)
- **IS 1.5-2.5:** Expected for single-object datasets (all bottles)
- **IS < 1.5:** Possible mode collapse or extremely low quality

**Limitations:**
- InceptionV3 trained on ImageNet (not defect-specific)
- **For single-object defect datasets (all bottles): IS will be naturally low (1.5-2.5) and is not meaningful**
- All images classified as "bottle" -> low diversity in ImageNet space -> low IS (expected, not a problem)
- IS measures object-level diversity, not defect-level diversity

---

## 2. Defect-Specific Metrics

### 2.1 Defect Detection Accuracy (DDA)

**Purpose:** Evaluate if generated defects are recognizable by a trained classifier.

**Method:**
1. Train a defect classifier on **real** bottle images (e.g., ResNet-50)
2. Generate 100 synthetic samples per defect class
3. Run classifier on synthetic images
4. Measure per-class accuracy

**Definition:**

$$\text{DDA}_c = \frac{\text{Correct predictions for class } c}{\text{Total synthetic samples of class } c}$$

**Target Benchmarks:**
- **broken_large:** DDA ≥ 90% (most distinctive defect)
- **broken_small:** DDA ≥ 80% (smaller features, harder)
- **contamination:** DDA ≥ 65% (most ambiguous)
- **good:** DDA ≥ 75% (should be clearly non-defective)

**Why This Matters:**
- If synthetic defects fool a real-trained classifier → they're realistic
- Low DDA indicates poor defect representation
- Class-specific DDA reveals which defects need improvement

### 2.2 Structural Similarity Index (SSIM)

**Purpose:** Measure structural preservation between real and synthetic defect patterns.

**Definition:**

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

**Application:**
- Compute SSIM between generated and nearest real sample per class
- Average SSIM across defect types
- **Target:** 0.5-0.7 (similar structure, not identical)
- **SSIM > 0.8:** Possible memorization/overfitting
- **SSIM < 0.4:** Poor defect structure preservation

### 2.3 Defect Localization Consistency (DLC)

**Purpose:** Verify defects appear in plausible bottle regions.

**Method:**
1. Apply Grad-CAM or attention maps to identify defect-salient regions
2. Compare with known defect distributions from real data
3. Measure spatial overlap with expected defect zones

**Definition:**

$$\text{DLC} = \frac{\text{Overlap}(\text{Generated defect region}, \text{Expected region})}{\text{Union}(\text{Generated defect region}, \text{Expected region})}$$

**Target:** DLC ≥ 0.6 (defects in realistic bottle locations)

---

## 3. Diversity and Coverage Metrics

### 3.1 Intra-Class Diversity (ICD)

**Purpose:** Ensure synthetic samples aren't mode-collapsed.

**Method:**
1. Extract deep features (e.g., ResNet penultimate layer) for 100 generated samples per class
2. Compute pairwise cosine distances
3. Average distance = diversity score

**Definition:**

$$\text{ICD}_c = \frac{1}{N(N-1)} \sum_{i \neq j} (1 - \cos(\mathbf{f}_i, \mathbf{f}_j))$$

**Interpretation:**
- **ICD > 0.3:** Good diversity (samples are distinct)
- **ICD < 0.1:** Mode collapse (all samples look similar)

### 3.2 Feature Space Coverage (FSC)

**Purpose:** Measure how well synthetic data covers real data distribution.

**Method:**
1. Embed real and synthetic images into feature space (e.g., InceptionV3 or ResNet)
2. Compute k-nearest-neighbor (k=5) distances from real to synthetic
3. Lower distance = better coverage

**Definition:**

$$\text{FSC} = \frac{1}{N_{\text{real}}} \sum_{i=1}^{N_{\text{real}}} \min_j ||\mathbf{f}_i^{\text{real}} - \mathbf{f}_j^{\text{syn}}||_2$$

**Target:** FSC < 0.5 (synthetic samples close to real distribution)

---

## 4. Human Evaluation Protocol

### 4.1 Visual Turing Test

**Procedure:**
1. Select 20 real + 20 synthetic images per defect class
2. Randomize order (blind test)
3. Ask 5+ domain experts to classify as real/synthetic
4. Calculate deception rate

**Success Criteria:**
- **Deception rate ≥ 40%:** Synthetic images are highly realistic
- **Deception rate < 20%:** Clear artifacts, needs improvement

### 4.2 Defect Authenticity Rating

**Procedure:**
1. Show 50 synthetic images to experts
2. Rate defect authenticity on 1-5 scale:
   - **5:** Indistinguishable from real defect
   - **4:** Realistic with minor artifacts
   - **3:** Recognizable but noticeable issues
   - **2:** Poor defect representation
   - **1:** Unrealistic/unrecognizable

**Target:** Average rating ≥ 3.5 per defect class

---

## 5. Augmentation Impact Metrics

### 5.1 Augmentation Enhancement Ratio (AER)

**Purpose:** Quantify improvement from augmentation (rotation, photometric).

**Method:**
1. Train classifier with **no augmentation** (v2: 83 samples)
2. Train classifier with **rotation augmentation** (v3: ~200-300 effective views)
3. Measure test accuracy improvement

**Definition:**

$$\text{AER} = \frac{\text{Accuracy}_{\text{augmented}} - \text{Accuracy}_{\text{baseline}}}{\text{Accuracy}_{\text{baseline}}} \times 100\%$$

**Current Observation:** Your v3 (rotation) significantly outperforms v2 (no rotation)

**Target:** AER ≥ 10% (augmentation provides meaningful benefit)

---

## 6. Implementation Roadmap

### Phase 1: Core Metrics (Week 1)
- [ ] Implement FID calculation using `pytorch-fid` or `torch-fidelity`
- [ ] Generate 100 samples per class at epoch 250
- [ ] Compute per-class FID vs real validation set

### Phase 2: Defect-Specific Metrics (Week 2)
- [ ] Train ResNet-50 defect classifier on real data
- [ ] Compute Defect Detection Accuracy (DDA)
- [ ] Calculate SSIM between generated and nearest real samples
- [ ] Implement Intra-Class Diversity (ICD)

### Phase 3: Advanced Evaluation (Week 3)
- [ ] Feature Space Coverage (FSC) analysis
- [ ] Grad-CAM for Defect Localization Consistency
- [ ] Compare v2 vs v3 augmentation impact

### Phase 4: Human Evaluation (Week 4)
- [ ] Prepare blind test with real + synthetic images
- [ ] Recruit 5+ domain experts
- [ ] Analyze deception rate and authenticity ratings

---

## 7. Code Snippets

### 7.1 FID Calculation

```python
from pytorch_fid import fid_score

# Generate 100 samples per class
real_path = "Defect_Spectrum/DS-MVTec/bottle/image/broken_large"
fake_path = "outputs/bottle_diffusion/synthetic_broken_large"

fid_value = fid_score.calculate_fid_given_paths(
    [real_path, fake_path],
    batch_size=50,
    device='cuda',
    dims=2048  # InceptionV3 pool3 layer
)

print(f"FID for broken_large: {fid_value:.2f}")
```

### 7.2 Defect Detection Accuracy

```python
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Load trained defect classifier
classifier = models.resnet50(pretrained=False)
classifier.fc = torch.nn.Linear(2048, 4)  # 4 defect classes
classifier.load_state_dict(torch.load("classifier_real.pth"))
classifier.eval()

# Evaluate on synthetic data
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

synthetic_loader = DataLoader(
    synthetic_dataset,
    batch_size=32,
    shuffle=False
)

correct = {i: 0 for i in range(4)}
total = {i: 0 for i in range(4)}

with torch.no_grad():
    for images, labels in synthetic_loader:
        outputs = classifier(images.cuda())
        preds = outputs.argmax(dim=1)
        
        for label, pred in zip(labels, preds):
            total[label.item()] += 1
            if label == pred:
                correct[label.item()] += 1

# Compute DDA per class
for class_id in range(4):
    dda = correct[class_id] / total[class_id] * 100
    print(f"Class {class_id} DDA: {dda:.2f}%")
```

### 7.3 Intra-Class Diversity

```python
import torch
import torch.nn.functional as F
from torchvision import models
from itertools import combinations

# Extract features using ResNet
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
resnet.eval()

features = []
for img in generated_images:  # 100 samples per class
    with torch.no_grad():
        feat = resnet(img.unsqueeze(0).cuda())
        feat = feat.squeeze().cpu()
        features.append(feat)

features = torch.stack(features)

# Compute pairwise cosine distances
diversity = 0
count = 0
for i, j in combinations(range(len(features)), 2):
    cos_sim = F.cosine_similarity(features[i:i+1], features[j:j+1])
    diversity += (1 - cos_sim.item())
    count += 1

icd = diversity / count
print(f"Intra-Class Diversity: {icd:.4f}")
```

### 7.4 SSIM Comparison

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def compute_ssim_with_real(generated_images, real_images):
    """
    Compute SSIM between each generated image and its nearest real sample.
    """
    ssim_scores = []
    
    for gen_img in generated_images:
        gen_arr = np.array(gen_img)
        
        # Find nearest real image (by pixel distance)
        min_ssim = 0
        for real_img in real_images:
            real_arr = np.array(real_img)
            score = ssim(gen_arr, real_arr, multichannel=True, channel_axis=2)
            min_ssim = max(min_ssim, score)
        
        ssim_scores.append(min_ssim)
    
    avg_ssim = np.mean(ssim_scores)
    return avg_ssim

# Usage
avg_ssim = compute_ssim_with_real(synthetic_broken_large, real_broken_large)
print(f"Average SSIM: {avg_ssim:.4f}")

if avg_ssim > 0.8:
    print("⚠ Warning: High SSIM suggests possible memorization")
elif avg_ssim < 0.4:
    print("⚠ Warning: Low SSIM suggests poor structure preservation")
else:
    print("✓ SSIM in acceptable range")
```

---

## 8. Recommended Evaluation Pipeline

### Step 1: Automatic Metrics (No Human Required)
1. **Generate samples:** 100 per class at epoch 250
2. **Compute FID:** Per-class and overall
3. **Compute IS:** Overall synthetic dataset
4. **Calculate ICD:** Ensure diversity within classes

### Step 2: Defect-Specific Evaluation
1. **Train classifier:** ResNet-50 on real data only
2. **Compute DDA:** How well classifier recognizes synthetic defects
3. **Calculate SSIM:** Structure preservation check
4. **Analyze FSC:** Distribution coverage

### Step 3: Human Validation (Final Check)
1. **Visual Turing Test:** 40 images per class (20 real, 20 synthetic)
2. **Authenticity Rating:** 50 synthetic images rated by experts
3. **Defect Localization:** Verify defects in plausible regions

### Step 4: Comparative Analysis
1. **v2 vs v3:** Quantify augmentation impact (AER)
2. **Epoch progression:** FID at epochs 100, 175, 250
3. **Class-specific:** Identify which defects need improvement

---

## 9. Success Criteria Summary

| Metric | Target | Priority |
|--------|--------|----------|
| **FID (per class)** | < 50 | HIGH |
| **Defect Detection Accuracy** | ≥ 75% avg | HIGH |
| **Intra-Class Diversity** | > 0.3 | MEDIUM |
| **SSIM** | 0.5-0.7 | MEDIUM |
| **Deception Rate** | ≥ 40% | HIGH |
| **Authenticity Rating** | ≥ 3.5/5 | MEDIUM |
| **Feature Space Coverage** | < 0.5 | LOW |

---

## 10. Known Challenges & Solutions

### Challenge 1: Small Dataset (83 samples)
**Problem:** FID unstable with few samples  
**Solution:** Compute per-class FID with bootstrapping (resample 50 times)

### Challenge 2: Contamination Class Ambiguity
**Problem:** Contamination overlaps with good bottles  
**Solution:** Use CLIP embeddings for semantic similarity + human evaluation

### Challenge 3: InceptionV3 Not Defect-Aware
**Problem:** IS may not reflect defect quality  
**Solution:** Train custom defect classifier (ResNet-50) for DDA metric

### Challenge 4: Augmentation vs Memorization
**Problem:** High SSIM could indicate overfitting  
**Solution:** Use ICD + FSC to verify diversity + coverage

---

## 11. Next Steps

1. **Immediate:** Implement FID calculation for v3 (epoch 250)
2. **Week 1:** Train defect classifier on real data for DDA
3. **Week 2:** Generate 100 samples per class + compute all metrics
4. **Week 3:** Compare v2 vs v3 quantitatively (validate augmentation benefit)
5. **Week 4:** Prepare human evaluation study

---

## References

1. Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS.
2. Salimans et al. (2016). "Improved Techniques for Training GANs." NeurIPS.
3. Wang et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity." IEEE TIP.
4. Bergmann et al. (2019). "MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR.

---

**Document Status:** Draft v1.0  
**Last Updated:** January 14, 2026  
**Contact:** Research Team - Siemens Energy Project
