# BAH 2026 — PS_02: Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery
## Project Documentation

**Team Role:** Somil Goyal — ML/CV  
**Hackathon:** Bhartiya Antariksha Hackathon 2026  
**Problem Statement:** PS_02 — Cloud removal and reconstruction for LISS-IV satellite imagery  
**Date:** June 2026

---

## 1. Problem Statement Summary

Persistent cloud cover over the North Eastern Region (NER) of India severely limits the usability of LISS-IV optical satellite imagery for applications like land use mapping, disaster monitoring, and environmental assessment. The goal is to build a **Generative AI-based framework** that:
- Automatically detects and removes clouds from LISS-IV imagery
- Reconstructs the hidden surface using spatial, spectral, and temporal information
- Produces analysis-ready, cloud-free satellite products
- Evaluates outputs quantitatively (SSIM, PSNR, rRMSE) and qualitatively

---

## 2. Research Paper Implemented

**Title:** *Evaluating Sentinel-2 gap filling techniques for cloud removal and data reconstruction*  
**Authors:** Said Grich, Jamal Elfarkh, Nadia Ouaadi, Boucha Ait Hssaine, Hamid Halim, Abdelghani Chehbouni  
**Published:** Scientific Reports, 2026. DOI: 10.1038/s41598-026-39488-2  
**Code:** https://github.com/said-grich/gap-filling

### Why this paper?
The paper provides a comprehensive comparative evaluation of gap-filling methods across 4 categories:

| Category | Methods | Best SSIM |
|---|---|---|
| Spatial | Kriging, IDW, Nearest-Neighbor, Bilinear, Bicubic | 0.72 |
| Temporal | Same methods along time axis | 0.94 |
| Spatio-Temporal | CLR, DTWS, Deep Learning (U-Net) | 0.98 |
| Spatio-Spectral | **SSRF (Spatial-Spectral Random Forest)** | **0.9856** |

**Key finding:** SSRF achieved the highest SSIM (0.9856) among all methods, including Deep Learning — making it the best choice for a 1-week hackathon prototype.

---

## 3. Strategy & Architecture Decisions

### Why not train a GAN or Diffusion model from scratch?
- Adversarial training is unstable and takes days to converge reliably
- The PS mentions these as options, not requirements
- SSRF matches or beats DL on SSIM with zero training time
- GAN/Diffusion mentioned as "future work" in pitch — satisfies PS without risking the prototype

### Why synthetic cloud simulation?
- Real paired (cloudy, same-scene-clean) LISS-IV data is not freely available
- Bhoonidhi approval takes time
- Synthetic clouds on clean images give **perfect ground truth** for quantitative evaluation
- This is the exact method used in the paper (Fig. 3) and is scientifically standard

### Why Sentinel-2 instead of LISS-IV?
- LISS-IV data requires Bhoonidhi registration (pending)
- Sentinel-2 is free, instant, covers NER India, and is one of the PS's own suggested auxiliary datasets
- SSRF is sensor-agnostic — swapping Sentinel-2 for LISS-IV is just a file-reading change
- Demonstrates the method works; LISS-IV fine-tuning is mentioned as next step

---

## 4. Environment Setup

### System
- Windows 11, HP OMEN 16
- GPU: NVIDIA RTX 4060 (CUDA confirmed working)
- Conda environment: `cloudremoval` (Python 3.10)

### Environment creation
```bash
conda create -n cloudremoval python=3.10
conda activate cloudremoval
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy matplotlib -c conda-forge
pip install opencv-python rasterio earthengine-api geemap scikit-learn scikit-image
conda install jupyter -c conda-forge
```

### GPU verification
```python
import torch
print(torch.cuda.is_available())  # → True
```

### Repository cloned
```bash
git clone https://github.com/said-grich/gap-filling
```

---

## 5. Project Folder Structure

```
D:\cloud-removal\
│
├── gap-filling\                    # Cloned research paper repo
│   ├── src\
│   │   ├── ssrf_gap_filling.py     # SSRF method (our primary method)
│   │   ├── clustring_lr_gapfilling.py  # CLR method (comparison)
│   │   ├── STpconvUnet.py          # U-Net with partial convolutions (DL method)
│   │   ├── STpconvLayer.py         # Partial convolution layer implementation
│   │   ├── Generate_Clouds_Masks.py # Paper's cloud mask generation
│   │   ├── DataGenerator.py        # Data loading utilities
│   │   ├── Losses.py               # Custom loss functions for DL
│   │   ├── dtw_seg.py              # DTWS method
│   │   ├── kriging.py              # Kriging method
│   │   ├── idw.py                  # IDW method
│   │   └── spatial.py              # Spatial interpolation methods
│   ├── gap_filling_landsat.ipynb   # Original paper notebook (Landsat)
│   └── gap_filling_modis.ipynb     # Original paper notebook (MODIS)
│
├── data\
│   └── bands\
│       ├── clean_reference.tif     # Cloud-free Sentinel-2 (ground truth)
│       ├── cloudy_target.tif       # Synthetic cloud applied to clean image
│       ├── cloud_mask.tif          # Binary mask (1=cloud, 0=clear)
│       ├── real_cloudy.tif         # Real cloudy Sentinel-2 (visual demo)
│       └── target_image.tif        # Second clean image (had no-data region)
│
├── generate_data.py                # Toy synthetic image + cloud blob generator
├── model.py                        # TinyUNet architecture (proof of concept)
├── test_model.py                   # Architecture shape verification
├── train.py                        # Toy training loop (single image pair)
├── download_sentinel2.py           # GEE download script
├── add_synthetic_clouds.py         # Applies circular cloud mask to clean TIF
├── check_tif.py                    # Verify GeoTIFF file is valid
├── view_tif.py                     # Visualize GeoTIFF as RGB
├── crop_and_run.py                 # Main SSRF runner on cropped patch
└── ssrf_result.png                 # Output visualization (after SSRF runs)
```

---

## 6. File-by-File Explanation

### `generate_data.py`
**Purpose:** Learning exercise — creates a fake "satellite scene" using OpenCV shapes (rectangles, circles), then paints a soft Gaussian blob on it to simulate a cloud. Produces `clean.png`, `cloudy.png`, `mask.png`.

**Why created:** To prove the core concept (take a clean image → add fake cloud → train model to remove it) before touching any real satellite data.

---

### `model.py`
**Purpose:** Defines a minimal U-Net architecture (`TinyUNet`) in PyTorch with 3 encoder levels, a bottleneck, and 3 decoder levels with skip connections. Input: 3-channel 128×128 image. Output: 3-channel 128×128 reconstructed image.

**Why created:** Hands-on understanding of the encoder-decoder architecture before using the paper's more advanced version. Also serves as the "basic DL baseline" comparison in the pitch.

**Key concept demonstrated:** Skip connections let fine spatial detail bypass the bottleneck, preserving edges and textures in the output.

---

### `test_model.py`
**Purpose:** Sanity check — passes a dummy tensor through TinyUNet and verifies output shape is `(1, 3, 128, 128)`.

**Why created:** Verify architecture is wired correctly before any training.

---

### `train.py`
**Purpose:** Loads `clean.png` and `cloudy.png`, converts to PyTorch tensors, trains TinyUNet for 300 epochs using L1 loss and Adam optimizer, saves `reconstructed.png`.

**Why created:** End-to-end proof that the training pipeline works. Demonstrates the model can memorize one image pair. Establishes the training loop pattern used in all subsequent scripts.

**Key result:** `reconstructed.png` showed the blob removed and shapes restored — pipeline confirmed working.

---

### `download_sentinel2.py`
**Purpose:** Uses Google Earth Engine Python API to download two Sentinel-2 SR scenes over Assam, NER India (bbox: 91.5°E–91.7°E, 26.0°N–26.2°N):
1. `clean_reference.tif` — least cloudy scene (< 20% cloud cover)
2. `real_cloudy.tif` — visually cloudy scene (> 40% cloud cover) for demo

**Why created:** LISS-IV data from Bhoonidhi is pending approval. Sentinel-2 is free, instant, and covers the same NER region specified in the PS. The PS itself lists Sentinel-2 as a valid auxiliary data source.

**GEE Project:** `firstproject-499806`  
**Scale:** 30m resolution (reduced from 10m to stay within GEE's 50MB download limit)  
**Bands:** B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)

---

### `add_synthetic_clouds.py`
**Purpose:** Takes `clean_reference.tif` and programmatically paints a circular cloud blob over the center (~25% of image area). Saves:
- `cloudy_target.tif` — same scene with cloud applied
- `cloud_mask.tif` — binary mask

**Why created:** To generate a controlled (cloudy, clean) pair where we know the exact ground truth, enabling quantitative SSIM/PSNR evaluation — exactly the methodology used in the research paper.

**Cloud simulation method:** Circular mask centered at image midpoint, radius = min(H,W)/4. Cloudy pixels set to 3000 reflectance units (typical Sentinel-2 bright cloud value).

---

### `check_tif.py`
**Purpose:** Quick verification that downloaded GeoTIFFs are valid — prints band count, image dimensions, CRS, data shape, and min/max values.

**Why created:** Windows Photos cannot open GeoTIFFs (they're not regular images — they contain geographic metadata). This script confirmed the downloads were valid satellite data, not corrupted files.

---

### `view_tif.py`
**Purpose:** Renders a GeoTIFF as an RGB image using matplotlib and saves as a viewable PNG.

**Band mapping:** B4→Red, B3→Green, B2→Blue (standard Sentinel-2 natural color composite)

---

### `crop_and_run.py` *(main prototype script)*
**Purpose:** The core implementation — runs SSRF gap filling on a cropped patch of the satellite image.

**Pipeline:**
1. Load `clean_reference.tif` (ground truth), `cloudy_target.tif` (input), `cloud_mask.tif`
2. Crop to 350×350 center patch (captures cloud + surrounding clear pixels for training)
3. For each band (B2, B3, B4, B8):
   - Set cloud pixels to NaN in the target band
   - Use the other 3 bands from clean reference as auxiliary features
   - Run `SSRF_GapFilling` → builds spatial-spectral 3×3 patch features → trains Random Forest → predicts cloud pixels
4. Compute SSIM and PSNR against ground truth
5. Save side-by-side visualization: Ground Truth | Cloudy Input | SSRF Result

---

## 7. How SSRF Works (Plain English)

SSRF (Spatial-Spectral Random Forest) is based on one insight: **different spectral bands of the same scene are correlated**. If you can see what Band 3 (Green) looks like in a region, you can predict what Band 4 (Red) should look like there — because vegetation, water, and soil all have characteristic spectral signatures across bands.

**Step by step:**
1. For each pixel in the image, extract a 3×3 spatial patch from each auxiliary band → 9 values × 3 bands = 27 features per pixel
2. For clear pixels: these 27 features + the target band value = one training sample
3. For cloudy pixels: these 27 features = one test sample (target band value is unknown)
4. Train a Random Forest on clear pixels → predict cloudy pixels
5. The result is a reconstructed band where cloud pixels have been filled in

**Why it works well:** Random Forest captures non-linear spectral relationships and is robust to noise. Using 3×3 spatial patches captures local texture context, not just point values.

---

## 8. Data Pipeline Summary

```
Google Earth Engine (free)
        ↓
Sentinel-2 SR, Assam NER India, 2023
        ↓
clean_reference.tif (4 bands, 747×676 px, 30m)
        ↓
add_synthetic_clouds.py
        ↓
cloudy_target.tif + cloud_mask.tif
        ↓
crop_and_run.py (350×350 center crop)
        ↓
SSRF_GapFilling (from research paper src/)
        ↓
Reconstructed image + SSIM/PSNR metrics
        ↓
ssrf_result.png (3-panel comparison)
```

---

## 9. Evaluation Metrics

| Metric | What it measures | Target |
|---|---|---|
| **SSIM** | Structural similarity (0–1, higher=better) | > 0.85 |
| **PSNR** | Peak signal-to-noise ratio in dB (higher=better) | > 25 dB |
| **rRMSE** | Relative root mean square error (%, lower=better) | < 10% |

Paper's SSRF benchmark on Sentinel-2: SSIM = 0.9856, rRMSE = 7.38–9.18%

---

## 10. Next Steps (Remaining Work)

- [ ] Finish running `crop_and_run.py` and record SSIM/PSNR results
- [ ] Run on `real_cloudy.tif` for visual demo (qualitative)
- [ ] Upgrade U-Net with partial convolutions from `STpconvUnet.py` (DL comparison)
- [ ] Apply to LISS-IV data once Bhoonidhi approval arrives
- [ ] Build Gradio/Streamlit demo app for interactive pitch demo
- [ ] Prepare pitch deck with: Problem → Method → Architecture diagram → Results → Future work

---

## 11. Key References

1. Said Grich et al. (2026). *Evaluating Sentinel-2 gap filling techniques for cloud removal and data reconstruction.* Scientific Reports. https://doi.org/10.1038/s41598-026-39488-2
2. Paper GitHub: https://github.com/said-grich/gap-filling
3. Sentinel-2 data via Google Earth Engine: `COPERNICUS/S2_SR_HARMONIZED`
4. BAH 2026 PS_02 problem statement: Bhartiya Antariksha Hackathon 2026
