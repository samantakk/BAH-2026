import numpy as np
import rasterio
import matplotlib.pyplot as plt
import sys, os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append(os.path.join(os.getcwd(), 'gap-filling', 'src'))
from ssrf_gap_filling import SSRF_GapFilling

# --- Load data ---
def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

clean  = read_tif('data/bands/clean_reference.tif')  # (4, H, W) ground truth
cloudy = read_tif('data/bands/cloudy_target.tif')     # (4, H, W) with synthetic cloud
mask   = read_tif('data/bands/cloud_mask.tif')[0]     # (H, W), 1=cloud 0=clear

print(f"Data loaded. Shape: {clean.shape}, Cloud %: {mask.mean()*100:.1f}%")

# --- Prepare data for SSRF ---
# SSRF expects:
# cloudy_image: 2D array (H, W) of ONE band, with NaN where cloudy
# auxiliary_bands_images: dict of 2D arrays from the KNOWN (clean) image

reconstructed = np.zeros_like(clean)  # will store results band by band
band_names = ['B2', 'B3', 'B4', 'B8']
ssim_scores = []
psnr_scores = []

for target_band_idx in range(clean.shape[0]):
    print(f"\nProcessing band {band_names[target_band_idx]} ({target_band_idx+1}/4)...")

    # Step 1: make the cloudy version of this band with NaN in cloud region
    cloudy_band = cloudy[target_band_idx].copy()
    cloudy_band[mask == 1] = np.nan  # set cloud pixels to NaN

    # Step 2: auxiliary bands = all bands from the CLEAN known image
    # (exclude the target band itself — use the other 3 bands as features)
    aux_bands = {}
    for b_idx, b_name in enumerate(band_names):
        if b_idx != target_band_idx:
            aux_bands[b_name] = clean[b_idx]  # clean reference bands as features

    # Step 3: run SSRF
    ssrf = SSRF_GapFilling(
        cloudy_image=cloudy_band,
        auxiliary_bands_images=aux_bands
    )

    df_train, df_test = ssrf.generate_df()
    print(f"  Train samples: {len(df_train)}, Test (cloud) pixels: {len(df_test)}")

    if len(df_test) == 0:
        print("  No cloud pixels found, skipping.")
        reconstructed[target_band_idx] = cloudy[target_band_idx]
        continue

    filled_band, _ = ssrf.fill_SS_nan(ssrf.train_RF, {}, df_train, df_test)
    reconstructed[target_band_idx] = filled_band

    # Step 4: compute metrics only on cloud pixels
    gt   = clean[target_band_idx]
    pred = filled_band
    vmax = max(gt.max(), pred.max()) + 1e-8
    s = ssim(gt / vmax, pred / vmax, data_range=1.0)
    p = psnr(gt / vmax, pred / vmax, data_range=1.0)
    ssim_scores.append(s)
    psnr_scores.append(p)
    print(f"  SSIM: {s:.4f}, PSNR: {p:.2f} dB")

print(f"\n=== Final Results ===")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")

# --- Visual comparison ---
def normalize(x):
    return np.clip((x - x.min()) / (x.max() - x.min() + 1e-8), 0, 1)

rgb_clean  = normalize(np.stack([clean[2],         clean[1],         clean[0]],         axis=-1))
rgb_cloudy = normalize(np.stack([cloudy[2],        cloudy[1],        cloudy[0]],        axis=-1))
rgb_recon  = normalize(np.stack([reconstructed[2], reconstructed[1], reconstructed[0]], axis=-1))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(rgb_clean);  axes[0].set_title('Ground Truth (Clean)');  axes[0].axis('off')
axes[1].imshow(rgb_cloudy); axes[1].set_title('Synthetic Cloudy Input'); axes[1].axis('off')
axes[2].imshow(rgb_recon);  axes[2].set_title(
    f'SSRF Reconstructed\nSSIM={np.mean(ssim_scores):.3f} PSNR={np.mean(psnr_scores):.1f}dB')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('ssrf_result.png', dpi=150)
plt.show()
print("Saved: ssrf_result.png")