import numpy as np
import rasterio
import matplotlib.pyplot as plt
import sys, os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append(os.path.join(os.getcwd(), 'gap-filling', 'src'))
from ssrf_gap_filling import SSRF_GapFilling

def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

clean  = read_tif('data/bands/clean_reference.tif')
cloudy = read_tif('data/bands/cloudy_target.tif')
mask   = read_tif('data/bands/cloud_mask.tif')[0]

# --- Crop to center 150x150 patch (includes the cloud circle) ---
H, W = clean.shape[1], clean.shape[2]
cy, cx = H // 2, W // 2
size = 150
r0, r1 = cy - size//2, cy + size//2
c0, c1 = cx - size//2, cx + size//2

clean  = clean[:, r0:r1, c0:c1]
cloudy = cloudy[:, r0:r1, c0:c1]
mask   = mask[r0:r1, c0:c1]

print(f"Cropped to {clean.shape}, Cloud %: {mask.mean()*100:.1f}%")

band_names = ['B2', 'B3', 'B4', 'B8']
reconstructed = np.zeros_like(clean)
ssim_scores, psnr_scores = [], []

for target_band_idx in range(clean.shape[0]):
    print(f"\nProcessing band {band_names[target_band_idx]} ({target_band_idx+1}/4)...")

    cloudy_band = cloudy[target_band_idx].copy()
    cloudy_band[mask == 1] = np.nan

    aux_bands = {}
    for b_idx, b_name in enumerate(band_names):
        if b_idx != target_band_idx:
            aux_bands[b_name] = clean[b_idx]

    ssrf = SSRF_GapFilling(
        cloudy_image=cloudy_band,
        auxiliary_bands_images=aux_bands
    )

    df_train, df_test = ssrf.generate_df()
    print(f"  Train: {len(df_train)} pixels, Test (cloud): {len(df_test)} pixels")

    if len(df_test) == 0:
        reconstructed[target_band_idx] = cloudy[target_band_idx]
        continue

    filled_band, _ = ssrf.fill_SS_nan(ssrf.train_RF, {}, df_train, df_test)
    reconstructed[target_band_idx] = filled_band

    gt = clean[target_band_idx]
    pred = filled_band
    vmax = max(gt.max(), pred.max()) + 1e-8
    s = ssim(gt/vmax, pred/vmax, data_range=1.0)
    p = psnr(gt/vmax, pred/vmax, data_range=1.0)
    ssim_scores.append(s)
    psnr_scores.append(p)
    print(f"  SSIM: {s:.4f}, PSNR: {p:.2f} dB")

print(f"\n=== Final Results ===")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")

def normalize(x):
    return np.clip((x - x.min()) / (x.max() - x.min() + 1e-8), 0, 1)

rgb_clean  = normalize(np.stack([clean[2], clean[1], clean[0]], axis=-1))
rgb_cloudy = normalize(np.stack([cloudy[2], cloudy[1], cloudy[0]], axis=-1))
rgb_recon  = normalize(np.stack([reconstructed[2], reconstructed[1], reconstructed[0]], axis=-1))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb_clean);  axes[0].set_title('Ground Truth'); axes[0].axis('off')
axes[1].imshow(rgb_cloudy); axes[1].set_title('Cloudy Input'); axes[1].axis('off')
axes[2].imshow(rgb_recon);  axes[2].set_title(
    f'SSRF Result\nSSIM={np.mean(ssim_scores):.3f}'); axes[2].axis('off')

plt.tight_layout()
plt.savefig('ssrf_result.png', dpi=150)
plt.show()
print("Saved: ssrf_result.png")