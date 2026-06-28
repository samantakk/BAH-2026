import rasterio
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'gap-filling', 'src'))

with rasterio.open('data/bands/clean_reference.tif') as src:
    data = src.read().astype(np.float32)  # shape: (4, H, W)
    profile = src.profile

H, W = data.shape[1], data.shape[2]

# Create a circular cloud mask (like the paper's Fig 3)
mask = np.zeros((H, W), dtype=np.uint8)
cy, cx = H // 2, W // 2
radius = min(H, W) // 4   # covers ~20% of image
Y, X = np.ogrid[:H, :W]
circle = (X - cx)**2 + (Y - cy)**2 <= radius**2
mask[circle] = 1  # 1 = cloudy pixel

# Apply cloud: set cloudy pixels to high reflectance value (white cloud)
cloudy_data = data.copy()
cloud_value = 3000.0  # typical Sentinel-2 reflectance value for bright cloud
for band in range(data.shape[0]):
    cloudy_data[band][mask == 1] = cloud_value

# Save the cloudy version as a new TIF
cloudy_profile = profile.copy()
with rasterio.open('data/bands/cloudy_target.tif', 'w', **cloudy_profile) as dst:
    dst.write(cloudy_data.astype(profile['dtype']))

# Save the mask
mask_profile = profile.copy()
mask_profile.update(count=1, dtype='uint8')
with rasterio.open('data/bands/cloud_mask.tif', 'w', **mask_profile) as dst:
    dst.write(mask[np.newaxis, :, :])

# Visual check
rgb_clean = np.stack([data[2], data[1], data[0]], axis=-1)
rgb_cloudy = np.stack([cloudy_data[2], cloudy_data[1], cloudy_data[0]], axis=-1)

# Normalize for display
def normalize(x):
    return np.clip((x - x.min()) / (x.max() - x.min() + 1e-8), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(normalize(rgb_clean))
axes[0].set_title('Clean Reference (Ground Truth)')
axes[0].axis('off')

axes[1].imshow(normalize(rgb_cloudy))
axes[1].set_title('Synthetic Cloudy Target')
axes[1].axis('off')

axes[2].imshow(mask, cmap='gray')
axes[2].set_title('Cloud Mask (white = cloud)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('synthetic_cloud_preview.png', dpi=150)
plt.show()
print("Saved: cloudy_target.tif, cloud_mask.tif, synthetic_cloud_preview.png")