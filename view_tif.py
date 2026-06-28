import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open('data/bands/real_cloudy.tif') as src:
    data = src.read()

# data shape is (4, H, W) — bands are B2(blue), B3(green), B4(red), B8(NIR)
# For RGB display we need B4(red)=band 3, B3(green)=band 2, B2(blue)=band 1
rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype(np.float32)

# Normalize to 0-1 for display (stretch the values)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title('Sentinel-2 cloudy image - Assam, NER India')
plt.axis('off')
plt.tight_layout()
plt.savefig('real_cloudy_preview.png', dpi=150)
plt.show()
print("Saved as real_cloudy_preview.png")