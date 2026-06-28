import numpy as np
import cv2

def make_clean_image(size=128):
    """Creates a fake 'satellite scene' using simple shapes and colors."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (40, 120, 40)  # greenish background, like vegetation

    # a few random colored rectangles, like fields/buildings
    cv2.rectangle(img, (10, 10), (60, 50), (180, 140, 90), -1)
    cv2.rectangle(img, (70, 60), (120, 110), (90, 90, 200), -1)
    cv2.circle(img, (40, 90), 20, (200, 200, 60), -1)

    return img

def add_cloud(img):
    """Paints a soft gray blob ('cloud') onto a copy of the image."""
    cloudy = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # draw a blurry white-ish blob to simulate a cloud
    cv2.circle(mask, (64, 64), 35, 255, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)

    # blend: where mask is strong, push pixel color toward light gray (cloud color)
    cloud_color = np.array([220, 220, 220], dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    cloudy = (cloudy.astype(np.float32) * (1 - alpha) + cloud_color * alpha).astype(np.uint8)

    return cloudy, mask

if __name__ == "__main__":
    clean = make_clean_image()
    cloudy, mask = add_cloud(clean)

    cv2.imwrite("clean.png", clean)
    cv2.imwrite("cloudy.png", cloudy)
    cv2.imwrite("mask.png", mask)
    print("Saved clean.png, cloudy.png, mask.png — open them and look!")