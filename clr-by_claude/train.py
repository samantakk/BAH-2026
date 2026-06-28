import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import TinyUNet

# --- Load images ---
clean = cv2.imread("clean.png")
cloudy = cv2.imread("cloudy.png")

# OpenCV loads as BGR; convert to RGB for consistency (not strictly required, but good habit)
clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
cloudy = cv2.cvtColor(cloudy, cv2.COLOR_BGR2RGB)

# --- Convert to PyTorch tensors ---
# Normalize pixel values from [0,255] to [0,1] — neural nets train much better on small numbers
def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))      # HWC -> CHW (PyTorch wants channels first)
    img = np.expand_dims(img, axis=0)        # add batch dimension: CHW -> 1CHW
    return torch.from_numpy(img)

clean_t = to_tensor(clean)
cloudy_t = to_tensor(cloudy)

# --- Set up device (GPU if available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

clean_t = clean_t.to(device)
cloudy_t = cloudy_t.to(device)

model = TinyUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()  # measures average pixel-wise difference between prediction and target

# --- Training loop ---
epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()              # clear gradients from last step
    prediction = model(cloudy_t)       # feed cloudy image in
    loss = loss_fn(prediction, clean_t)  # compare to clean target
    loss.backward()                    # compute how to adjust weights
    optimizer.step()                   # apply the adjustment

    if epoch % 30 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")

# --- Save the result so you can look at it ---
with torch.no_grad():
    output = model(cloudy_t)[0].cpu().numpy()       # remove batch dim, move to CPU
    output = np.transpose(output, (1, 2, 0))         # CHW -> HWC
    output = (output * 255).clip(0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # back to BGR for saving with OpenCV
    cv2.imwrite("reconstructed.png", output)

print("Saved reconstructed.png — compare it to clean.png!")