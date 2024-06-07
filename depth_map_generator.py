import os
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt

# Define the paths
input_image_path = "input/your_image.jpg"  # Replace 'your_image.jpg' with the actual image name
output_folder = "output"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Select the model type
model_type = "DPT_Hybrid"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load the MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Select the appropriate transform based on the model type
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Load and preprocess the image
img = cv2.imread(input_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Convert the prediction to a numpy array
output = prediction.cpu().numpy()

# Display the input image and the depth map in grayscale
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Depth Map")
plt.imshow(output, cmap='gray')
plt.axis('off')

plt.show()

# Generate the output file name based on the input file name
input_file_name = os.path.basename(input_image_path)
output_file_name = os.path.splitext(input_file_name)[0] + "_Depth_Map.jpg"
output_image_path = os.path.join(output_folder, output_file_name)

# Save the depth map
cv2.imwrite(output_image_path, (output * 255).astype("uint8"))

print(f"Depth map saved as {output_image_path}")
