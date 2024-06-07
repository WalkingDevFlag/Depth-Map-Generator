# Depth Estimation using MiDaS

This project performs depth estimation on an input image using the MiDaS (Monocular Depth Estimation) model from Intel's Intelligent Systems Lab. The MiDaS model provides state-of-the-art performance for depth estimation tasks, and this implementation allows you to use either "DPT_Large", "DPT_Hybrid", or "MiDaS_small" models based on your accuracy and performance needs.

## Features

-Load and preprocess images.
-Perform depth estimation using the MiDaS model.
-Display the input image and its corresponding depth map.
-Save the depth map with a modified file name.

## Prerequisites

Conda should be installed on your machine.

## Creating a Conda Environment

To create a new Conda environment for this project, follow these steps:

Create a new environment:
```sh
conda create --name depth_estimation python=3.8
```
Activate the environment:
```
conda activate depth_estimation
```
Install necessary dependencies:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
pip install timm
```

## Project Structure
```
depth_estimation/
│
├── input/
│   └── your_image.jpg     # Place your input images here
│
├── output/
│   └──                    # The output depth maps will be saved here
│
├── depth_estimation.ipynb # Jupyter notebook for running the depth estimation
│
└── README.md              # Project readme file
```

## Usage
1) Clone the repository:

```
git clone https://github.com/yourusername/depth_estimation.git
cd depth_estimation
```
2) Place your input images in the input/ directory.

3) Run the Jupyter Notebook:
```
jupyter notebook depth_estimation.ipynb
```
4) Follow the steps in the Jupyter Notebook to load the image, perform depth estimation, display the results, and save the depth map.

## Notes

Ensure that the input/ directory contains the image you want to process.
The output depth map will be saved in the output/ directory with the filename format your_image_Depth_Map.jpg.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Intel Intelligent Systems Lab for the MiDaS model.
PyTorch for providing a robust deep learning framework.
OpenCV and Matplotlib for image processing and visualization.
