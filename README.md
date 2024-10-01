# Principal-Component-Analysis

Introduction to Artificial Intelligence - HW3

This project implements Principal Component Analysis (PCA) on a dataset of facial images using Python, with the aim of performing facial analysis, image perturbation, and hybrid image creation. The project utilizes key Python libraries including NumPy, SciPy, and Matplotlib.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Function Descriptions](#function-descriptions)
6. [Results and Visualization](#results-and-visualization)
7. [Perturbation and Hybrid Image](#perturbation-and-hybrid-image)
8. [License](#license)

## Project Overview

The purpose of this project is to explore PCA and its applications in facial image analysis. PCA is performed on a dataset of 13,233 images, where each image is represented by a 64x64 grayscale matrix, resulting in a 4096-dimensional vector for each image.

The main goals include:

-   Loading and centering the dataset.
-   Calculating the covariance matrix.
-   Performing eigendecomposition.
-   Reconstructing images using principal components.
-   Visualizing original and projected images.
-   Applying Gaussian noise to perturb images.
-   Combining two images using a convex combination of PCA projections.

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy scipy matplotlib
```

You will also need Python 3.x and a version of SciPy >= 1.5.0.

## Dataset

The project uses the **Label Faces in the Wild (LFW)** dataset. The dataset is saved in `face_dataset.npy`, where each image is a 4096-dimensional vector (64x64). The dataset includes 13,233 images.

Download the dataset file `face_dataset.npy` from the provided link in the course materials.

## Usage

1. Clone this repository and navigate to the project folder:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Ensure the dataset (`face_dataset.npy`) is in the same directory as the Python script.

3. Run the main Python script:

```bash
python hw3.py
```

This will execute the test cases and display results in the terminal and via graphical plots.

## Function Descriptions

### 1. `load_and_center_dataset(filename)`

Loads the dataset from the `.npy` file, centers it by subtracting the mean of each feature, and returns the centered dataset as a NumPy array.

-   **Input**: `filename` (string) – Path to the `.npy` dataset file.
-   **Output**: A centered NumPy array of shape `(n, m)` where `n` is the number of images and `m` is 4096.

### 2. `get_covariance(dataset)`

Calculates and returns the covariance matrix of the given dataset.

-   **Input**: `dataset` (NumPy array) – Centered dataset.
-   **Output**: Covariance matrix of shape `(m, m)` where `m` is the number of features (4096).

### 3. `get_eig(S, m)`

Performs eigendecomposition on the covariance matrix `S`, returning the top `m` eigenvalues and their corresponding eigenvectors.

-   **Input**: `S` (NumPy array) – Covariance matrix, `m` (integer) – Number of top eigenvalues to retain.
-   **Output**: Tuple containing a diagonal matrix of the top `m` eigenvalues and the matrix of corresponding eigenvectors.

### 4. `get_eig_prop(S, prop)`

Performs eigendecomposition and returns eigenvalues and eigenvectors that explain more than a given proportion of the variance.

-   **Input**: `S` (NumPy array) – Covariance matrix, `prop` (float) – Desired proportion of variance to explain.
-   **Output**: Tuple containing a diagonal matrix of eigenvalues and the corresponding eigenvectors.

### 5. `project_image(image, U)`

Projects an image onto the PCA subspace defined by eigenvectors `U`.

-   **Input**: `image` (NumPy array) – The image vector, `U` (NumPy array) – Matrix of eigenvectors.
-   **Output**: Projected image in the PCA subspace.

### 6. `display_image(orig, proj)`

Displays the original and projected images side-by-side.

-   **Input**: `orig` (NumPy array) – Original image, `proj` (NumPy array) – Projected image.
-   **Output**: Matplotlib figure.

### 7. `perturb_image(image, U, sigma)`

Perturbs the projection of an image by adding Gaussian noise, then reconstructs the image from the perturbed projection.

-   **Input**: `image` (NumPy array), `U` (NumPy array), `sigma` (float) – Standard deviation of Gaussian noise.
-   **Output**: Reconstructed perturbed image.

### 8. `combine_image(image1, image2, U, lam)`

Combines two images by linearly interpolating their projections, then reconstructs the hybrid image.

-   **Input**: `image1`, `image2` (NumPy arrays), `U` (NumPy array), `lam` (float) – Interpolation weight.
-   **Output**: Reconstructed hybrid image.

## Results and Visualization

This project includes a function `display_image()` that visualizes the original and projected images. For example:

```python
centered_data = load_and_center_dataset('face_dataset.npy')
S = get_covariance(centered_data)
Lambda, U = get_eig(S, 100)
projected_image = project_image(centered_data[50], U)
fig, ax1, ax2 = display_image(centered_data[50], projected_image)
plt.show()
```

The above code will plot the original and projected image for the 50th image in the dataset.

## Perturbation and Hybrid Image

1. **Perturbed Image**: Using `perturb_image()`, the image can be altered with Gaussian noise to explore its effects on reconstruction.

2. **Hybrid Image**: Using `combine_image()`, two images can be blended with a weighted average, creating a hybrid image.
