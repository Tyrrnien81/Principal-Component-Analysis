from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!

    # Load the dataset from the npy file
    x = np.load(filename)
    # Calculate the mean of the dataset
    mean_x = np.mean(x, axis=0)
    # Center the dataset
    centered_x = x - mean_x

    return centered_x


def get_covariance(dataset):
    # Your implementation goes here!

    # Number of the data points
    n = dataset.shape[0]

    # Calculate the covariance matrix
    # dataset.T = mxn, dataset = nxm
    covariance_matrix = (1 / (n - 1)) * np.dot(dataset.T, dataset)

    return covariance_matrix


def get_eig(S, m):
    # Your implementation goes here!

    # Get the eigenvalues and eigenvectors of the convarianc matrix
    eigvals, eigvecs = eigh(S, subset_by_index=[S.shape[0] - m, S.shape[0] - 1])

    # Sort the eigenvalues in descending order
    index = np.argsort(eigvals)[::-1]
    # Select the sorted eigenvalues
    eigvals_sorted = eigvals[index]
    # Select the sorted eigenvectors
    eigvecs_sorted = eigvecs[:, index]

    # Convert the eigenvalues into a diagonal matrix
    return np.diag(eigvals_sorted), eigvecs_sorted


def get_eig_prop(S, prop):
    # Your implementation goes here!

    # Get the eigenvalues and eigenvectors of the convarianc matrix
    eigvals, eigvecs = eigh(S)

    # Sort the eigenvalues in descending order
    index = np.argsort(eigvals)[::-1]
    eigvals = eigvals[index]
    eigvecs = eigvecs[:, index]

    # Calculate the cumulative eigenvalues
    total_eigvals = np.sum(eigvals)
    # Calculate the proportion of the variance
    proportion_variance = np.cumsum(eigvals) / total_eigvals

    # Find the smallest k such that the proportion of the 
    # variance is greater than or equal to prop
    k = np.searchsorted(proportion_variance, prop) + 1

    # Compute only the top k eigenvalues and eigenvectors using subset_by_index
    eigvals_subset, eigvecs_subset = eigh(S, subset_by_index=[S.shape[0] - k, S.shape[0] - 1])
    
    # Sort in descending order
    index_subset = np.argsort(eigvals_subset)[::-1]
    eigvals_subset = eigvals_subset[index_subset]
    eigvecs_subset = eigvecs_subset[:, index_subset]

    # Transpose the eigenvalues
    return np.diag(eigvals_subset), eigvecs_subset


def project_image(image, U):
    # Your implementation goes here!

    # Project the image onto the eigenvector U
    projection = np.dot(U.T, image)
    # Reconstruct the projected image
    reconstructed_image = np.dot(U, projection)

    return reconstructed_image


def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2

    # Reshape the images to be 64x64
    orig_image = orig.reshape(64, 64)
    proj_image = proj.reshape(64, 64)

    # Create a subplot using matplotlib
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)

    # Display the original and projected images
    pic1 = ax1.imshow(orig_image, aspect='equal')
    ax1.set_title('Original')
    fig.colorbar(pic1, ax=ax1)

    pic2 = ax2.imshow(proj_image, aspect='equal')
    ax2.set_title('Projection')
    fig.colorbar(pic2, ax=ax2)

    # Return the figure and two sublot objects
    return fig, ax1, ax2


def perturb_image(image, U, sigma):
    # Your implementation goes here!

    # Project the image onto the eigenvector U
    z = np.dot(U.T, image)

    # Add Gaussian noise to the projection
    noise = np.random.normal(0, sigma, z.shape)
    perturbed_projection = z + noise

    # Reconstruct the perturbed image
    perturbed_image = np.dot(U, perturbed_projection)

    return perturbed_image


def combine_image(image1, image2, U, lam):
    # Your implementation goes here!

    # Project the images onto the eigenvector U
    z1 = np.dot(U.T, image1)
    z2 = np.dot(U.T, image2)

    # Combine the two images with a linear combination of lamda ratio
    combined_projection = lam * z1 + (1 - lam) * z2

    # Reconstruct the combined image
    combined_image = np.dot(U, combined_projection)

    return combined_image







# Test the functions
#1. Load the dataset and center the data
print("Test 1")
centered_data = load_and_center_dataset('face_dataset.npy')
print(centered_data.shape)

#2. Calculate the covariance matrix of the centered dataset
print("Test 2")
centered_data = load_and_center_dataset('face_dataset.npy')
S = get_covariance(centered_data)
print(S.shape)

#3. Calculate the eigenvectors and eigenvalues of the covariance matrix
print("Test 3")
S = get_covariance(centered_data)
Lambda, U = get_eig(S, 100)
print(Lambda.shape)
print(U.shape)

#4. Calculate the eigenvalues and eigenvectors that represent 95% or moe of the variance
print("Test 4")
S = get_covariance(centered_data)
Lambda, U = get_eig_prop(S, 0.07)
print(Lambda.shape)
print(U.shape)

#5. Project the image onto the eigenvector U
print("Test 5")
centered_data = load_and_center_dataset('face_dataset.npy')
S = get_covariance(centered_data)
Lambda, U = get_eig(S, 100)
projected_image = project_image(centered_data[0], U)
print(projected_image.shape)

#6. Display the original and projected image
print("Test 6")
centered_data = load_and_center_dataset('face_dataset.npy')
S = get_covariance(centered_data)
Lambda, U = get_eig(S, 100)
projected_image = project_image(centered_data[50], U)
fig, ax1, ax2 = display_image(centered_data[50], projected_image)
plt.show()

#7. Perturb the image
print("Test 7")
centered_data = load_and_center_dataset('face_dataset.npy')
S = get_covariance(centered_data)
Lambda, U = get_eig(S, 100)
perturbed_image = perturb_image(centered_data[50], U, sigma=1000)
fig, ax1, ax2 = display_image(centered_data[50], perturbed_image)
plt.show()

#8. Combine two images
print("Test 8")
centered_data = load_and_center_dataset('face_dataset.npy')
S = get_covariance(centered_data)
Lambda, U = get_eig(S, 100)
combined_image = combine_image(centered_data[50], centered_data[80], U, lam=0.5)
fig, ax1, ax2 = display_image(centered_data[50], combined_image)
plt.show()