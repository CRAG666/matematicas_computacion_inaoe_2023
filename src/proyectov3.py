import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd

faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces_data.images
n_samples, h, w = images.shape
X = images.reshape((n_samples, -1))

sparse_X = csr_matrix(X)

n_components = 150
U, S, VT = randomized_svd(sparse_X, n_components=n_components, random_state=42)


def reconstruct_image(u, vt):
    return np.dot(u, vt).reshape(h, w)


def compare_faces(input_image, u, vt):
    input_image = input_image.reshape(1, -1)
    u_input = input_image.dot(vt.T)  # Proyecta la imagen de entrada en el espacio de U
    return u_input.reshape(h, w)


def find_similar_faces(input_image, dataset, u, vt):
    mse_values = []  # Almacenar las diferencias medias cuadráticas (MSE)

    for face in dataset:
        reconstructed_face = compare_faces(face, u, vt)
        mse = mean_squared_error(face, reconstructed_face)
        mse_values.append(mse)

    n_similar = 5
    most_similar_indices = np.argsort(mse_values)[:n_similar]

    return [dataset[i] for i in most_similar_indices]


input_image_path = "imagen_de_prueba.jpg"
input_image = imread(input_image_path)
input_image = resize(rgb2gray(input_image), (h, w))

reconstructed_input = compare_faces(input_image, U, VT)

similar_faces = find_similar_faces(input_image, images, U, VT)

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap="gray")
plt.title("Imagen de entrada")

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_input, cmap="gray")
plt.title("Reconstrucción")

plt.subplot(1, 3, 3)
for i, face in enumerate(similar_faces):
    plt.subplot(1, n_similar, i + 1)
    plt.imshow(face, cmap="gray")
    plt.title(f"Similar {i + 1}")

plt.tight_layout()
plt.show()
