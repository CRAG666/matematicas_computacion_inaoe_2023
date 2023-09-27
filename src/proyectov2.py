import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.extmath import randomized_svd

faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces_data.images
n_samples, h, w = images.shape
X = images.reshape((n_samples, -1))

sparse_X = csr_matrix(X)

U, S, VT = randomized_svd(sparse_X, n_components=150, random_state=42)


def reconstruct_image(u, vt):
    return np.dot(u, vt).reshape(h, w)


def compare_faces(input_image):
    input_image = input_image.reshape(1, -1)
    u_input = input_image.dot(VT.T)  # Proyecta la imagen de entrada en el espacio de U
    reconstructed_input = reconstruct_image(u_input, VT)
    return reconstructed_input


input_image = X[0]

reconstructed_input = compare_faces(input_image)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(input_image.reshape(h, w), cmap="gray")
plt.title("Imagen de entrada")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_input, cmap="gray")
plt.title("Reconstrucción")

plt.tight_layout()
plt.show()

reconstructed_input = reconstructed_input.reshape(-1)

difference = np.mean(np.abs(input_image - reconstructed_input))
print(f"Diferencia promedio: {difference:.2f}")
