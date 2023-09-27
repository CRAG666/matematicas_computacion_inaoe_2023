import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.extmath import randomized_svd

# Cargar el conjunto de datos de caras Olivetti
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces_data.images
n_samples, h, w = images.shape
X = images.reshape((n_samples, -1))

# Convertir la matriz de imágenes en una matriz dispersa
sparse_X = csr_matrix(X)

# Realizar la descomposición SVD en la matriz dispersa
U, S, VT = randomized_svd(sparse_X, n_components=150, random_state=42)

# Reconstruir las imágenes originales a partir de VT
reconstructed_faces = np.dot(U, np.dot(np.diag(S), VT))
reconstructed_faces = reconstructed_faces.reshape((n_samples, h, w))

# Visualizar algunas de las imágenes originales y reconstruidas
n_images_to_display = 4
plt.figure(figsize=(15, 6))
for i in range(n_images_to_display):
    plt.subplot(4, n_images_to_display, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title("Original")

    plt.subplot(4, n_images_to_display, i + n_images_to_display + 1)
    plt.imshow(sparse_X[i].toarray().reshape(h, w), cmap="binary")
    plt.title("Imagen dispersa")

    plt.subplot(4, n_images_to_display, i + 2 * n_images_to_display + 1)
    u_image = np.dot(U[i], VT)  # Reconstruir la imagen de U usando U[i] y VT
    u_image = u_image.reshape(h, w)
    plt.imshow(u_image, cmap="gray")
    plt.title("Imagen de U")

    plt.subplot(4, n_images_to_display, i + 3 * n_images_to_display + 1)
    plt.imshow(reconstructed_faces[i], cmap="gray")
    plt.title("Reconstruido")

plt.tight_layout()
plt.show()
