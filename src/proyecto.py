import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import resize  # Importar la función resize
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


class ImageCompressionApp:
    def __init__(self):
        self.data = fetch_olivetti_faces(shuffle=True, random_state=42)
        self.images = self.data.images
        self.target = self.data.target
        self.n_samples, self.n_features = self.data.data.shape
        self.n_components = 0
        self.svd = None

    def compress_images(self, n_components):
        if n_components <= 0:
            raise ValueError("El número de componentes debe ser mayor que 0.")
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.images_compressed = self.svd.fit_transform(self.data.data)
        self.reconstructed_images = self.svd.inverse_transform(self.images_compressed)

    def _check_compression_state(self):
        if self.n_components <= 0 or self.svd is None:
            raise ValueError("Primero debes comprimir las imágenes.")

    def convert_images_to_jpg_format(self, image):
        self._check_compression_state()
        return (image * 255).astype(np.uint8)

    def calculate_error(self):
        self._check_compression_state()
        return mean_squared_error(self.data.data, self.reconstructed_images)

    def analyze_images(self):
        self._check_compression_state()
        num_images_to_analyze = len(self.data.images)
        metrics = []

        # Redimensionar todas las imágenes reconstruidas al tamaño de la primera imagen original
        reference_shape = self.data.images[0].shape

        for i in range(num_images_to_analyze):
            mse = mean_squared_error(
                self.data.images[i].ravel(),
                self.reconstructed_images[i].ravel(),
            )
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

            metrics.append(
                {
                    "Imagen": i + 1,
                    "MSE": mse,
                    "PSNR": psnr,
                }
            )

        return metrics

    def plot_images_with_metrics(self, num_images_to_show=4):
        if num_images_to_show <= 0:
            raise ValueError("El número de imágenes a mostrar debe ser mayor que 0.")

        # random_indices = random.sample(range(len(self.data.images)), num_images_to_show)
        random_indices = [129, 171, 255, 41]

        plt.figure(figsize=(15, 12))

        for i, idx in enumerate(random_indices):
            plt.subplot(4, num_images_to_show, i + 1)
            plt.imshow(self.data.images[idx], cmap=plt.cm.gray)
            plt.title(f"Imagen {idx + 1}")
            plt.axis("off")

            image_jpg = self.convert_images_to_jpg_format(self.data.data[idx])
            plt.subplot(4, num_images_to_show, i + num_images_to_show + 1)
            plt.hist(
                image_jpg.ravel(),
                bins=256,
                range=(0, 256),
                density=True,
                color="b",
                alpha=0.7,
            )
            plt.title(f"Histograma {idx + 1}")
            plt.xlabel("Valor de Pixel")
            plt.ylabel("Frecuencia Normalizada")

            plt.subplot(4, num_images_to_show, i + 2 * num_images_to_show + 1)
            plt.imshow(
                self.reconstructed_images[idx].reshape(self.data.images[0].shape),
                cmap=plt.cm.gray,
            )
            plt.title(f"Reconstruida {idx + 1}")
            plt.axis("off")

            image_reconstructed_jpg = self.convert_images_to_jpg_format(
                self.reconstructed_images[idx]
            )

            plt.subplot(4, num_images_to_show, i + 3 * num_images_to_show + 1)
            plt.hist(
                image_reconstructed_jpg.ravel(),
                bins=256,
                range=(0, 256),
                density=True,
                color="r",
                alpha=0.7,
            )
            plt.title(f"Histograma Reconstruido {idx + 1}")
            plt.xlabel("Valor de Pixel")
            plt.ylabel("Frecuencia Normalizada")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = ImageCompressionApp()
    num_components = 50
    app.compress_images(num_components)

    metrics_list = app.analyze_images()

    with open("metrics.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Imagen",
            "MSE",
            "PSNR",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(metrics_list)

    app.plot_images_with_metrics(num_images_to_show=4)
