import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD


class ImageCompressionApp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_image(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def compress_image(self, n_components):
        if hasattr(self, "image"):
            try:
                self.svd = TruncatedSVD(n_components=n_components)
                self.components = self.svd.fit_transform(self.image)
            except ValueError as e:
                raise ValueError("Error during compression: " + str(e))

    def restore_image(self):
        if hasattr(self, "svd") and self.svd is not None:
            try:
                reconstructed_image_data = np.dot(self.components, self.svd.components_)
                reconstructed_image_data = np.clip(reconstructed_image_data, 0, 255)
                reconstructed_image_data = reconstructed_image_data.astype(np.uint8)
                self.decompressed_image = reconstructed_image_data
            except Exception as e:
                raise ValueError("Error during restoration: " + str(e))

    def plot_images(self):
        if hasattr(self, "image") and hasattr(self, "decompressed_image"):
            plt.figure(figsize=(10, 5))

            # Imagen original
            plt.subplot(2, 3, 1)
            plt.imshow(self.image, cmap="gray")
            plt.title("Imagen Original")
            plt.axis("off")

            # Imagen comprimida
            plt.subplot(2, 3, 2)
            plt.imshow(self.components, cmap="gray")
            plt.title("Imagen Comprimida")
            plt.axis("off")

            # Imagen restaurada
            plt.subplot(2, 3, 3)
            plt.imshow(self.decompressed_image, cmap="gray")
            plt.title("Imagen Restaurada")
            plt.axis("off")

            # Histograma de la imagen original
            plt.subplot(2, 3, 4)
            plt.hist(self.image.ravel(), bins=256, range=(0, 256), density=True)
            plt.title("Histograma Original")
            plt.xlabel("Valor de Pixel")
            plt.ylabel("Frecuencia Normalizada")

            # Histograma de la imagen restaurada
            plt.subplot(2, 3, 5)
            plt.hist(
                self.decompressed_image.ravel(), bins=256, range=(0, 256), density=True
            )
            plt.title("Histograma Restaurado")
            plt.xlabel("Valor de Pixel")
            plt.ylabel("Frecuencia Normalizada")

            # Varianza explicada
            plt.subplot(2, 3, 6)
            explained_variance = self.svd.explained_variance_ratio_
            x_values = list(range(1, len(explained_variance) + 1))
            plt.plot(x_values, explained_variance, marker="o", linestyle="-")
            plt.title("Varianza Explicada")
            plt.xlabel("Número de Componentes Principales")
            plt.ylabel("Varianza Explicada")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    app = ImageCompressionApp()
    image_path = "./lenna.jpg"  # Cambia esto por la ruta de tu imagen
    n_components = 50  # Cambia esto al número deseado de componentes principales
    app.load_image(image_path)
    app.compress_image(n_components)
    app.restore_image()
    app.plot_images()
