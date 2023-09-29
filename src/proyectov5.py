import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import Cascade
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Caras con PCA y SVM")

        # Cargar el conjunto de datos de caras Olivetti
        self.faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
        self.images = self.faces_data.images
        self.X = self.images.reshape((self.faces_data.target.shape[0], -1))
        self.y = self.faces_data.target

        # Crear la interfaz gráfica
        self.create_ui()

    def create_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.grid(column=0, row=0, padx=10, pady=10)

        # Selector de componentes principales
        components_label = ttk.Label(
            main_frame, text="Número de Componentes Principales:"
        )
        components_label.grid(column=0, row=0, padx=10, pady=5)
        self.components_entry = ttk.Entry(main_frame)
        self.components_entry.grid(column=1, row=0, padx=10, pady=5)
        self.components_entry.insert(0, "100")

        # Botón para entrenar y evaluar el modelo
        detect_button = ttk.Button(
            main_frame, text="Detectar Caras", command=self.detect_faces
        )
        detect_button.grid(column=2, row=0, padx=10, pady=5)

        # Área de visualización de imágenes
        self.image_canvas = tk.Canvas(main_frame, width=400, height=400)
        self.image_canvas.grid(column=0, row=1, columnspan=3, padx=10, pady=10)

    def detect_faces(self):
        n_components = int(self.components_entry.get())

        # Realizar PCA para reducción de dimensionalidad
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X)

        # Entrenar un clasificador SVM
        svm_classifier = SVC(kernel="linear")
        svm_classifier.fit(X_pca, self.y)

        # Cargar un clasificador de detección de caras basado en Haar Cascades
        face_cascade = Cascade("haarcascade_frontalface_default.xml")

        # Seleccionar una imagen aleatoria para la detección de caras
        random_image_index = np.random.randint(0, len(self.images))
        test_image = self.images[random_image_index]
        test_image_flat = self.X[random_image_index]

        # Realizar la detección de caras
        faces = face_cascade.detect_multi_scale(test_image)

        # Mostrar la imagen con marcos alrededor de las caras detectadas
        plt.imshow(test_image, cmap="gray")
        plt.axis("off")

        for x, y, w, h in faces:
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
            )

        # Actualizar la imagen en la interfaz gráfica
        self.update_image(plt)

    def update_image(self, plt):
        plt.gcf().canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
