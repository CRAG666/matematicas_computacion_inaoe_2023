import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.extmath import randomized_svd


class OlivettiFaceExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Explorador de Caras Olivetti")

        # Variables
        self.num_components = tk.IntVar()
        self.num_components.set(150)
        self.selected_image1 = tk.IntVar()
        self.selected_image2 = tk.IntVar()

        # Cargar el conjunto de datos de caras Olivetti
        self.faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
        self.images = self.faces_data.images
        self.n_samples, self.h, self.w = self.images.shape
        self.X = self.images.reshape((self.n_samples, -1))

        # Convertir la matriz de imágenes en una matriz dispersa
        self.sparse_X = csr_matrix(self.X)

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
        components_entry = ttk.Entry(main_frame, textvariable=self.num_components)
        components_entry.grid(column=1, row=0, padx=10, pady=5)

        # Botón para actualizar la visualización
        update_button = ttk.Button(
            main_frame,
            text="Actualizar Visualización",
            command=self.update_visualization,
        )
        update_button.grid(column=2, row=0, padx=10, pady=5)

        # Selector de imágenes
        image1_label = ttk.Label(main_frame, text="Imagen 1:")
        image1_label.grid(column=0, row=1, padx=10, pady=5)
        image1_combobox = ttk.Combobox(
            main_frame,
            values=list(range(self.n_samples)),
            textvariable=self.selected_image1,
        )
        image1_combobox.grid(column=1, row=1, padx=10, pady=5)

        image2_label = ttk.Label(main_frame, text="Imagen 2:")
        image2_label.grid(column=0, row=2, padx=10, pady=5)
        image2_combobox = ttk.Combobox(
            main_frame,
            values=list(range(self.n_samples)),
            textvariable=self.selected_image2,
        )
        image2_combobox.grid(column=1, row=2, padx=10, pady=5)

        # Figuras para mostrar las imágenes
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 6))

        # Lienzo para mostrar las figuras
        canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        canvas.get_tk_widget().grid(column=0, row=3, columnspan=3, padx=10, pady=10)

        # Actualizar la visualización inicial
        self.update_visualization()

    def update_visualization(self):
        n_components = self.num_components.get()

        # Realizar la descomposición SVD en la matriz dispersa
        U, S, VT = randomized_svd(
            self.sparse_X, n_components=n_components, random_state=42
        )

        # Reconstruir las imágenes originales a partir de VT
        reconstructed_faces = np.dot(U, np.dot(np.diag(S), VT))
        reconstructed_faces = reconstructed_faces.reshape(
            (self.n_samples, self.h, self.w)
        )

        # Obtener las imágenes seleccionadas
        image1_index = self.selected_image1.get()
        image2_index = self.selected_image2.get()

        # Actualizar las figuras
        self.ax[0, 0].imshow(self.images[image1_index], cmap="gray")
        self.ax[0, 0].set_title("Original 1")

        self.ax[0, 1].imshow(self.images[image2_index], cmap="gray")
        self.ax[0, 1].set_title("Original 2")

        self.ax[1, 0].imshow(reconstructed_faces[image1_index], cmap="gray")
        self.ax[1, 0].set_title("Reconstrucción 1")

        self.ax[1, 1].imshow(reconstructed_faces[image2_index], cmap="gray")
        self.ax[1, 1].set_title("Reconstrucción 2")

        self.fig.tight_layout()
        self.fig.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OlivettiFaceExplorer(root)
    root.mainloop()
