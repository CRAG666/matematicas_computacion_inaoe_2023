import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.decomposition import TruncatedSVD


class ImageCompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression with SVD")

        # Crear el marco principal
        self.frame = ttk.Frame(root)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        # Botones
        self.load_button = ttk.Button(
            self.frame, text="Cargar Imagen", command=self.load_image
        )
        self.load_button.grid(row=0, column=0, pady=10)

        self.svd_button = ttk.Button(
            self.frame, text="Comprimir", command=self.compress_image
        )
        self.svd_button.grid(row=0, column=1, pady=10)
        self.svd_button.state(["disabled"])

        self.decompress_button = ttk.Button(
            self.frame, text="Descomprimir", command=self.decompress_image
        )
        self.decompress_button.grid(row=0, column=2, pady=10)
        self.decompress_button.state(["disabled"])

        # Lienzos para mostrar imágenes
        self.canvas_original = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_original.grid(row=1, column=0, padx=5)

        self.canvas_compressed = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_compressed.grid(row=1, column=1, padx=5)

        self.canvas_decompressed = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_decompressed.grid(row=1, column=2, padx=5)

        # Etiqueta y entrada para el número de componentes
        self.n_components_label = ttk.Label(self.frame, text="Número de Componentes:")
        self.n_components_label.grid(row=2, column=0, pady=5)

        self.n_components_entry = ttk.Entry(self.frame)
        self.n_components_entry.grid(row=2, column=1, pady=5)
        self.n_components_entry.insert(0, "10")

        # Cargar el clasificador de detección de caras de OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.bmp")]
        )
        if file_path:
            self.image = Image.open(file_path)
            self.photo_original = ImageTk.PhotoImage(self.image)
            self.canvas_original.create_image(
                0, 0, anchor="nw", image=self.photo_original
            )
            self.svd_button.state(["!disabled"])

    def compress_image(self):
        if hasattr(self, "image"):
            try:
                n_components = int(self.n_components_entry.get())
                if n_components <= 0:
                    raise ValueError("El número de componentes debe ser mayor que 0.")

                image_data = np.array(
                    self.image.convert("L")
                )  # Convertir a escala de grises

                # Realizar la descomposición SVD
                svd = TruncatedSVD(n_components=n_components)
                self.components = svd.fit_transform(image_data)

                # Reconstruir la imagen comprimida
                reconstructed_image = np.dot(self.components, svd.components_)
                reconstructed_image = np.uint8(reconstructed_image)
                reconstructed_image = Image.fromarray(reconstructed_image)

                # Mostrar la imagen comprimida
                self.photo_compressed = ImageTk.PhotoImage(reconstructed_image)
                self.canvas_compressed.create_image(
                    0, 0, anchor="nw", image=self.photo_compressed
                )

                # Habilitar el botón de descompresión
                self.decompress_button.state(["!disabled"])
            except ValueError as e:
                tk.messagebox.showerror("Error", str(e))

    def decompress_image(self):
        if hasattr(self, "components"):
            n_components = self.components.shape[1]
            svd = TruncatedSVD(n_components=n_components)

            # Utiliza el mismo número de componentes que se utilizó para la compresión
            n_components_decompression = n_components

            # Reduce la matriz de componentes a los componentes necesarios
            reduced_components = self.components[:, :n_components_decompression]

            # Reconstruye la imagen comprimida
            decompressed_image_data = np.dot(reduced_components, svd.components_)
            decompressed_image_data = np.uint8(decompressed_image_data)
            decompressed_image = Image.fromarray(decompressed_image_data)

            self.photo_decompressed = ImageTk.PhotoImage(decompressed_image)
            self.canvas_decompressed.create_image(
                0, 0, anchor="nw", image=self.photo_decompressed
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressionApp(root)
    root.mainloop()
