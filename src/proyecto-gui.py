import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.decomposition import TruncatedSVD


class ImageCompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression with SVD")
        self.root.geometry(
            "800x400"
        )  # Establece un tamaño predeterminado para la ventana

        # Crear el marco principal
        self.frame = ttk.Frame(root)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        # Botones
        self.load_button = ttk.Button(
            self.frame, text="Cargar Imagen", command=self.load_image
        )
        self.load_button.grid(row=0, column=0, pady=10)

        self.svd_button = ttk.Button(
            self.frame, text="Comprimir", command=self.on_compress_image
        )
        self.svd_button.grid(row=0, column=2, pady=10)
        self.svd_button.state(["disabled"])

        self.restore_button = ttk.Button(
            self.frame, text="Restaurar", command=self.restore_image
        )
        self.restore_button.grid(row=0, column=3, pady=10)
        self.restore_button.state(["disabled"])

        # Lienzos para mostrar imágenes
        self.canvas_original = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_original.grid(row=1, column=0, padx=5)

        self.canvas_original_components = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_original_components.grid(row=1, column=1, padx=5)

        self.canvas_compressed = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_compressed.grid(row=1, column=2, padx=5)

        self.canvas_decompressed = tk.Canvas(self.frame, width=256, height=256)
        self.canvas_decompressed.grid(row=1, column=3, padx=5)

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
            self.image_data = np.array(
                self.image.convert("L")
            )  # Convertir a escala de grises
            n_components = min(self.image_data.shape)
            compressed_image = self.compress_image(n_components)
            self.photo_org = ImageTk.PhotoImage(compressed_image)
            self.canvas_original_components.create_image(
                0,
                0,
                anchor="nw",
                image=self.photo_org,  # Mostrar la imagen comprimida aquí
            )

    def on_compress_image(self):
        n_components = int(self.n_components_entry.get())
        if n_components <= 0:
            raise ValueError("El número de componentes debe ser mayor que 0.")
        compressed_image = self.compress_image(n_components)
        self.photo_compressed = ImageTk.PhotoImage(compressed_image)
        self.canvas_compressed.create_image(
            0,
            0,
            anchor="nw",
            image=self.photo_compressed,  # Mostrar la imagen comprimida aquí
        )

    def compress_image(self, n_components):
        if hasattr(self, "image"):
            try:
                # Realizar la descomposición SVD
                self.svd = TruncatedSVD(n_components=n_components)
                self.components = self.svd.fit_transform(self.image_data)

                # Habilitar el botón de descompresión
                self.restore_button.state(["!disabled"])
                return self.get_compressed_components(self.components)
            except ValueError as e:
                messagebox.showerror("Error", str(e))

    def get_compressed_components(self, components):
        num_components = components.shape[1]

        noise_size = (256, 256)

        noise_image = np.random.randint(0, 256, size=noise_size, dtype=np.uint8)
        noise_image = Image.fromarray(noise_image, "L")

        component_images = []

        for i in range(num_components):
            # Escalar los valores de los componentes para que estén en el rango 0-255
            component_data = components[:, i]
            component_min = component_data.min()
            component_max = component_data.max()

            if component_max != component_min:
                scaled_component_data = (
                    (component_data - component_min)
                    / (component_max - component_min)
                    * 255
                ).astype(np.uint8)
            else:
                scaled_component_data = np.zeros_like(component_data, dtype=np.uint8)

            component_image_data = scaled_component_data.reshape(
                (
                    int(np.sqrt(len(scaled_component_data))),
                    int(np.sqrt(len(scaled_component_data))),
                )
            )

            component_image = Image.fromarray(component_image_data, "L")
            component_image = component_image.resize(noise_size)

            component_images.append(component_image)

        for component_image in component_images:
            noise_image = Image.blend(noise_image, component_image, alpha=0.5)

        return noise_image

    def restore_image(self):
        if hasattr(self, "svd") and self.svd is not None:
            try:
                # Reconstruye la matriz original a partir de los componentes y los valores singulares
                reconstructed_image_data = np.dot(self.components, self.svd.components_)

                # Asegúrate de que los valores estén en el rango adecuado (0-255)
                reconstructed_image_data = np.clip(reconstructed_image_data, 0, 255)

                # Redimensiona la matriz a la forma original de la imagen
                reconstructed_image_data = reconstructed_image_data.reshape(
                    self.image.size[1], self.image.size[0]
                )

                # Convierte la matriz reconstruida en una imagen
                reconstructed_image_data = np.uint8(reconstructed_image_data)
                decompressed_image = Image.fromarray(reconstructed_image_data)

                # Reescala la imagen reconstruida para que tenga el mismo tamaño que la original
                decompressed_image = decompressed_image.resize(self.image.size)

                self.photo_decompressed = ImageTk.PhotoImage(decompressed_image)
                self.canvas_decompressed.create_image(
                    0, 0, anchor="nw", image=self.photo_decompressed
                )
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.restore_button.state(["disabled"])


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressionApp(root)
    root.mainloop()
