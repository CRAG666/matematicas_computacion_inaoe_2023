import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


class ImageCompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression with SVD")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        self.frame = ttk.Frame(root)
        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.create_columns()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def create_columns(self):
        self.create_buttons()

        # Configuración uniforme de ancho de columna y margen
        uniform_width = 400
        uniform_height = 256  # Altura uniforme
        uniform_margin = 5

        # Columna 1: Original
        self.column1 = ttk.Frame(self.frame)
        self.column1.grid(row=1, column=0, sticky="nsew")

        self.canvas_original_container = ttk.Frame(self.column1)
        self.canvas_original_container.grid(row=0, column=0)

        self.canvas_original = tk.Canvas(self.canvas_original_container, width=225, height=225)
        self.canvas_original.grid(row=0, column=0, padx=94, pady=0)

        self.canvas_decompressed_container = ttk.Frame(self.column1)
        self.canvas_decompressed_container.grid(row=0, column=1)

        self.canvas_decompressed = tk.Canvas(self.canvas_decompressed_container, width=225, height=225)
        self.canvas_decompressed.grid(row=0, column=0, padx=100, pady=0)

        # Columna 2: components
        self.column2 = ttk.Frame(self.frame)
        self.column2.grid(row=2, column=0, sticky="nsew")

        self.canvas_components_container = ttk.Frame(self.column2)
        self.canvas_components_container.grid(row=0, column=0)

        self.canvas_original_components = tk.Canvas(self.canvas_components_container, width=256, height=256)
        self.canvas_original_components.grid(row=0, column=0, padx=82, pady=0)

        self.canvas_ccomponents_container = ttk.Frame(self.column2)
        self.canvas_ccomponents_container.grid(row=0, column=1)

        self.canvas_compressed = tk.Canvas(self.canvas_ccomponents_container, width=256, height=256)
        self.canvas_compressed.grid(row=0, column=0, padx=77, pady=0)

        # Columna 3: Histogramas
        self.column3 = ttk.Frame(self.frame)
        self.column3.grid(row=3, column=0, sticky="nsew")

        self.histogram_canvas = tk.Canvas(self.column3, width=uniform_width, height=uniform_height)
        self.histogram_canvas.grid(row=0, column=0, padx=uniform_margin, pady=0)

        self.histogram_reconstruction_canvas = tk.Canvas(
            self.column3, width=uniform_width, height=uniform_height
        )
        self.histogram_reconstruction_canvas.grid(row=0, column=1, padx=uniform_margin, pady=0)

        # Columna 4: MSA y PSNR
        self.column4 = ttk.Frame(self.frame)
        self.column4.grid(row=4, column=0, sticky="nsew")

        mse_label = ttk.Label(self.column4, text="MSE:")
        mse_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.mse_value_label = ttk.Label(self.column4, text="0.0")
        self.mse_value_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        psnr_label = ttk.Label(self.column4, text="PSNR:")
        psnr_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")

        self.psnr_value_label = ttk.Label(self.column4, text="0.0")
        self.psnr_value_label.grid(row=0, column=3, padx=10, pady=5, sticky="w")

    def create_buttons(self):
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.grid(
            row=0, column=0, padx=5, pady=10, columnspan=5, sticky="w"
        )

        self.load_button = ttk.Button(
            self.button_frame, text="Cargar Imagen", command=self.load_image
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.n_components_label = ttk.Label(
            self.button_frame, text="Número de Componentes:"
        )
        self.n_components_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.n_components_entry = ttk.Entry(self.button_frame)
        self.n_components_entry.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.svd_button = ttk.Button(
            self.button_frame, text="Comprimir", command=self.on_compress_image
        )
        self.svd_button.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.svd_button.state(["disabled"])

        self.restore_button = ttk.Button(
            self.button_frame, text="Restaurar", command=self.restore_image
        )
        self.restore_button.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.restore_button.state(["disabled"])

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
            self.image_data = np.array(self.image.convert("L"))
            n_components = min(self.image_data.shape)
            compressed_image = self.compress_image(n_components)
            self.photo_org = ImageTk.PhotoImage(compressed_image)
            self.canvas_original_components.create_image(
                0,
                0,
                anchor="nw",
                image=self.photo_org,
            )
            self.plot_histogram(self.image_data)

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
            image=self.photo_compressed,
        )

        explained_variance = self.svd.explained_variance_ratio_
        x_values = list(range(1, n_components + 1))

    def compress_image(self, n_components):
        if hasattr(self, "image"):
            try:
                self.svd = TruncatedSVD(n_components=n_components)
                self.components = self.svd.fit_transform(self.image_data)
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
                reconstructed_image_data = self.svd.inverse_transform(self.components)
                reconstructed_image = np.uint8(reconstructed_image_data)
                reconstructed_image = Image.fromarray(reconstructed_image_data)
                reconstructed_image = reconstructed_image.resize(self.image.size)
                self.photo_decompressed = ImageTk.PhotoImage(reconstructed_image)
                self.canvas_decompressed.create_image(
                    0, 0, anchor="nw", image=self.photo_decompressed
                )

                self.plot_histogram_reconstruction(reconstructed_image_data)

                mse = mean_squared_error(
                    self.image_data.ravel(),
                    reconstructed_image_data.ravel(),
                )
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                self.mse_value_label.config(text=str(mse))
                self.psnr_value_label.config(text=str(psnr))
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.restore_button.state(["disabled"])

    def plot_histogram(self, image_data):
        plt.figure(figsize=(6, 4))
        plt.hist(image_data.ravel(), bins=256, range=(0, 256), density=True)
        plt.title("Histograma de la Imagen Original")
        plt.xlabel("Valor de Pixel")
        plt.ylabel("Frecuencia Normalizada")
        plt.grid()
        plt.tight_layout()

        self.histogram_canvas.delete("all")
        self.plot_to_canvas(self.histogram_canvas, plt)

    def plot_histogram_reconstruction(self, image_data):
        plt.figure(figsize=(6, 4))
        plt.hist(image_data.ravel(), bins=256, range=(0, 256), density=True)
        plt.title("Histograma de la Imagen Reconstruida")
        plt.xlabel("Valor de Pixel")
        plt.ylabel("Frecuencia Normalizada")
        plt.grid()
        plt.tight_layout()

        self.histogram_reconstruction_canvas.delete("all")
        self.plot_to_canvas(self.histogram_reconstruction_canvas, plt)

    def calculate_reconstruction_error(self, n_components, x_values):
        svd = TruncatedSVD(n_components=n_components)
        components = svd.fit_transform(self.image_data)

        reconstruction_error = []

        for i in range(1, n_components + 1):
            reconstructed_image_data = np.dot(components[:, :i], svd.components_[:i, :])
            error = np.linalg.norm(self.image_data - reconstructed_image_data)
            reconstruction_error.append(error)

        return reconstruction_error

    def plot_to_canvas(self, canvas, plt):
        plt.savefig("temp_plot.png", bbox_inches="tight", format="png", dpi=100)
        plt.close()

        image = Image.open("temp_plot.png")
        image = image.resize((canvas.winfo_width(), canvas.winfo_height()))
        photo = ImageTk.PhotoImage(image)

        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressionApp(root)
    root.mainloop()
