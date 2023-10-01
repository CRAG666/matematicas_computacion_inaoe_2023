import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("num_components", type=int, help="Número de componentes")
args = parser.parse_args()
num_components = args.num_components
df = pd.read_csv(f"metrics-{args.num_components}.csv")

imagenes = df["Imagen"]
mse = df["MSE"]
psnr = df["PSNR"]

fig, axs = plt.subplots(2, 1, figsize=(15, 15))

axs[0].scatter(imagenes, mse, marker="o", linestyle="-")
axs[0].set_title("Error Cuadrático Medio (MSE)")
axs[0].set_xlabel("Imagen")
axs[0].set_ylabel("Valor de MSE")

axs[1].scatter(imagenes, psnr, marker="o", linestyle="-")
axs[1].set_title("Relación Señal-Ruido de Pico (PSNR)")
axs[1].set_xlabel("Imagen")
axs[1].set_ylabel("Valor de PSNR")

plt.tight_layout()

plt.show()
