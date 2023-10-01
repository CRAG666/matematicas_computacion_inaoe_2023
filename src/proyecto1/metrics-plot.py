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

fig, ax = plt.subplots(figsize=(15, 7))

ax.scatter(mse, psnr, marker="o", linestyle="-")
ax.set_title("Relación entre MSE y PSNR")
ax.set_xlabel("Valor de MSE")
ax.set_ylabel("Valor de PSNR")

plt.tight_layout()

plt.show()
