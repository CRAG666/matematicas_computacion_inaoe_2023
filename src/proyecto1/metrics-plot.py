import argparse

import pandas as pd
import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("num_components", type=int, help="Número de componentes")
args = parser.parse_args()
num_components = args.num_components
df = pd.read_csv(f"metrics-{args.num_components}.csv")

imagenes = df["Imagen"]
mse = df["MSE"]
psnr = df["PSNR"]

# Crear un DataFrame con los datos y las etiquetas
data = pd.DataFrame({"MSE": mse, "PSNR": psnr, "Imagen": imagenes})

# Crear el gráfico interactivo con etiquetas al hacer hover
fig = px.scatter(
    data, x="MSE", y="PSNR", text="Imagen", title="Relación entre MSE y PSNR"
)
fig.update_traces(
    textposition="top center",
    marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
)
fig.update_xaxes(title="Valor de MSE")
fig.update_yaxes(title="Valor de PSNR")

# Mostrar el gráfico
fig.show()
