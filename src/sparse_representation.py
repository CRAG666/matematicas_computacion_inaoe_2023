import numpy as np

# Crear una matriz dispersa como un NumPy array
dense_matrix = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 3]])

# Imprimir la matriz densa (original)
print("Matriz densa:")
print(dense_matrix)

# Convertir la matriz densa en una matriz dispersa
sparse_matrix = np.where(dense_matrix != 0)

# Imprimir la matriz dispersa (valores no cero y sus índices)
print("Matriz dispersa:")
print(sparse_matrix)

# Acceder a un valor en la matriz densa
row = 1
col = 1
value = dense_matrix[row, col]
print(f"Valor en ({row}, {col}): {value}")

# Convertir la matriz dispersa de nuevo a densa (opcional)
dense_matrix_reconstructed = np.zeros_like(dense_matrix)
for row, col in zip(sparse_matrix[0], sparse_matrix[1]):
    dense_matrix_reconstructed[row, col] = dense_matrix[row, col]

print("Matriz densa reconstruida:")
print(dense_matrix_reconstructed)
