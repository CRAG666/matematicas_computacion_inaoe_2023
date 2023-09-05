"""
Título: Compresión de imágenes
Autor: Diego Crag
Fecha: 29/08/2023
Descripción: Comprimir imágenes usando el método Transformada Discreta de Coseno (DCT) y su inversa
Versión: 1.1

Este programa comprime una imagen dada a un formato mas pequeño de pixels
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

jpeg_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


class ImageCompressor:
    def __init__(self, image_path: str, block_size: int, compression_ratio: int):
        self.__image = mpimg.imread(image_path)

        if len(self.__image.shape) == 3:
            self.__image = np.mean(self.__image, axis=2)
        self.__block_size = block_size
        self.__compression_ratio = compression_ratio

    @staticmethod
    @nb.njit(parallel=True)
    def dct_2d(block: np.ndarray) -> np.ndarray:
        """
        Realiza una Transformada de Coseno Discreta bidimensional (2D DCT) en el bloque de entrada.

        Calcula la Transformada de Coseno Discreta 2D de un bloque de datos de entrada
        utilizando el algoritmo de la DCT tipo-II. La función aplica coeficientes de ponderación
        y suma productos de valores de pixels multiplicados por funciones de coseno en 2D.

        Args:
            block (np.ndarray): Bloque de entrada en forma de matriz NumPy. Debe ser de tamaño NxN.

        Returns:
            np.ndarray: Matriz resultante de la 2D DCT del bloque de entrada.

        Notes:
            Los valores en el bloque de entrada deben estar en el rango [0, 255] para obtener
            resultados precisos. Se utiliza la implementación paralela optimizada de Numba para
            acelerar el cálculo.

        References:
            [2] Numba: http://numba.pydata.org/
        """
        N = block.shape[0]
        result = np.zeros((N, N), dtype=float)

        for u in nb.prange(N):
            for v in nb.prange(N):
                cu = 1.0 if u == 0 else np.sqrt(2) / 2.0
                cv = 1.0 if v == 0 else np.sqrt(2) / 2.0

                sum_val = 0.0
                for x in range(N):
                    for y in range(N):
                        sum_val += (
                            block[x, y]
                            * np.cos((2 * x + 1) * u * np.pi / (2 * N))
                            * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                        )

                result[u, v] = 0.25 * cu * cv * sum_val

        return result

    @staticmethod
    @nb.njit(parallel=True)
    def idct_2d(block: np.ndarray) -> np.ndarray:
        """
        Realiza una Transformada Inversa de Coseno Discreta bidimensional (2D IDCT) en el bloque de entrada.

        Calcula la Transformada Inversa de Coseno Discreta 2D de un bloque de datos de entrada
        utilizando el algoritmo de la IDCT tipo-III. La función aplica coeficientes de ponderación
        y suma productos de valores de coeficientes DCT multiplicados por funciones de coseno en 2D.

        Args:
            block (np.ndarray): Bloque de coeficientes DCT en forma de matriz NumPy. Debe ser de tamaño NxN.

        Returns:
            np.ndarray: Matriz resultante de la 2D IDCT del bloque de entrada.

        Notes:
            Los coeficientes DCT en el bloque deben ser calculados previamente utilizando una DCT.
            Se utiliza la implementación paralela optimizada de Numba para acelerar el cálculo.

        References:
            [2] Numba: http://numba.pydata.org/

        """
        N = block.shape[0]
        result = np.zeros((N, N), dtype=float)

        for x in nb.prange(N):
            for y in nb.prange(N):
                sum_val = 0.0
                for u in range(N):
                    for v in range(N):
                        cu = 1.0 if u == 0 else np.sqrt(2) / 2.0
                        cv = 1.0 if v == 0 else np.sqrt(2) / 2.0
                        sum_val += (
                            cu
                            * cv
                            * block[u, v]
                            * np.cos((2 * x + 1) * u * np.pi / (2 * N))
                            * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                        )

                result[x, y] = 0.25 * sum_val

        return result

    def __compress_image(self) -> np.ndarray:
        """
        Comprime la imagen utilizando la Transformada de Coseno Discreta (DCT).

        Aplica la Transformada de Coseno Discreta 2D y cuenta los bloques de la imagen
        para comprimir la información. Utiliza el tamaño de bloque y la relación de compresión
        definidos en la instancia.

        Returns:
            np.ndarray: Imagen comprimida resultante.
        """
        image = self.__image
        height, width = image.shape
        compressed_image = np.zeros_like(image)

        for i in range(0, height, self.__block_size):
            for j in range(0, width, self.__block_size):
                block = image[i : i + self.__block_size, j : j + self.__block_size]
                dct_block = self.dct_2d(block)
                quantized_block = np.round(dct_block / jpeg_quantization_matrix)
                # np.savetxt("matrix.txt", quantized_block, fmt="%d")
                compressed_image[
                    i : i + self.__block_size, j : j + self.__block_size
                ] = quantized_block

        return compressed_image

    def __decompress_image(self, compressed_image) -> np.ndarray:
        """
        Descomprime la imagen comprimida utilizando la Transformada Inversa de Coseno Discreta (IDCT).

        Aplica la Transformada Inversa de Coseno Discreta 2D a los bloques de la imagen comprimida para
        restaurar la información original. Utiliza el tamaño de bloque y la relación de compresión
        definidos en la instancia.

        Returns:
            np.ndarray: Imagen descomprimida resultante.

        Notes:
            La Transformada Inversa de Coseno Discreta recupera los datos originales.
        """
        height, width = compressed_image.shape
        decompressed_image = np.zeros_like(compressed_image)

        for i in range(0, height, self.__block_size):
            for j in range(0, width, self.__block_size):
                quantized_block = compressed_image[
                    i : i + self.__block_size, j : j + self.__block_size
                ]
                idct_block = self.idct_2d(quantized_block)
                decompressed_image[
                    i : i + self.__block_size, j : j + self.__block_size
                ] = idct_block

        return decompressed_image

    def display_images(self):
        """
        Compara la imagen original con la imagen descomprimida
        """
        compressed_image = self.__compress_image()
        decompressed_image = self.__decompress_image(compressed_image)
        print(compressed_image)
        print(decompressed_image)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        dct_result_log = np.log(np.abs(compressed_image) + 1)
        plt.imshow(dct_result_log, cmap="gray")
        plt.title("Compressed Image")
        plt.subplot(1, 2, 2)
        plt.imshow(decompressed_image, cmap="gray")
        plt.title("Decompressed Image")
        plt.show()


block_size = 8
compression_ratio = 100

compressor = ImageCompressor("image.jpg", block_size, compression_ratio)
compressor.display_images()
