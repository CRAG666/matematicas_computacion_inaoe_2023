import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numba as nb
import numpy as np


@nb.njit(parallel=True)
def dct_2d(image):
    N = image.shape[0]
    result = np.zeros((N, N), dtype=float)

    for u in nb.prange(N):
        for v in nb.prange(N):
            cu = 1.0 if u == 0 else np.sqrt(2) / 2.0
            cv = 1.0 if v == 0 else np.sqrt(2) / 2.0

            sum_val = 0.0
            for x in nb.prange(N):
                for y in nb.prange(N):
                    cos_factor_x = np.cos((2 * x + 1) * u * np.pi / (2 * N))
                    cos_factor_y = np.cos((2 * y + 1) * v * np.pi / (2 * N))
                    sum_val += image[x, y] * cos_factor_x * cos_factor_y
            result[u, v] = 0.25 * cu * cv * sum_val
    return result


image_path = "image.jpg"
image = mpimg.imread(image_path)

if len(image.shape) == 3:
    image = np.mean(image, axis=2)

dct_result = dct_2d(image)
print(image)
print(dct_result)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)

# Mejorando la visualización
dct_result_log = np.log(np.abs(dct_result) + 1)
plt.imshow(dct_result_log, cmap="gray")
# plt.imshow(dct_result)
plt.title("DCT Processing (Log Transform)")
plt.tight_layout()
plt.show()
