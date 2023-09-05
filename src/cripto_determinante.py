import numpy as np

key = np.array([[5, 8], [7, 9]])
det_key = np.linalg.det(key)

if det_key != 0:
    attached_matrix = np.array([[key[1, 1], -key[0, 1]], [-key[1, 0], key[0, 0]]])

    reverse_key = (1 / det_key) * attached_matrix

    msg = np.array([1, 2])

    encrypted_msg = key @ msg

    deciphered_msg = reverse_key @ encrypted_msg

    print("Mensaje original:", msg)
    print("Mensaje cifrado:", encrypted_msg)
    print("Mensaje descifrado:", deciphered_msg)
else:
    print("La clave no es invertible debido a un determinante igual a cero.")
