# # 8 Sտեղծել երկու երկչափ զանգված և հաշվել դրանց ա) տարր առ տարր արտադրյալը, բ) մատրիցային արտադրյալը։
# import numpy as np

# array1 = np.array([[1, 2], [3, 4]])
# array2 = np.array([[5, 6], [7, 8]])

# element_wise_product = array1 * array2

# matrix_product = np.dot(array1, array2)

# print("Array 1:\n", array1)
# print("Array 2:\n", array2)
# print("\nElement-wise product:\n", element_wise_product)
# print("\nMatrix product:\n", matrix_product)


# 10 Ստեղծել (3, 3) մատրից, որի տարրերը կլինեն 11-19,
#  սակայն բացահայտ կերպով սահմանել տարրերի տիպը որպես float։
import numpy as np

matrix = np.arange(11, 20, dtype=float).reshape(3, 3)

print(matrix)

