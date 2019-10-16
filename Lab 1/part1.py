import numpy as np

matrix = np.loadtxt("matrix.csv", delimiter=',')
vector = np.load('vector.npy')

# Creating a matrix of zeros of the same dimensions of the product of the matrix and vector
output = np.zeros([matrix.shape[0], vector.shape[1]])

# Looping through to dot product the values
for index, row in enumerate(matrix):
    sum = 0
    for i, elem in enumerate(vector):
        sum += row[i] * elem
    output[index] = sum

np.savetxt("output_forloop.csv", output, delimiter=" ")

# Using NumPy's dot product function
output_2 = np.dot(matrix, vector)
np.save('output_dot.npy', output_2)

# Creating the difference vector by subtracting the values of the same ndex
output_difference = np.zeros(output.shape)
for i, element in enumerate(output):
    output_difference[i] = output[i]-output_2[i]
np.savetxt("output_difference", output_difference, delimiter=" ")