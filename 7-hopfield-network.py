import numpy as np

v1 = np.array([1, -1, 1, -1])
v2 = np.array([-1, 1, -1, 1])
v3 = np.array([1, 1, -1, -1])
v4 = np.array([-1, -1, 1, 1])

W = np.zeros((len(v1), len(v1)))
for v in [v1, v2, v3, v4]:
    W += np.outer(v, v)

np.fill_diagonal(W, 0)

def hopfield_network(input_vector):
    global W
    output_vector = np.copy(input_vector)
    iterations = 0
    max_iterations = 1000
    while iterations < max_iterations:
        new_output_vector = np.sign(np.dot(W, output_vector))
        if np.array_equal(new_output_vector, output_vector):
            return new_output_vector
        output_vector = new_output_vector
        iterations += 1
    print("Max iterations reached. Unable to converge.")
    return None

test_vector = np.array([1, -1, 1, -1])
result = hopfield_network(test_vector)

if result is not None:
    print("Input Vector:", test_vector)
    print("Reconstructed Vector:", result)
