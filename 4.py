import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)

def normalize_inputs(inputs):
    min_val = min(inputs)
    max_val = max(inputs)
    return [(val - min_val) / (max_val - min_val) for val in inputs]

def main():
    training_data = {
        '0': [48, 0],
        '1': [49, 1],
        '2': [50, 0],
        '3': [51, 1],
        '4': [52, 0],
        '5': [53, 1],
        '6': [54, 0],
        '7': [55, 1],
        '8': [56, 0],
        '9': [57, 1]
    }

    inputs = np.array([data[0] for data in training_data.values()])
    normalized_inputs = normalize_inputs(inputs)
    labels = np.array([data[1] for data in training_data.values()])

    perceptron = Perceptron(input_size=1, learning_rate=0.1, epochs=10000)
    perceptron.train(normalized_inputs, labels)

    test_inputs = [ord(digit) for digit in ['2', '9', '7']]
    normalized_test_inputs = normalize_inputs(test_inputs)
    for input_val, norm_input in zip(test_inputs, normalized_test_inputs):
        prediction = perceptron.predict(norm_input)
        result = 'even' if prediction == 0 else 'odd'
        print(f"ASCII {input_val}: Predicted as {result}")

if __name__ == "__main__":
    main()
