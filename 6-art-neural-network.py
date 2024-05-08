import numpy as np

class ARTNetwork:
    def __init__(self, input_size, rho=0.5, beta=1):
        self.input_size = input_size
        self.rho = rho 
        self.beta = beta 
        self.W = np.random.rand(input_size)
        self.T = 1 

    def predict(self, input_vector):
        similarity = np.dot(self.W, input_vector)
        return similarity > self.T

    def learn(self, input_vector):
        while True:
            similarity = np.dot(self.W, input_vector)
            if similarity > self.rho * np.sum(self.W):
                self.W = (1 - self.beta) * self.W + self.beta * input_vector
                break  
            else:
                self.W = np.maximum(self.W, input_vector * self.beta)
                if np.any(self.W == 0):
                    self.W = np.random.rand(self.input_size)
                    break

def main():
    input_size = 5
    network = ARTNetwork(input_size)

    training_data = [
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0]
    ]

    for data in training_data:
        network.learn(data)

    test_data = [
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0]
    ]

    for data in test_data:
        prediction = network.predict(data)
        print(f"Input: {data}, Predicted Class: {'A' if prediction else 'B'}")

if __name__ == "__main__":
    main()
