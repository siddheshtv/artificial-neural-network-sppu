import numpy as np

class McCullohPittsNeuron:
    def __init__(self, weights, threshold) -> None:
        self.weights = weights
        self.threshold = threshold
        return None
    
    def activate(self, inputs) -> int:
        weighted_sum = np.dot(inputs, self.weights)
        return int(weighted_sum > self.threshold)
    

def main():
    weights = np.array([1, -1])
    threshold = 0

    neuron = McCullohPittsNeuron(weights=weights, threshold=threshold)

    inputs = [
        np.array([0,0]),
        np.array([0,1]),
        np.array([1,0]),
        np.array([1,1]),
    ]

    for input_data in inputs:
        output = neuron.activate(inputs=input_data)
        print(f"Input: {input_data} Output: {output}")

if __name__ == "__main__":
    main()