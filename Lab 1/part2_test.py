from part2 import *

def main():
    weight_1 = np.array([1, 2, 3, 4])
    weight_2 = np.array([-1, -2, -3, -4])
    bias_1 = 3
    bias_2 = -2
    alpha = 0.1
    input = np.array([10, 5, -5, -10])
    print("Input: ", input)

    elem_mult_1 = ElementwiseMultiply(weight_1)
    add_bias_1 = AddBias(bias_1)
    leaky_relu = LeakyRelu(alpha)
    elem_mult_2 = ElementwiseMultiply(weight_2)
    add_bias_2 = AddBias(bias_2)
    layers = Compose([elem_mult_1, add_bias_1, leaky_relu, elem_mult_2, add_bias_2, leaky_relu])
    output = layers(input)

    print("Output:", output)

if __name__ == '__main__':
    main()