# zerotohero

My code from following Andrej Karpathy's Zero to Hero tutorial series.

For each section, I included an explanation of the important parts of these code. These are mainly for my benefit, and as such, they are
very long-winded.

## Micrograd

Micrograd is a version of the automatic differentiation engines used in deep learning frameworks like PyTorch.

The basic idea behind automatic differentation is that when tuning the weights of a neural network, differentiation of complex functions is required. To compute these derivatives efficiently, we write a program that converts a function into a graph of the dependicies. This graph allows us to organize the order of differentiation using the chain rule.
