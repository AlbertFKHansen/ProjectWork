import numpy as np
import matplotlib.pyplot as plt

# Create a range of x and y values
x = np.linspace(-5, 5, 400)
y = np.zeros(400)

# Define gamma values for illustration
gammas = [0.5, 1, 2]


def RBF(x, y, gamma):
    return np.exp(-gamma * (x-y)**2)


def poly_gamma_example(x, y, gamma, c, d):
    return (gamma * x*y + c) ** d


def sigmoid_gamma_example(x, y, gamma, c):
    return np.tanh(gamma * x*y + c)


# Create the plot for RBF
plt.figure(figsize=(8, 5))
for gamma in gammas:
    k = RBF(x, y, gamma)
    plt.plot(x, k, label=f'γ = {gamma}')

plt.title('Effect of γ on the Gaussian (RBF) Kernel')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.legend()
plt.grid(True)
plt.show()

y = np.ones_like(x)

# Create the plot for Poly
plt.figure(figsize=(8, 5))
for gamma in gammas:
    k = poly_gamma_example(x, y, gamma, 1, 2)
    plt.plot(x, k, label=f'γ = {gamma}')

plt.title('Effect of γ on the Polynomial Kernel')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.legend()
plt.grid(True)
plt.show()

# Create the plot for sigmoid
plt.figure(figsize=(8, 5))
for gamma in gammas:
    k = sigmoid_gamma_example(x, y, gamma, 1)
    plt.plot(x, k, label=f'γ = {gamma}')

plt.title('Effect of γ on the Polynomial Kernel')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.legend()
plt.grid(True)
plt.show()