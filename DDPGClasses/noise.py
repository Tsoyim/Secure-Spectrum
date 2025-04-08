import numpy as np
import matplotlib.pyplot as plt
class OUNoise:
    def __init__(self, action_dimension=1, mu=0, theta=0.2, sigma=1.0):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.dt = 1e-2
        self.reset()

    def reset(self, var=0.5):
        self.sigma = var
        self.theta = 0.2
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    noise = OUNoise()
    arr = []
    for i in range(10):
        arr.append(noise.noise())

    plt.plot(arr)
    plt.show()
