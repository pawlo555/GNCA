"""
This script shows an animation of the Voronoi GCA.
"""
import matplotlib.pyplot as plt
import numpy as np
from spektral.layers.ops import sp_matrix_to_sp_tensor

#from models import GNNCASimple
from matplotlib import animation

from modules.ca import VoronoiCA



def plotting():
    n_cells = 1000
    mu = 0.0
    sigma = 0.50
    steps = 1000

    # Run
    initial_state = np.random.randint(0, 2, n_cells)

    gca = VoronoiCA(n_cells, mu=mu, sigma=sigma)
    # model = GNNCASimple(activation="sigmoid", batch_norm=False)
    # a = sp_matrix_to_sp_tensor(gca.graph.a)
    # Save the weights
    # model.save_weights('./checkpoints/my_checkpoint')

    # Restore the weights
    # model.load_weights('./trained_model')
    for step in range(steps):
        pass
    history = gca.evolve(initial_state, steps=steps)

    # Animation
    # plt.ion()
    plt.show()
    for i, state in enumerate(history):
        print(plt.isinteractive())
        gca.plot(state)
        print(i)
        plt.draw()
        plt.show()
        plt.pause(0.1)

if __name__ == '__main__':
    plotting()
