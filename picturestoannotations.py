import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_block_sequence(block_sequence):
    
    shape_color_mapping = {
        'a': ('triangle', 'red'),
        'b': ('circle', 'blue'),
        'c': ('rectangle', 'green'),
        'd': ('hexagon', 'purple')
    }

    fig, ax = plt.subplots()
    x_position = 0

    for block in block_sequence:
        shape, color = shape_color_mapping[block]

        if shape == 'triangle':
            triangle = patches.Polygon(
                [[x_position, 0], [x_position, 1], [x_position + 1, 0.5]],
                closed=True, facecolor=color, edgecolor='black'
            )
            ax.add_patch(triangle)
        elif shape == 'circle':
            circle = patches.Circle(
                (x_position + 0.5, 0.5), radius=0.5, facecolor=color, edgecolor='black'
            )
            ax.add_patch(circle)
        elif shape == 'rectangle':
            rectangle = patches.Rectangle(
                (x_position, 0), 1, 1, facecolor=color, edgecolor='black'
            )
            ax.add_patch(rectangle)
        elif shape == 'hexagon':
            hexagon = patches.RegularPolygon(
                (x_position + 0.5, 0.5), numVertices=6, radius=0.5, orientation=np.pi / 6,
                facecolor=color, edgecolor='black'
            )
            ax.add_patch(hexagon)

        x_position += 1

    ax.set_xlim(0, len(block_sequence))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

# Example usage
block_sequence = 'acbcbbcbcbbcbcdadadadccc' #example of a sequence of blocks
plot_block_sequence(block_sequence)

