from matplotlib.axes import Axes

def set_square_aspect_ratio(ax: Axes):
    ax.set_aspect('equal')
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_mean, y_mean = (x_min + x_max) / 2, (y_min + y_max) / 2
    max_range = max(x_range, y_range)
    ax.set_xlim(x_mean - max_range / 2, x_mean + max_range / 2)
    ax.set_ylim(y_mean - max_range / 2, y_mean + max_range / 2)
