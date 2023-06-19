from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def create_kde(residuals):
    """
    This function creates a KDE object that estimates the Kernel Density of the residuals
    :param residuals:
    :return: gaussian_kde object
    """
