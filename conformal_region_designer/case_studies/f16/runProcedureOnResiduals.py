
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


from sklearn.cluster import MeanShift, estimate_bandwidth


import sys
sys.path.append("../ellipseCode/")
from cmaesExMultipleEllipse import *


import pickle

def main():

    prediction_future_window = 25#20#10#2#5#10
    skip_steps = 3#5


    data_folder = 'data/stackedOutput/3400DataPoints/pred_future_window_' + str(prediction_future_window) + '/skip_step_' + str(skip_steps)
    image_dir = 'images/stackedOutput/3400DataPoints/pred_future_window_' + str(prediction_future_window) + '/skip_step_' + str(skip_steps) + '/alt_theta/'

    residuals_save_file = data_folder + "/testResiduals/"
    with open(residuals_save_file + "residuals.pkl","rb") as f:
        all_errs_one_point = pickle.load(f)

    
    dataset = np.array(all_errs_one_point)

    bandwidth = estimate_bandwidth(dataset, quantile=0.2, n_samples=500)
    bandwidth = bandwidth/1.5
    clustering = MeanShift(bandwidth=bandwidth).fit(dataset)

    labels = clustering.labels_
    cluster_centers = clustering.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)


    plt.figure()
    plt.clf()

    colors = ["orange", "blue", "green", "red"]
    markers = ["x", "o", "^", "*"]

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(dataset[my_members, 0], dataset[my_members, 1], markers[k], color=col)
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.savefig(image_dir + "clusteredResiduals.png")
    plt.clf()




    delta = 0.05
    Q_matrices,ellipse_centers = callCMAESMatrixManyEllipse(dataset,delta,n_clusters_,ellipse_centers=cluster_centers)

    Q_matrices = np.array(Q_matrices)
    ellipse_centers = np.array(ellipse_centers)

    D_cp = computeCPEllipseMatrixManyEllipse(dataset,Q_matrices,ellipse_centers,delta)
    print("D_cp: " + str(D_cp))



    plt.clf()
    plt.plot(dataset[:,0], dataset[:,1], 'o')
    for i in range(n_clusters_):
        plot_ellipse(ellipse_centers[i],Q_matrices[i],D_cp,'green','ellipse ' + str(i))
    plt.legend()
    plt.savefig(image_dir + "residualsWithEllipses.png")
    plt.clf()

if __name__ == '__main__':
    main()
