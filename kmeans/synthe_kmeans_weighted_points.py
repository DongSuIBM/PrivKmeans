import sys
sys.path.append("..")
import utilities
from utilities.utilities import *

KMEANS_ETA = 0.000001

#do the hard k-means over synthetic data
#using weighted points
def synthe_hard_k_means_weighted_points_on_cell_count_list_cell_center_list(cell_count_list, cell_center_list, nbclusters, init_centroids, dim, grid_info):

    distance = euclidean_distance_squared

    nbobs = len(cell_count_list)
    nbfeatures = dim

    tmpdist = np.ndarray([nbobs,nbclusters], np.float64)

    clusters = [[] for c in xrange(nbclusters)]

    centroids = copy.deepcopy(init_centroids)
    old_centroids = [np.array([-1 for f in xrange(nbfeatures)], np.float64) for c in xrange(nbclusters)]
    old_sum = sys.maxint
    new_sum = math.fsum([distance(centroids[c], old_centroids[c]) for c in xrange(nbclusters)])

    loop = 0

    while True:

        if abs(old_sum - new_sum) < KMEANS_ETA  or loop > 100:
            break

        old_centroids = copy.deepcopy(centroids)
        old_sum = new_sum
        for c in xrange(nbclusters):
            clusters[c] = []

        for c in xrange(nbclusters):
            for o in xrange(nbobs):
                tmpdist[o,c] = np.dot(centroids[c]-cell_center_list[o],(centroids[c]-cell_center_list[o]).conj())

        for o in xrange(nbobs):
            clusters[tmpdist[o,:].argmin()].append(o)

        for c in xrange(nbclusters):
            mean = np.array([0.0 for i in xrange(nbfeatures)], np.float64)
            denom = 0
            for o in clusters[c]:
                coord_p = cell_center_list[o]
                weight = cell_count_list[o]
                mean += coord_p * weight
                denom += weight

            if denom != 0:
                mean = mean / denom

            for i in xrange(nbfeatures):
                if mean[i] < -1.0:
                    mean[i] = -1.0
                if mean[i] > 1.0:
                    mean[i] = 1.0

            centroids[c] = mean

        new_sum = math.fsum([distance(centroids[c], old_centroids[c]) for c in xrange(nbclusters)])
        loop += 1
    #end while

    centroids = [c.tolist() for c in old_centroids]
    return centroids

