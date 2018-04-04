import random, copy, math, sys, os, operator, itertools
import numpy as np
import time, shutil

# Eucldean distance
def euclidean_distance(x,y):
    return math.sqrt(np.dot(x-y,(x-y).conj()))


def euclidean_distance_squared(x,y): 
    return (np.dot(x-y,(x-y).conj()))


def calc_label(records, centroids):
    label = []
    for data in records:
        dist = [euclidean_distance(np.array(data, float), np.array(centroid, float)) for centroid in centroids]
        label.append(dist.index(min(dist)))
    return label

def calc_label_on_cell_count_list_cell_center_list(cell_center_list, centroids):
    label = []

    dim = len(centroids[0])
    for i in xrange(len(cell_center_list)):
        dist = [euclidean_distance(cell_center_list[i], np.array(centroid, float)) for centroid in centroids]
        label.append(dist.index(min(dist)))

    return label

def NICV(data, centroids):

    label = calc_label(data, centroids)

    intra_dist = {}
    for i in range(len(centroids)):
        intra_dist[i] = 0

    for i in range(len(data)):
#		print 'data[%d] = %d' % (i, label[i])
        intra_dist[label[i]] += euclidean_distance_squared(np.array(data[i], float), np.array(centroids[label[i]], float))

    icv = 0
    for dist in intra_dist.values():
        icv += dist

    nicv = icv/len(data)
    return nicv

def NICV_weighted_points_on_cell_count_list_cell_center_list(cell_count_list, cell_center_list, centroids):

    total_num_data = sum(cell_count_list)

    label = calc_label_on_cell_count_list_cell_center_list(cell_center_list, centroids)

    intra_dist = {}
    for i in range(len(centroids)):
        intra_dist[i] = 0

    for i in xrange(len(cell_count_list)):
        intra_dist[label[i]] += euclidean_distance_squared(cell_center_list[i], np.array(centroids[label[i]], float)) * cell_count_list[i]

    icv = 0
    for dist in intra_dist.values():
        icv += dist

    nicv = icv/total_num_data

    return nicv


def compute_bounding_box(dataset):

    bounding_box = []
    for i in xrange(len(dataset[0])):
        bounding_box.append([np.amin(dataset[:,i]), np.amax(dataset[:,i])])

    return bounding_box


def load(fName, dim):

    f = open(fName, 'r')

    data_list = []
    for i, line in enumerate(f):
        t = []
        for item in line.split():
            item = float(item)
            t.append(item)
        data_list.append(np.array(t, float))

#	print data_list
    data = np.array(data_list)
    #print data.shape
    nobs, nfeatures = data.shape

    min_max = []
    for i in xrange(dim):
        min_max.append([np.amin(data[:,i]), np.amax(data[:,i])])

    return nobs, nfeatures, min_max, data


def load_processed_dataset(datasetName, dim):
    fName = '../datasets/' + datasetName + '/'
    fName += datasetName + '_processed.txt'

    nobs, nfeatures, min_max, data = load(fName, dim)

    return nobs, nfeatures, min_max, data


def load_init_centroids(datasetName):
    f = open(datasetName, 'r')
    centroids = []
    for i, line in enumerate(f):
        newLine = line.split()
        if len(newLine) <= 0:
            continue
        point = []
        for item in newLine:
            point.append(item)
        centroids.append(point)

    f.close()
    return centroids

def load_true_centroids(datasetName):
    prefix = '../datasets/' + datasetName + '/'
    suffix = '-true-centroids.txt'

    f = open(prefix + datasetName + suffix, 'r')
    centroids = []
    for i, line in enumerate(f):
        newLine = line.split()
        if len(newLine) <= 0:
            continue
        point = []
        for item in newLine:
            point.append(item)
        centroids.append(point)

    f.close()
    return centroids

def centroids_alignment(centroids, true_centroids):

    true_centroids_matched = dict.fromkeys(range(len(centroids)), -1)
    reordered_centroids = []

    for i in xrange(len(true_centroids)):

        c1 = true_centroids[i]
        min_idx = 0
        min_dist = sys.maxint

        for j in xrange(len(centroids)):
            c2 = centroids[j]
            dist = ((np.array(c1, float) - np.array(c2, float)) ** 2).sum()

            if dist < min_dist:
                min_idx = j
                min_dist = dist

        true_centroids_matched[i] = min_idx

    for i in true_centroids_matched.keys():
        matched_idx = true_centroids_matched[i]
        reordered_centroids.append(copy.deepcopy(centroids[matched_idx]))

    return np.array(reordered_centroids)


def write_init_centroids(datasetName, centroids):
    suffix = '-init-centroids.txt'
    f = open(datasetName + suffix, 'w')

    #print 'centroids = ', centroids

    dim  = len(centroids[0])

    for c in centroids:
        line = ''
        for i in xrange(dim):
            line += str(c[i]) + ' '
        line += '\n'
        f.write(line)

    f.close()
    return


def load_real_true_centroids(datasetName, num_clusters):
    prefix = '../../datasets/' + datasetName + '/true_centroids/'
    suffix = '-true-centroids.txt'

    f = open(prefix + datasetName + '-' + str(num_clusters) + suffix, 'r')
    centroids = []
    for i, line in enumerate(f):
        newLine = line.split()
        if len(newLine) <= 0:
            continue
        point = []
        for item in newLine:
            point.append(item)
        centroids.append(point)

    f.close()
    return centroids