import numpy as np
import sys, os

sys.path.append("..")
import utilities
from utilities.utilities import *
import itertools

OLD_ETA_UG = 0.1

#return a d-dim vector. Each element specify the num of partitions in the corresponding dimension.  
def calc_m_new_ug(N, dim, epsilon):
    M = (N * epsilon * OLD_ETA_UG) ** (2.0 * dim/(dim+2.0))

    m_ceil = int(math.ceil(M ** (1.0/dim)))
    m_floor = int(math.floor(M ** (1.0/dim)))
    m_list = [m_floor] * dim
    tmp_prod = 1
    for i in xrange(len(m_list)):
        m_list[i] = m_ceil
        tmp_prod = np.product(m_list)
        if tmp_prod > M:
            break
    return m_list


def uniform_grid_HD_extended(dataset, epsilon):
    dim = len(dataset[0])

    m_list = calc_m_new_ug(len(dataset), dim, epsilon)
    bounding_box = compute_bounding_box(dataset)
    all_cells, gran, origin = quantize_HD_extended(dataset, epsilon, dim, m_list, bounding_box)

    grid_info = {}
    grid_info['gran'] = gran
    grid_info['origin'] = origin
    grid_info['bounding_box'] = bounding_box
    grid_info['all_cells'] = all_cells

    cell_count_list = []
    cell_center_list = []

    for key in sorted(all_cells.keys()):
        cell_count_list.append(all_cells[key])
        bounding_box, centroid, centroid_p = reverse_cell_key(key, gran, origin, dim)
        cell_center_list.append(centroid_p)

    synopsis = {"cell_count_list": cell_count_list, "cell_center_list": cell_center_list, "m_list": m_list, "grid_info": grid_info}

    return synopsis


def rounding_point_HD(point, origin, granularity, boundary, m_list):

    boundary_int = []
    for d in xrange(len(point)):
        lr = [int(math.floor((point[d] - origin[d])/granularity[d])), int(math.ceil((point[d] - origin[d])/granularity[d]))]
        boundary_int.append(lr)

    for i in xrange(len(point)):

        if boundary_int[i][0] == boundary_int[i][1]:
            tmp = boundary_int[i][0] - 1
            if tmp < boundary[i][0]:
                boundary_int[i][1] += 1
            else:
                boundary_int[i][0] = tmp


        if boundary_int[i][0] >= m_list[i]:
            boundary_int[i][0] -= 1

    key_list = []
    for i in xrange(len(boundary_int)):
        key_list.append(boundary_int[i][0])

    return tuple(key_list)

#from the cell_key, reconstruct the corresponding boundaries and centroids
def reverse_cell_key(cell_key, granularity, origin, dim):

    bounding_box = []
    centroid = []
    for d in xrange(dim):
        left = cell_key[d] * granularity[d] + origin[d]
        right = left + granularity[d]
        bounding_box.append([round(left, 8), round(right, 8)])
        centroid.append( (left + right)*0.5 )

        centroid_p = np.array(centroid, float)

    return bounding_box, centroid, centroid_p

def correctness_checking(point, point_key, gran, origin, dim):

    cell_bounding_box, centroid, centroid_p = reverse_cell_key(point_key, gran, origin, dim)

    status = [round(cell_bounding_box[d][0], 6) <= round(point[d], 6) <= round(cell_bounding_box[d][1], 6) for d in xrange(dim)]
    #status = [cell_bounding_box[d][0] <= point[d] <= cell_bounding_box[d][1] for d in xrange(dim)]

    if all(status) ==  False:
        print "incorrecty!!!!!"
        print 'status = ', status

    return all(status)

#specialized partition scheme. 
#m_list tells the number of partitions that each dimension should be
def quantize_HD_extended(dataset, epsilon, d, m_list, bounding_box = None):
    print '===================quantize_HD_extended start=================='

    origin = []
    for i in xrange(d):
        origin.append(bounding_box[i][0])

    gran = []
    for i in xrange(d):
        gran.append((bounding_box[i][1] - bounding_box[i][0])*1.0/m_list[i])

    cell_boundary_int = []
    for i in xrange(d):
        tmp = [int(math.floor((bounding_box[i][0] - origin[i])/gran[i])), int(math.ceil((bounding_box[i][1] - origin[i])/gran[i]))]
        cell_boundary_int.append(tmp)

    cell_count_dict = {}
    count = 0

    m_list_range = [range(int(m)) for m in m_list]
    for bb in itertools.product(*m_list_range):

        cell_parameter = []
        for i in xrange(len(bb)):
            cell_parameter.append(tuple([origin[i] + bb[i]*gran[i], origin[i] + (bb[i]+1)*gran[i]]))

        cell_count_dict[bb] = 0
        count += 1

    point_id = 0
    missed_count = 0
    correctness_count = 0

    for point in dataset:
        point_key = rounding_point_HD(point, origin, gran, cell_boundary_int, m_list)

        if cell_count_dict.has_key(point_key) == False:
            print 'missed point = ', point, ', point_key = ', point_key
            missed_count += 1
            continue

        if correctness_checking(point, point_key, gran, origin, d) == True:
            correctness_count += 1
        else:
            print 'INCORRECTNESS, point = ', point, 'point_key = ', point_key

        cell_count_dict[point_key] = cell_count_dict[point_key] + 1

        point_id += 1

    print 'missed_count = ', missed_count, ", correctness_count = ", correctness_count

    verifyCount = 0
    for key in cell_count_dict.keys():
        verifyCount += cell_count_dict[key]

    print "verifyCount = ", verifyCount

    for key in cell_count_dict.keys():
        cell_count_dict[key] = cell_count_dict[key] + np.random.laplace(scale=1.0/epsilon)

    return cell_count_dict, gran, origin


