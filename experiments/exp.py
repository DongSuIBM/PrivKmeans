import sys
sys.path.append("..")
import utilities
from utilities.utilities import *

import UG
import UG.UG
import kmeans
import kmeans.synthe_kmeans_weighted_points

NUM_CENTROIDS_SET = 30
NUM_REPEATS = 1

class Experiment(object):
    def __init__(self):
        self.datapath = {
            'S1': (15, 2, None, ""),
        }

        self.epsilon_list = [1.0]

        self.anonymization_methods = {
            'HDUGExtended': UG.UG.uniform_grid_HD_extended,
        }

        self.experiment_results = {}

    def run_experiments(self, evalF):

        print '=================================================Run Experiments================================================='

        for datasetName, datasetPara in sorted(self.datapath.iteritems()):

            for method_name, method in sorted(self.anonymization_methods.iteritems()):

                print '\ndatasetName = ', datasetName, ', datasetPara = ', datasetPara, ', method_name = ', method_name

                nbobs, nbfeatures, min_max, data = load_processed_dataset(datasetName, datasetPara[1])

                true_centroids = load_true_centroids(datasetName)

                for eps in sorted(self.epsilon_list):

                    for rep in xrange(NUM_REPEATS):

                        synopsis = method(data, eps)

                        for i in xrange(NUM_CENTROIDS_SET):

                            initial_centroids = self.load_initial_centroids(datasetName, datasetPara[0], i)
                            initial_centroids_np = [np.array(c, float) for c in initial_centroids]

                            #print 'initial_centroids = ', initial_centroids
                            print '\neps = %f, rep = %d, init_c = %d, method_name = %s' % (eps, rep, i, method_name)

                            nicv_trueData, nicv_synthData, centroids = self.one_experiment(data, synopsis['cell_count_list'], synopsis['cell_center_list'], initial_centroids_np, datasetPara[0], true_centroids, nbfeatures, synopsis['grid_info'])
                            print '\n\tnicv_trueData = %f, nicv_synthData = %f' % (nicv_trueData, nicv_synthData)
                            self.experiment_results[(datasetName, eps, i, rep, method_name)] = (synopsis['m_list'], nicv_trueData, nicv_synthData)

        print 'self.experiment_results = ', self.experiment_results

        for datasetName, datasetPara in sorted(self.datapath.iteritems()):
            evalF.write('%s\t' % datasetName)
            for method_name, method in sorted(self.anonymization_methods.iteritems()):
                evalF.write('%s\t' % method_name)
            evalF.write('\n')
            for eps in sorted(self.epsilon_list):
                for rep in xrange(NUM_REPEATS):
                    for i in xrange(NUM_CENTROIDS_SET):
                        evalF.write('%.3f\t%d\t%d\t' % (eps, rep, i))
                        for method_name, method in sorted(self.anonymization_methods.iteritems()):
                            res_tuple = self.experiment_results[(datasetName, eps, i, rep, method_name)]
                            m_list_s = '\t'.join([ str(elem) for elem in res_tuple[0]])
                            evalF.write('%s \t %.8f \t%.8f' % (m_list_s, res_tuple[1], res_tuple[2]))
                        evalF.write('\n')

    def load_initial_centroids(self, datasetName, num_clusters, centroids_set_idx):

        fName = '../datasets/' + datasetName + '/init_centroids/' + str(num_clusters) + '/' + datasetName + '-' + str(centroids_set_idx) + '-init-centroids.txt'
        init_centroids = load_init_centroids(fName)
        return init_centroids

    #To avoid the memory issue, we have to use released grids rather than the synthe_data, the list of Point objects.
    def one_experiment(self, true_data, cell_count_list, cell_center_list, initial_centroids_np, num_clusters, true_centroids, dim, grid_info):

        centroids = kmeans.synthe_kmeans_weighted_points.synthe_hard_k_means_weighted_points_on_cell_count_list_cell_center_list(cell_count_list, cell_center_list, num_clusters, initial_centroids_np, dim, grid_info)
        aligned_actual_centroids = centroids_alignment(centroids, true_centroids)
        centroids = aligned_actual_centroids.tolist()

        true_centroids_np = np.array(true_centroids, float)
        act_mse_list = []
        for i in xrange(num_clusters):
            squared_diff = np.square(aligned_actual_centroids[i] - true_centroids_np[i])
            act_mse_list.append(squared_diff.sum())

        nicv_trueData = NICV(true_data, centroids)
        nicv_synthData = NICV_weighted_points_on_cell_count_list_cell_center_list(cell_count_list, cell_center_list, centroids)
        return nicv_trueData, nicv_synthData, centroids

if __name__ == "__main__":
    e = Experiment()
    evalF = open('evaluation.txt', 'w')
    e.run_experiments(evalF)
    evalF.close()
