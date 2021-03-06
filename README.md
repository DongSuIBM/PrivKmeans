# PrivKmeans: Differentially Private K-Means Clustering
Developed by Dong Su <sudong.tom@gmail.com>

If you use this code for any of your work, please cite the following article:
[1] Dong Su, Jianneng Cao, Ninghui Li, Elisa Bertino, Hongxia Jin: Differentially Private K-Means Clustering. CODASPY 2016: 26-37

Datasets
------------
In the datasets directory, S1_processed.txt contains the dataset of the 2D S1 dataset.  datasets/init_centroids directory contains the initial centroids generated by our proposed initial centroids generation method in Section 3.1.3.  S1-best-nicv.txt file contains the best NICV that NoPrivacy k-means can achieve on this dataset.  S1-true-centroids.txt contains the true centroids of clusters in S1. 

How to run?
------------
The entrance of the experiment is the experiments/exp.py file.  The experimental results is in experiments/evaluation.txt file.  Our EUGkM method is implemented UG/UG.py.  The initial centroids are generated in the utility/generate_init_centroids.py.  

Running environment:
-------------
Python 2.7.6, Numpy 1.10.2. 
