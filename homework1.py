
from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys, os, time
import random as rand


def MRApproxOutliers(points_RDD, D, M, K):
    
    #Step A
    Lambda = D / (2*sqrt(2))
    
    def map_to_cells(point):
        i = floor(point[0]/Lambda)
        j = floor(point[1]/Lambda)
        return ((i,j), 1)
    
    cells_RDD = points_RDD.map(map_to_cells).reduceByKey(lambda a, b: a + b)
    
    local_cells = cells_RDD.collectAsMap() #Think we can download locally here

    #Step B
    def calculate_N3_N7(cell):
        i, j = cell[0]
        N3 = sum([local_cells.get((i+di, j+dj), 0) for di in range(-1, 2) for dj in range(-1, 2)])
        N7 = sum([local_cells.get((i+di, j+dj), 0) for di in range(-3, 4) for dj in range(-3, 4)])
        return (cell[0], (cell[1], N3, N7))
        
    cells_with_N3_N7 = cells_RDD.map(calculate_N3_N7)
    
    
    #Structure of x : ((i,j), (count, N3, N7))
    
    non_outliers = cells_with_N3_N7.filter(lambda x: x[1][1] > M).count()
    sure_outliers = cells_with_N3_N7.filter(lambda x: x[1][2] <= M).count()
    uncertain_outliers = cells_with_N3_N7.filter(lambda x: x[1][1] <= M and x[1][2] > M).count()
    
    
    
    first_K_cells = cells_RDD.sortBy(lambda x: x[1], ascending=True).take(K)
    
    
        
    print("Number of sure (D,M)-outliers: ", sure_outliers)
    print("Number of uncertain points: ", uncertain_outliers)
    print("Number of non outlier points: ", non_outliers)
        
    for cell, size in first_K_cells:
        print(f"Cell: {cell}, Size: {size}")

def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 6, "Usage: python HW1.py <file_name> <D> <M> <K> <L>"
    
	# INPUT READING
    
    data_path, D, M, K, L = sys.argv[1:]
    assert os.path.isfile(data_path), "File or folder not found"
    D, M, K, L = float(D), int(M), int(K), int(L)
    print(f"data path: {data_path}, D: {D}, M: {M}, K: {K}, L: {L}")

	
    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)
    
    # Read the file and parse the points
    rawData = sc.textFile(data_path, L) #L is num of partitions
    points_rdd = rawData.map(lambda line: tuple(map(float, line.split(','))))
    points_num = points_rdd.count()
    print(f"Number of points = {points_num}")
        
    if points_num <= 200000:
        points_list = points_rdd.collect()
        print("Less than 200k points")
        time_start = time.time()
        MRApproxOutliers(points_rdd, D, M, K)
        time_stop = time.time()
        time_ms = (time_stop - time_start)*1000
        print (f"Running time of MRApproxOutliers = {time_ms} ms")
    else:
        time_start = time.time()
        MRApproxOutliers(points_rdd, D, M, K)
        time_stop = time.time()
        time_ms = (time_stop - time_start)*1000
        print (f"Running time of MRApproxOutliers = {time_ms} ms")
        
    
    
    

if __name__ == "__main__":
    main()
