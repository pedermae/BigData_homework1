from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys
import os
import random as rand


def MRApproxOutliers(points_RDD, D, M, K):
    #RDD_points should be an RDD of points
    L = D / (2*sqrt(2))
    
    ###Step A: Transforms RDD_points to RDD of the non empty cells, and (i,j) for each cell and number of points it contains.
    
    cells_RDD = points_RDD.map(lambda p : ((floor(p[0]/L), floor(p[1]/L)), 1)).reduceByKey(lambda a, b: a + b)

    ###Step B:
        
    #Compute N3 and N7 for each cell
    N3_RDD = cells_RDD.flatMap(lambda c: [((c[0][0] + i, c[0][1] + j), c[1]) for i in range(-1, 2) for j in range(-1, 2) if c[1] > 0]).reduceByKey(lambda a, b: a + b)
    N7_RDD = cells_RDD.flatMap(lambda c: [((c[0][0] + i, c[0][1] + j), c[1]) for i in range(-3, 4) for j in range(-3, 4) if c[1] > 0]).reduceByKey(lambda a, b: a + b)
    
    
    #Compute number of sure and uncertain outliers
    sure_outliers = N3_RDD.filter(lambda c: c[1] <= M).subtractByKey(N7_RDD.filter(lambda c: c[1] > M)).count()
    uncertain_outliers = N3_RDD.filter(lambda c: c[1] <= M).join(N7_RDD.filter(lambda c: c[1] > M)).count()
    
    print("Len of N3: ", N3_RDD.count())
    print("Len of N7: ", N7_RDD.count())
    
    print("Number of sure outliers: ", sure_outliers)
    print("Number of uncertain outliers: ", uncertain_outliers)
    
    
    size_sorted_cells = cells_RDD.map(lambda c : (c[1], c[0])).sortByKey()
    first_K_cells = size_sorted_cells.take(K)
    
    for size, cell in first_K_cells:
        print(f"Cell: {cell}, Size: {size}")
    

def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 3, "Usage: python HW1.py <K> <file_name>"
    
	# INPUT READING

	# 1. Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

	# 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
	
    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)

    # Read the file and parse the points
    points_rdd = sc.textFile(data_path).map(lambda line: tuple(map(float, line.split(','))))
    
    for point in points_rdd.collect():
        print(point)
    
    MRApproxOutliers(points_rdd, 1, 3, 9)
    
    

if __name__ == "__main__":
    main()
