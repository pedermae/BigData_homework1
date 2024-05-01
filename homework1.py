
from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys, os, time, zipfile
import random as rand
from collections import defaultdict
from multiprocessing import Pool

def MRApproxOutliers(rawData, D, M, K, L):
    points_RDD = rawData.map(lambda line: tuple(map(float, line.split(','))))
    points_RDD.repartition(L).cache()

    def map_to_cell(point):
        Lambda = D / (2*sqrt(2))
        i = floor(point[0]/Lambda)
        j = floor(point[1]/Lambda)
        return (i,j)
    
    #Assign each point to a cell
    cells_RDD = points_RDD.map(lambda point: (map_to_cell(point), 1)).reduceByKey(lambda a , b: a + b)
    local_cells = cells_RDD.collectAsMap()
    
    #Calculate N3 and N7 for each cell
    count_N3N7 = {}
    for cell in local_cells.items():
        count_N3N7[cell[0]] = [cell[1], 0, 0]
        
        for i in range(-3, 4):
            for j in range(-3, 4):
                cell_ij = (cell[0][0] + i, cell[0][1] + j)
                if cell_ij in local_cells:
                    cell_count = local_cells[cell_ij]
                    if (i < -1 or i > 1) or (j < -1 or j > 1):
                        #In C7
                        count_N3N7[cell[0]][2] += cell_count
                    else:                                   
                        #In C3
                        count_N3N7[cell[0]][2] += cell_count
                        count_N3N7[cell[0]][1] += cell_count

        #Calculate outliers
        sure_outliers = 0
        uncertain_outliers = 0
        for cell, counts in count_N3N7.items():
            if counts[2] <= M:
                sure_outliers += counts[0]
            
            if counts[1] <= M and counts[2] > M:
                uncertain_outliers += counts[0]
    print(f"Sure outliers: {sure_outliers}")
    print("Uncertain outliers: {uncertain_outliers}")
    first_K_cells = cells_RDD.sortBy(lambda x: x[1], ascending=True).take(K)
    for cell, size in first_K_cells:
        print(f"Cell: {cell}, Size: {size}")
    
    
def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 6, "Usage: python HW1.py <file_name> <D> <M> <K> <L>"
    file_name, D, M, K, L = sys.argv[1:]
    assert os.path.isfile(file_name), "File or folder not found"
    D, M, K, L = float(D), int(M), int(K), int(L)
    print(f"data path: {file_name}, D: {D}, M: {M}, K: {K}, L: {L}")
    
    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)
    if file_name.endswith(".zip"):
        fn = file_name.removesuffix(".zip")
        rawData = sc.textFile(fn)
        time_start = time.time()
        MRApproxOutliers(rawData, D, M, K, L)
        time_stop = time.time()
        time_ms = round((time_stop - time_start)*1000)
        print(f"Running time of MRApproxOutliers = {time_ms} ms")
    

if __name__ == "__main__":
    main()