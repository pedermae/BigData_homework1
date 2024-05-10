from pyspark import SparkContext, SparkConf
from math import sqrt, floor, dist
import sys, os, time
import random as rand        
import numpy as np

conf = SparkConf().setAppName('HW2')
sc = SparkContext(conf=conf)
P = []
S = []
Assign = {}
        
def MRApproxOutliers(points_RDD, D, M):

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
    print("Sure Outliers: ", sure_outliers)
    print("Uncertain Outliers: ", uncertain_outliers)
        

        
def eucl_dist(p1,p2):
    return dist(p1,p2)
        

def SequentialFFT(P: list, K: int) -> list:
    dist = []
    S = list()
    
    try:
        new_center = P[rand.randint(0, len(P) - 1)]
        S.append(new_center)
    except ValueError: 
        return []

    for i, point in enumerate(P):
        dist.append(eucl_dist(point, new_center))

    while len(S) < K:
        for i in np.argsort(dist)[::-1]:
            if P[i] not in S:
                new_center = P[i]
                S.append(new_center)
                break

        for i, point in enumerate(P):
            new_dist = eucl_dist(point, new_center)
            if new_dist < dist[i]:
                dist[i] = new_dist

    return S

def radius_calculation(point, cluster_centers):
    centers = cluster_centers.value
    return min(eucl_dist(point, c) for c in centers)            

    
def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python HW1.py <file_name> <M> <K> <L>"
    
	# INPUT READING
    
    file_name, M, K, L = sys.argv[1:]
    M, K, L = int(M), int(K), int(L)
    print("Data Path: ", file_name, ", M:", M, ", K:", K, ", L:", L)
            
    rawData = sc.textFile(file_name) #L is num of partitions
    inputPoints = rawData.map(lambda line: tuple(map(float, line.split(','))))
    inputPoints = inputPoints.repartition(L).cache()
    points_num = inputPoints.count()
    print("Number of Points: ", points_num)
    
    time_start = time.time()
    round1 = inputPoints.mapPartitions(lambda partition: SequentialFFT(list(partition), K)).persist()
    round1.count()
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print("Running time of MRFFT Round 1 = ", time_ms, " ms")
    
    time_start = time.time()
    round2 = SequentialFFT(round1.collect(), K)
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print("Running time of MRFFT Round 2 = ", time_ms, " ms")
    cluster_centers = sc.broadcast(round2)
    
    time_start = time.time()
    round3 = inputPoints.map(lambda x: radius_calculation(x, cluster_centers)).reduce(max)
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print("Running time of MRFFT Round 3 = ", time_ms, " ms")
    print("Radius = ", round3)
    
    time_start = time.time()
    MRApproxOutliers(inputPoints, round3, M)
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print("Running Time of MRApproxOutliers = ", time_ms, " ms")

    

if __name__ == "__main__":
    main()

