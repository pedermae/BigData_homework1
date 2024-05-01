
from pyspark import SparkContext, SparkConf, RDD
from math import sqrt, floor
import sys, os, time, zipfile
import random as rand        


def MRApproxOutliers(points_RDD, D, M):
    
    #Step A
    Lambda = D / (2*sqrt(2))
    
    def map_to_cells(point):
        i = floor(point[0]/Lambda)
        j = floor(point[1]/Lambda)
        return ((i,j), 1)
    
    cells_RDD = points_RDD.map(map_to_cells).reduceByKey(lambda a, b: a + b)
    
    local_cells = cells_RDD.collectAsMap() #Think we can download locally here, not sure if already done in caching in Main.

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
    ordered_cells = cells_RDD.sortBy(lambda x: x[1], ascending=True)
    
    print("Number of sure (D,M)-outliers: ", sure_outliers)
    print("Number of uncertain points: ", uncertain_outliers)
    print("Number of non outlier points: ", non_outliers)
        
    for cell, size in ordered_cells:
        print(f"Cell: {cell}, Size: {size}")
        
        
def eucl_dist(p1,p2):
        return (((p1[0] - p2[0])**2) + (p1[1]-p2[1])**2)**0.5
        

def SequentialFFT(P: list, K: int) -> list:
    print("P: ", P)
    try:
        S = [P.pop(rand.randint(0, len(P) - 1))]
    except ValueError:
        return []

    while len(S) < K:
        point = max(P, key=lambda x: min(eucl_dist(x, c) for c in S))
        S.append(point)
        P.remove(point)
    
    print("S: ", S)
    return S

        
    
def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python HW1.py <file_name> <M> <K> <L>"
    
	# INPUT READING
    
    file_name, M, K, L = sys.argv[1:]
    assert os.path.isfile(file_name), "File or folder not found"
    M, K, L = int(M), int(K), int(L)
    print(f"data path: {file_name}, M: {M}, K: {K}, L: {L}")
    

            
    conf = SparkConf().setAppName('HW2')
    sc = SparkContext(conf=conf)
    
    point_map = []
    rawData = sc.textFile(file_name).repartition(L).cache() #L is num of partitions
    inputPoints = rawData.map(lambda line: tuple(map(float, line.split(','))))
    points_num = inputPoints.count()
    print(f"Number of points = {points_num}")
    round1 = inputPoints.mapPartitions(lambda partition: SequentialFFT(list(partition), K))
    round1 = round1.collect()
    print("Round1: ", round1)
    round2 = SequentialFFT(round1, K)
    print("Round2: ", round2)
    round3 = inputPoints.map(lambda x: min(eucl_dist(x,c) for c in round2)).max()
    print("Round3: ", round3)
    time_start = time.time()
    MRApproxOutliers(inputPoints, round3, M)
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print (f"Running time of MRApproxOutliers = {time_ms} ms")

    

if __name__ == "__main__":
    main()

