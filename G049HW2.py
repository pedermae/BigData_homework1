from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys, os, time
import random as rand        

        
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
    print(f"Sure outliers: {sure_outliers}")
    print(f"Uncertain outliers: {uncertain_outliers}")
        

        
def eucl_dist(p1,p2):
        return (((p1[0] - p2[0])**2) + (p1[1]-p2[1])**2)**0.5
        

def SequentialFFT(P: list, K: int) -> list:
    try:
        S = [P.pop(rand.randint(0, len(P) - 1))]
    except ValueError:
        return []

    while len(S) < K:
        point = max(P, key=lambda x: min(eucl_dist(x, c) for c in S))
        S.append(point)
        P.remove(point)
    
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
    
    rawData = sc.textFile(file_name) #L is num of partitions
    inputPoints = rawData.map(lambda line: tuple(map(float, line.split(','))))
    inputPoints = inputPoints.repartition(L).cache()
    points_num = inputPoints.count()
    print(f"Number of points = {points_num}")
    time_start = time.time()
    round1 = inputPoints.mapPartitions(lambda partition: SequentialFFT(list(partition), K))
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print (f"Running time of Round1 = {time_ms} ms")
    round1 = round1.collect()
    print("Round1: ", round1)
    time_start = time.time()
    round2 = SequentialFFT(round1, K)
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print (f"Running time of Round2 = {time_ms} ms")
    print("Round2: ", round2)
    time_start = time.time()
    round3 = inputPoints.map(lambda x: min(eucl_dist(x,c) for c in round2)).max()
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print (f"Running time of Round3 = {time_ms} ms")
    print("Round3: ", round3)
    time_start = time.time()
    MRApproxOutliers(inputPoints, round3, M)
    time_stop = time.time()
    time_ms = round((time_stop - time_start)*1000)
    print (f"Running time of MRApproxOutliers = {time_ms} ms")

    

if __name__ == "__main__":
    main()

