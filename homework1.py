
from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys, os, time, zipfile
import random as rand
from collections import defaultdict
from multiprocessing import Pool


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def eucl_dist(self, p1) -> float:
        return (((self.x-p1.x)**2) + (self.y - p1.y)**2)**0.5
    
    def __str__(self) -> str:
        return "\nPoint: (" + str(self.x) + "," + str(self.y) + ")"
    
    def __eq__(self, __value: object) -> bool:
        if self.x == __value.x and self.y == __value.y: return True
        return False
    
    def __hash__(self) -> int:
        return hash(self.x + self.y)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    
def eucl_dist(p1,p2):
    return (((p1[0] - p2[0])**2) + (p1[1]-p2[1])**2)**0.5

# def threading(args):
#     point, neighbors, D = args
#     counter = 0
#     for n in neighbors:
#         if point != n:
#             d = eucl_dist(point, n)
#             if d <= D:
#                 counter += 1
#     return point, counter

# def dist_calc(coords, D, M, K):
#     """Parallel Approach"""
#     point_dist = {}
#     arg_list = [(point, coords, D) for point in coords]
    
#     with Pool() as pool:
#         results = pool.map(threading, arg_list)
#     for point, counter in results:
#         if counter <= M:
#             point_dist[str(point)] = counter
#     sorted_outliers = dict(sorted(point_dist.items(), key=lambda x:x[1]))
#     sorted_outlier = list(sorted_outliers.keys())
#     return sorted_outlier

def threading(args):
    point, points, D, M = args
    count = 0
    
    for p in points:
        dist = eucl_dist(point, p)
        if dist <= D:
            count += 1
        if count > M:
            break
    
    if count <= M:
        return point, count
    else:
        return point, None
        
    
def exact_count(coords, D, M, K):
    outliers = {}
    arg_list = [(point, coords, D, M) for point in coords]
    with Pool() as pool:
        results = pool.map(threading, arg_list)
        
    for point, count in results:
        if count != None:
            outliers[str(point)] = count
    # for point1 in coords:
    #     count = 0
    #     spoint = str(point1)
    #     for point2 in coords:
    #         if point1 != point2:
    #             dist = eucl_dist(point1,point2)
    #             if dist <= D:
    #                 count += 1
    #             if count > M:
    #                 break
    #     if count == 0:
    #         outliers[spoint] = 0
    #     elif count <= M:
    #         outliers[spoint] = count
    # while len(coords) > 1:
    #     p0 = coords.pop(0)
    #     sp0 = str(p0)
    #     count = 0
    #     for coord in coords:
    #         dist = eucl_dist(p0,coord)
    #         if dist <= D:
    #             count += 1
    #             outliers[str(coord)] += 1
    #         if count > M: 
    #             break
        
        # if count == 0:
        #     outliers[sp0] = 0
        # elif count <= M:
        #     outliers[sp0] = count
            
    # new_outlier = outliers.copy()
    
    # for outlier, neighbors in outliers.items():
    #     if neighbors > M:
    #         new_outlier.pop(outlier)

    # outliers.clear()
    sorted_outliers = dict(sorted(outliers.items(), key=lambda x:x[1]))
    sorted_outlier = list(sorted_outliers.keys())
    return sorted_outlier
        


def MRApproxOutliers(points_RDD, D, M, K):
    
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
    
    file_name, D, M, K, L = sys.argv[1:]
    assert os.path.isfile(file_name), "File or folder not found"
    D, M, K, L = float(D), int(M), int(K), int(L)
    print(f"data path: {file_name}, D: {D}, M: {M}, K: {K}, L: {L}")
    

            
    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)
    

 

    point_map = []
    if file_name.endswith(".zip"):
        with zipfile.ZipFile(file_name) as z:
            fn = file_name.removesuffix(".zip")
            with z.open(fn, "r") as f:
                
            #Exact algo
                for line in f:
                    coord = line.decode().split(",")
                    point_map.append([float(coord[0]), float(coord[1])])
            t1 = time.time()
            print(fn, " D=", D, " M=", M, " K=", K)
            print("Number of points = ", len(point_map))
            outliers = exact_count(point_map, float(D), int(M), int(K))
            print("Number of Outliers = ", len(outliers))
            print(outliers[:int(K)])
            ftime = round((time.time() - t1)*1000)
            print (f"Running time of ExactOutliers = {ftime} ms")
            
            #Approx algo
            rawData = sc.textFile(fn).repartition(L).cache() #L is num of partitions
            points_rdd = rawData.map(lambda line: tuple(map(float, line.split(','))))
            points_num = points_rdd.count()
            print(f"Number of points = {points_num}")
            time_start = time.time()
            MRApproxOutliers(points_rdd, D, M, K)
            time_stop = time.time()
            time_ms = round((time_stop - time_start)*1000)
            print (f"Running time of MRApproxOutliers = {time_ms} ms")

    

if __name__ == "__main__":
    main()

