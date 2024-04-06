#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:54:47 2024

@author: pedermaeland
"""

from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys
import os
import random as rand


def MRApproxOutliers(points_RDD, D, M, K):
    
    #Step A
    Lambda = D / (2*sqrt(2))
    
    def map_to_cells(point):
        i = floor(point[0]/Lambda)
        j = floor(point[1]/Lambda)
        return ((i,j), 1)
    
    cells_RDD = points_RDD.map(map_to_cells).reduceByKey(lambda a, b: a + b)
    
    local_cells = cells_RDD.collectAsMap()

    #Step B
    def calculate_N3_N7(cell):
        i, j = cell[0]
        N3 = sum([local_cells.get((i+di, j+dj), 0) for di in range(-1, 2) for dj in range(-1, 2)])
        N7 = sum([local_cells.get((i+di, j+dj), 0) for di in range(-3, 4) for dj in range(-3, 4)])
        return (cell[0], (cell[1], N3, N7))
        
    cells_with_N3_N7 = cells_RDD.map(calculate_N3_N7)
    
    
    #x : ((i,j), (count, N3, N7))
    non_outliers = cells_with_N3_N7.filter(lambda x: x[1][1] > M).count()
    sure_outliers = cells_with_N3_N7.filter(lambda x: x[1][2] <= M).count()
    uncertain_outliers = cells_with_N3_N7.filter(lambda x: x[1][1] <= M and x[1][2] > M).count()
    
    first_K_cells = cells_RDD.sortBy(lambda x: x[1], ascending=False).take(K)
    
    
        
    print("Number of sure (D,M)-outliers: ", sure_outliers)
    print("Number of uncertain points: ", uncertain_outliers)
    print("Number of non outlier points: ", non_outliers)
        
    print(first_K_cells)

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
    points_rdd = sc.textFile(data_path).map(lambda line: tuple(map(float, line.split(',')))).repartition(2)
    
    for point in points_rdd.collect():
        print(point)
    
    MRApproxOutliers(points_rdd, 1, 3, 9)
    
    

if __name__ == "__main__":
    main()
