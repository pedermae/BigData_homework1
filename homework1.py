from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand


def MRApproxOutliers(points_RDD, D, M, K):
    #RDD_points should be an RDD of points
    L = D / (2*sqrt(2))
    
    ###Step A: Transforms RDD_points to RDD of the non empty cells, and (i,j) for each cell and number of points it contains.
    cells_RDD = points_RDD.map(lambda p : ((floor(p[0]/L), floor(p[1]/L)), 1)).reduceByKey(lambda a, b: a + b)

    ###Step B: Compute N3 and N7 for each 
    
    

def main():
    
    
if __name__ == "__main__":
	main()