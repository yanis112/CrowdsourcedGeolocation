import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, linregress, beta
import cmdstanpy as cmd
import seaborn as sn
from fitter import Fitter, get_common_distributions, get_distributions

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
from sklearn.neighbors import KernelDensity


def annotators_reports(real_points, num_annotators, annotators_sigmas, num_map_parts):
  
    
    if num_map_parts == 1:
        rep_points = []

        for point in real_points:
            rep_points_aux = []
            for annotator in range(num_annotators):
                rep_point = point + np.random.normal(loc=0, scale=annotators_sigmas[annotator])
                rep_points_aux.append(rep_point)
            rep_points.append(rep_points_aux)
            
     
    else:
        rep_points = []
        #First we build an array with the coordinates of the map partitions (e.g. if num_map_parts=2 -> map_partitions = [0.5])
        map_partitions = []
        aux_variable = 0
        for parts in range(num_map_parts - 1):
            map_partitions.append(aux_variable + 1/num_map_parts)
            aux_variable += 1/num_map_parts
                
        for point in real_points:
            rep_points_aux = []
            for annotator in range(num_annotators):
                for part in range(num_map_parts-1): #We check all map's parts in order except the last one
                    point_checked = False #This variable will tell us if the point is in one of these parts
                    if point < map_partitions[part]:
                        rep_point = point + np.random.normal(loc=0, scale=annotators_sigmas[annotator][part])
                        rep_points_aux.append(rep_point)
                        point_checked = True #The point was in one of these parts
                        break
                        
                if point_checked == False: #If the point was not in one of these parts, it is in the last part of the map
                    rep_point = point + np.random.normal(loc=0, scale=annotators_sigmas[annotator][-1])
                    rep_points_aux.append(rep_point)

            rep_points.append(rep_points_aux)        
    
    return np.array(rep_points)


def get_map_partitions(num_map_parts):
    
    map_partitions = []
    aux_variable = 0
    for parts in range(num_map_parts - 1):
        map_partitions.append(aux_variable + 1/num_map_parts)
        aux_variable += 1/num_map_parts
        
    return np.array(map_partitions)


def reports_frompool(real_points, num_annotators, annotators_sigmas, num_annotators_per_group):
    
    rep_points = [[-121313 for i in range(num_annotators)] for j in range(len(real_points))]   #Arbitrary mark (-121313)
    index_aux = 0
    
    for point in real_points:
        group = np.random.choice(np.arange(num_annotators), num_annotators_per_group, replace=False)

        for annotator in group:
            rep_point = point + np.random.normal(loc=0, scale=annotators_sigmas[annotator])
            rep_points[index_aux][annotator] = rep_point
            
        index_aux += 1
        
    return rep_points


def mark_annotators_frompool(rep_points):
    
    marked_ann = np.ones(np.array(rep_points).shape)
    for point in range(len(rep_points)):
        for annotator in range(len(rep_points[0])):
            if rep_points[point][annotator] == -121313:   #Arbitrary mark (-121313)
                marked_ann[point][annotator] = 0
       
    return marked_ann
                
                
#The following function igets Tokio's data:
def read_Tokio_1D(number_of_points):
    
    if number_of_points > 10000:
        raise NameError("Number of points in Tokio's dataset should be less or equal than 10000")
    
    db = pd.read_csv("https://geographicdata.science/book/_downloads/7fb86b605af15b3c9cbd9bfcbead23e9/tokyo_clean.csv")
    crs='epsg:4326'
    geometry=[Point(xy) for xy in zip(db["longitude"], db["latitude"])]
    gdf=gpd.GeoDataFrame(db,crs=crs, geometry=geometry)
    kde = KernelDensity(bandwidth=0.001).fit(gdf[["longitude","latitude"]])
    gdfs = gpd.GeoDataFrame(crs=crs, geometry=[Point(xy) for xy in kde.sample(10000)])[0:number_of_points]
    
    return np.array([x.coords[:][0][0] for x in gdfs.geometry.values])

                
                
                
                
    
