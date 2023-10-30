import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
import contextily as cx
from sklearn.neighbors import KernelDensity

from scipy.stats import norm
import cmdstanpy as cmd
import seaborn as sn

from crowdgeoloc.module import *

#The very same functions that we had but now in 1D:

def compute_long_lat_sigmas_1D(gdfs, long_error):
    
    minx, miny, maxx, maxy = tuple(gdfs.total_bounds)
    
    long_sigma = (maxx-minx)*long_error
    
    return long_sigma 


def single_annotator_rep_1D(gdfs, long_error):
    
    coords = points_to_array(gdfs)
    rep_coords = []
    
    long_sigma = compute_long_lat_sigmas_1D(gdfs, long_error)
    
    for coord in coords:
        long_rep = coord[0] + np.random.normal(loc=0, scale=long_sigma)
        
        rep_coords.append(long_rep)
    
    return np.array(rep_coords)


def multi_annotators_rep_1D(gdfs, num_annotators, annotators_errors):
    
    total_rep_coords = []
    
    for i in range(num_annotators):
        rep_coords = single_annotator_rep_1D(gdfs, annotators_errors[i])
        total_rep_coords.append(rep_coords)
        
    return np.array(total_rep_coords)


def smart_mean_consensus_1D(multi_reported_points, accuracy_weight):

    smart_consensus_aux = np.zeros(multi_reported_points.shape)
    for annotator in range(len(multi_reported_points)):
        smart_consensus_aux[annotator] = multi_reported_points[annotator]*accuracy_weight[annotator]
       
    smart_consensus_points = np.zeros(multi_reported_points.shape)
    for i in range(len(multi_reported_points)):
        smart_consensus_points += smart_consensus_aux[i]
    
    return smart_consensus_points



def annotator_sigma_1D(consensus_points, reported_points):
    
    difference = consensus_points - reported_points
    n = len(reported_points)

    squared_long_dif = []

    for i in range(n):
        squared_long_dif.append((np.mean(difference)-difference[i])**2)

    
    rec_long_sigma = np.sqrt(sum(np.array(squared_long_dif))/(n-1))

    return rec_long_sigma


def annotators_errors_1D(consensus_points, reported_points):
    
    rec_long_sigma = []
    for annotator in range(len(reported_points)):
        rec_long_sigma.append(annotator_sigma_1D(consensus_points[annotator], reported_points[annotator]))
    
    maxx = max(consensus_points[0])
    minx = min(consensus_points[0])

    rec_long_error = rec_long_sigma/(maxx-minx)

    return rec_long_error, rec_long_sigma



def best_accuracy_weight_1D(annotators_sigmas):
    
    #weights = 1/(np.array(annotators_sigmas)**2)/sum(1/(np.array(annotators_sigmas)**2))  

    numerator = []
    
    for i in range(len(annotators_sigmas)):
        numerator_aux = 1
        
        for j in range(len(annotators_sigmas)):
            if i != j:
                numerator_aux *= annotators_sigmas[j]**2

        numerator.append(numerator_aux)

    denominator = sum(numerator)

    weights = np.array(numerator)/denominator
    
    return weights



def ndarray_to_stanarray(reported_points):
    """
    This function lets us reshape the array containing all reported points from different annotators with 
        dimensions (num_annotators, points) to an array used in stan with dimensions (points, num_annotators).
    """
    
    rep_points_stanarray = []
    for i in range(len(reported_points[0])):
        rep_points_stanarray.append(reported_points[:,i])
    
    return rep_points_stanarray


def spatial_annotator_rep_1D(gdfs, annotator_errors, map_parts):
    """
    A single annotator reports different points depending on their position.
    """
    
    coords = points_to_array(gdfs)
    sigmas = []
    rep_coords = []
    
    minx, miny, maxx, maxy = tuple(gdfs.total_bounds)
    partitioned_map = [minx]
    for i in range(map_parts):
        partitioned_map.append(partitioned_map[-1] + (maxx-minx)/map_parts)
    
    
    for error in range(len(annotator_errors)):
        sigmas.append((maxx-minx)*annotator_errors[error])
    print(sigmas)
    for coord in coords:
        for part in range(len(partitioned_map)-1):
            if coord[0] >= partitioned_map[part] and coord[0] <= partitioned_map[part + 1]:
                rep_coords.append(coord[0] + np.random.normal(loc=0, scale=sigmas[part]))
                break
    
    return rep_coords, partitioned_map


def spatial_multi_annotators_rep_1D(gdfs, num_annotators, annotators_errors, map_parts):
    """
    Different annotators report various points depending on their position.
    """
    
    total_rep_coords = []
    
    for i in range(num_annotators):
        rep_coords, map_partitions = spatial_annotator_rep_1D(gdfs, annotators_errors[i], map_parts)
        total_rep_coords.append(rep_coords)
        
    return np.array(total_rep_coords), map_partitions
