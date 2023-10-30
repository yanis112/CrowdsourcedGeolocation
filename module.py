import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
import contextily as cx
from sklearn.neighbors import KernelDensity
from numpy import asarray


def show_map(gdf):
    ax = gdf.plot(figsize=(10, 10), alpha=0.3, edgecolor='k')
    cx.add_basemap(ax, crs=gdf.crs)

    
def sample(kde, crs, size=10000):
    s = kde.sample(size)
    #print(s.shape)
    #print(s[0][:])
    geometry=[Point(xy) for xy in s]
    return gpd.GeoDataFrame(crs=crs, geometry=geometry)


def points_to_array(gdfs):
    """  
    Converts a shapely.Points darray to a numerical darray
    
    Args:
        gdfs: geodataframe.
    
    Returns:
        An array with the numerical coordinates of the gdfs' points.
    """
    return np.array([x.coords[:][0] for x in gdfs.geometry.values])


def array_to_points(array):
    """  
    Converts a numerical darray into a shapely.Points darray.
    
    Args:
        array: array with numerical coordinates.
        
    Returns:
        points: an array with shapely.Points corresponding to the given numerical coordinates. 
    """
    points = []
    
    for coord in array:
        points.append(Point(coord))
        
    return points  

#Nota: voisin ehkä käyttää coord_array ja sit gdfs ei tarvitaan
def compute_long_lat_sigmas(gdfs, long_error, lat_error):
    """
    Computes a longitude- and a latitude-sigma depending on the map's dimensions and annotators' errors. Later on these sigmas are taken into account to add noise and get reported points.
    
    Args:
        gdfs (geodataframe): geodataframe
        long_error (double): double representing the annotator's longitude-error 
        lat_error (double): double representing the annotator's latitude-error 
    
    Returns:
        tuple: (double, double)(long_error: annotator's longitude-error, lat_error: annotator's latitude-error)
    """
    
    minx, miny, maxx, maxy = tuple(gdfs.total_bounds)
    
    long_sigma = (maxx-minx)*long_error
    lat_sigma = (maxy-miny)*lat_error
    
    return long_sigma, lat_sigma 


#Nota: voisin ehkä käyttää coord_array ja sit gdfs ei tarvitaan
#Nota: ehkä voisin käyttää vaan multi_annotators_rep
def single_annotator_rep(gdfs, long_error, lat_error):
    """
    Returns the points reported by an annotator depending on its errors.
    
    Args:
        gdfs (geodataframe): geodataframe
        long_error (double): double representing the annotator's longitude-error 
        lat_error (double): double representing the annotator's latitude-error 
    
    Returns:
        rep_coords (darray): a darray of dimensions (reported points, 2) containing all reported points by the annotator
    """
    coords = points_to_array(gdfs)
    rep_coords = []
    
    long_sigma, lat_sigma = compute_long_lat_sigmas(gdfs, long_error, lat_error)
    
    for coord in coords:
        long_rep = coord[0] + np.random.normal(loc=0, scale=long_sigma)
        lat_rep = coord[1] + np.random.normal(loc=0, scale=lat_sigma)
        
        rep_coords.append([long_rep, lat_rep])
    
    return np.array(rep_coords)

def annotator_sigmas(consensus_points, reported_points):
    """
    Computes the longitude- and latitude-errors made by the annotator when reporting the points compairing them to the consensual points.
    
    Args:
        consensus_points (darray): a darray of dimensions (reported points, 2) containing the consensual points' coordinates.
        reported_points (darray): a darray of dimensions (reported points, 2) representing reported points by a single annotator.
    
    Returns:
        tuple (double, double): (rec_long_error: recuperated annotator's longitude error, rec_lat_error: recuperated annotator's latitude error).
    
    """
    difference = consensus_points - reported_points
    n = len(reported_points)
    
    long_dif = []
    lat_dif = []
    for i in range(n):
        long_dif.append(difference[i][0])
        lat_dif.append(difference[i][1])
    
    squared_long_dif = []
    squared_lat_dif = []
    for i in range(n):
        squared_long_dif.append((np.mean(long_dif)-long_dif[i])**2)
        squared_lat_dif.append((np.mean(lat_dif)-lat_dif[i])**2)
    
    rec_long_sigma = np.sqrt(sum(np.array(squared_long_dif))/(n-1))
    rec_lat_sigma = np.sqrt(sum(np.array(squared_lat_dif))/(n-1))
    
    return rec_long_sigma, rec_lat_sigma


def annotator_error(consensus_points, reported_points):
    """
    Computes the longitude- and latitude-errors made by the annotator when reporting the points compairing them to the consensual points.
    
    Args:
        consensus_points (darray): a darray of dimensions (reported points, 2) containing the consensual points' coordinates.
        reported_points (darray): a darray of dimensions (reported points, 2) representing reported points by a single annotator.
    
    Returns:
        tuple (double, double): (rec_long_error: recuperated annotator's longitude error, rec_lat_error: recuperated annotator's latitude error).
    
    """
    rec_long_sigma, rec_lat_sigma = annotator_sigmas(consensus_points, reported_points)
    
    long_boundaries = []
    lat_boundaries = []
    for i in range(len(consensus_points)):
        long_boundaries.append(consensus_points[i][0])
        lat_boundaries.append(consensus_points[i][1])
    maxx = max(long_boundaries)
    minx = min(long_boundaries)
    maxy= max(lat_boundaries)
    miny = min(lat_boundaries)
    
    rec_long_error = rec_long_sigma/(maxx-minx)
    rec_lat_error = rec_lat_sigma/(maxy-miny)
    
    return rec_long_error, rec_lat_error


def multi_annotators_rep(gdfs, num_annotators, annotators_errors):
    """
    Returns the points reported by all annotators depending on their errors.
    
    Args:
        gdfs (geodataframe): geodataframe
        num_annotators (int): number of annotators.
        annotators_errors (darray): darray of dimensions (number of annotators, 2) containing annotators' longitude- and latitude-errors.
    
    Returns:
        rep_coords (darray): a darray of dimensions (reported points, 2) containing all reported points by the annotator
    
    
    """
    total_rep_coords = []
    
    for i in range(num_annotators):
        rep_coords = single_annotator_rep(gdfs, annotators_errors[i][0], annotators_errors[i][1])
        total_rep_coords.append(rep_coords)
        
    return np.array(total_rep_coords)


def define_annotators(manual=False, num_annotators="undefined", annotators_errors="undefined"):
    """
    Function to define annotators' characteristics manually or by arguments.
    
    Args:
        manual (bool): If true, one can define all annotators' characteristics by hand through different inputs. If False, the function needs the following two arguments. False by default.
        num_annotators (int): number of annotators. By default "undefined".
        annotators_errors (darray): darray of dimensions (num_annotators, 2) containing the longitude- and latitude- errors for all annotators).
        
    
    Returns:
        tuple (int, darray): (num_annotators: number of annotators, annotators_errors: darray of dimensions (num_annotators, 2) containing the longitude- and latitude- errors for all annotators).
    """
    if manual == True:
        
        num_annotators = int(input("Select the number of annotators: "))
        annotators_errors = []

        for i in range(num_annotators):
            long_error = float(input("Long error of the annotator: "))
            lat_error = float(input("Lat error of the annotator: "))     
            annotators_errors.append([long_error, lat_error])
    
    return num_annotators, annotators_errors
    
    
def mean_consensus(multi_reported_points):
    """
    Computes the mean of all reported points.
    
    Args:
        multi_reported_points (darray): a darray of dimensions (nummber of annotators, points to report, 2) that represents all the reported points for each annotator.
        
    Returns:
        consensus_points (darray): a darray of dimensions (reported points, 2) with the consensual coordinates from the reported points (i.e. the mean between all reported points).
    """
    consensus_points = np.zeros(multi_reported_points.shape)
    for annotator in range(len(multi_reported_points)):
        consensus_points += multi_reported_points[annotator]
    
    consensus_points /= len(multi_reported_points)
    
    #return consensus_points[0]
    return consensus_points


def accuracy_weight(annotators_errors):
    """
    Computes a weight for each annotator depending on their error to later on get a better consensus.
    
    Args:
        annotators_errors (darray): a darray of dimensions (number of annotators, 2) containing the longitude- and latitude-errors of each annotator.
        
    Returns:
        tuple: (array, array)(longitude weight for each annotator, latitude weight for each annotator).
    """
    long_error_weight = []
    lat_error_weight = []
    for i in range(len(annotators_errors)):
        long_error_weight.append(annotators_errors[i][0])
        lat_error_weight.append(annotators_errors[i][1])
    
    long_proportions = []
    lat_proportions = []
    for j in range(len(long_error_weight)):
        long_proportions.append(long_error_weight[j]/sum(long_error_weight))
        lat_proportions.append(lat_error_weight[j]/sum(lat_error_weight))
        
    long_normalized_proportions = []
    lat_normalized_proportions = []
    for k in range(len(long_proportions)):
        long_normalized_proportions.append(max(long_proportions)/long_proportions[k])
        lat_normalized_proportions.append(max(lat_proportions)/lat_proportions[k])
        
    long_weighted_accuracy = []
    lat_weighted_accuracy = []
    for l in range(len(long_normalized_proportions)):
        long_weighted_accuracy.append(long_normalized_proportions[l]/sum(long_normalized_proportions))
        lat_weighted_accuracy.append(lat_normalized_proportions[l]/sum(lat_normalized_proportions))
        
   
    return long_weighted_accuracy, lat_weighted_accuracy


def best_accuracy_weight(annotators_sigmas):
    """
    Computes the optimal weight for each annotator depending on their error to later on get a better consensus.
    
    Args:
        annotators_errors (darray): a darray of dimensions (number of annotators, 2) containing the longitude- and latitude-errors of each annotator.
        
    Returns:
        tuple: (array, array)(longitude optimal weight for each annotator, latitude optimal weight for each annotator).
    """
    
    long_weights = []
    lat_weights = []
    
    total_long_sigmas_sq = []
    total_lat_sigmas_sq = []
    
    for i in range(len(annotators_sigmas)):
        long_sigmas_sq = 1
        lat_sigmas_sq = 1
        
        for j in range(len(annotators_sigmas)):
            if i != j:
                long_sigmas_sq *= annotators_sigmas[j][0]**2
                lat_sigmas_sq *= annotators_sigmas[j][1]**2
        
        long_weights.append(long_sigmas_sq)
        lat_weights.append(lat_sigmas_sq)
        
        total_long_sigmas_sq.append(long_sigmas_sq)
        total_lat_sigmas_sq.append(lat_sigmas_sq)

    return np.array(long_weights)/sum(total_long_sigmas_sq), np.array(lat_weights)/sum(total_lat_sigmas_sq)


def smart_mean_consensus(multi_reported_points, accuracy_weight):
    """
    Computes a weighted mean.
    
    Args:
        multi_reported_points (darray): a darray of dimensions (nummber of annotators, points to report, 2) that represents all the reported points for each annotator.
        accuracy_weight (darray): a darray of dimensions (2, number of annotators) that contains a certain longitude weight and a certain latitude weight for each annotator.
        
    Returns:
        smart_consensus_points (darray): a darray of dimensions (reported points, 2) with the consensual coordinates from the reported points.
    """
    smart_consensus_aux = np.zeros(multi_reported_points.shape)
    for annotator in range(len(multi_reported_points)):
        smart_consensus_aux[annotator, :, 0] = multi_reported_points[annotator, :, 0]*accuracy_weight[0][annotator]
        smart_consensus_aux[annotator, :, 1] = multi_reported_points[annotator, :, 1]*accuracy_weight[1][annotator]
       
    smart_consensus_points = np.zeros(multi_reported_points.shape)
    for i in range(len(multi_reported_points)):
        smart_consensus_points += smart_consensus_aux[i]
    
    return smart_consensus_points



def rec_errors_plotting(iterations, num_annotators, annotators_errors, rec_annotators_errors_plot):
    """
    Shows a plot for each annotator with its recuperated longitude- and latitude-errors for each iteration and its real errors.
    
    Args:
        iterations (int): number of iterations.
        num_annotators (int): number of annotators.
        annotators_errors (array of dimensions (1, num_annotators, 2)): longitude- and latitude-errors given for each annotator.
        rec_annotators_errors_plot (array of dimensions (iterations, num_annotators, 2)): recuperated annotators' errors for each
                                                                                         iteration.
    
    """
    
    for annotator in range(num_annotators):
        long_plot = []
        lat_plot = []

        for i in range(iterations):
            long_plot.append(rec_annotators_errors_plot[i][annotator][0])
            lat_plot.append(rec_annotators_errors_plot[i][annotator][1])

        plt.step(np.linspace(1,iterations, iterations), long_plot, 'steelblue')
        plt.plot(np.linspace(1, iterations, iterations), np.linspace(annotators_errors[annotator][0],
                                                                     annotators_errors[annotator][0],iterations), 'lightskyblue')
        
        plt.step(np.linspace(1,iterations, iterations), lat_plot, 'firebrick')
        plt.plot(np.linspace(1, iterations, iterations), np.linspace(annotators_errors[annotator][1],
                                                                     annotators_errors[annotator][1],iterations), 'lightcoral')
        
        plt.show()
        
    return

