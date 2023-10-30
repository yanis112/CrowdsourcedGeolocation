import random

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from scipy.stats import beta
from scipy.interpolate import CubicSpline


def sample_tokyo_latitudes(n, normalized=False):
    db = pd.read_csv("https://geographicdata.science/book/_downloads/7fb86b605af15b3c9cbd9bfcbead23e9/tokyo_clean.csv")
    crs = 'epsg:4326'
    geometry = [Point(xy) for xy in zip(db["longitude"], db["latitude"])]
    gdf = gpd.GeoDataFrame(db, crs=crs, geometry=geometry)
    kde = KernelDensity(bandwidth=0.001).fit(gdf[["longitude", "latitude"]])  # entenc que fiteja els punts a una dist.
    points = kde.sample(n)
    one_d_points = points[:, 0]
    if normalized:
        one_d_points -= np.min(one_d_points)
        one_d_points /= np.max(one_d_points)
    return one_d_points


class Annotator:
    def annotate(self, points: np.ndarray):
        pass

    def sigma(self, x: np.ndarray):
        pass


class SimpleNormalAnnotator(Annotator):
    def __init__(self, sigma):
        self._sigma = sigma

    def annotate(self, points: np.ndarray):
        return points + np.random.normal(loc=np.zeros(len(points)), scale=np.ones(len(points))*self._sigma)

    def sigma(self, x):
        return self._sigma


class FunctionNormalAnnotator(Annotator):
    def __init__(self, f):
        self.f = f

    def annotate(self, points):
        sigmas = self.sigma(points)
        return points + np.random.normal(loc=np.zeros(len(points)), scale=sigmas)

    def sigma(self, x: np.ndarray):
        return self.f(x)

class SimpleNormalAnnotatorPopulation:
    def __init__(self, sigma_distribution=None):
        if sigma_distribution is None:
            sigma_distribution = beta(1, 10)
        self.sigma_distribution = sigma_distribution

    def sample(self, n_annotators):
        annotators = np.array([SimpleNormalAnnotator(sigma) for sigma in self.sigma_distribution.rvs(n_annotators)])
        return AnnotatorSet(annotators)


def cubic_spline_generator(splits=11, quality_diversity=0.5, max_height=0.5, min_height=0.01):
    x = np.arange(splits) / (splits - 1.)
    not_fulfilling_min_height = True
    while not_fulfilling_min_height:
        max_height = (np.random.uniform() ** quality_diversity) * 0.9 * max_height + 0.1 * max_height
        y = np.random.uniform(size=splits) * (max_height-min_height) + 2 * min_height
        cs = CubicSpline(x, y)
        roots = cs.derivative().roots()
        # print("roots=", roots)
        min_val = np.min(cs(roots))
        not_fulfilling_min_height = (min_val < min_height)
    return cs


class FunctionNormalAnnotatorPopulation:
    def __init__(self, function_distribution=None):
        if function_distribution is None:
            function_distribution = cubic_spline_generator
        self.function_distribution = function_distribution

    def sample(self, n_annotators):
        annotators = np.array([FunctionNormalAnnotator(self.function_distribution()) for i in range(n_annotators)])
        return AnnotatorSet(annotators)


def random_assignment(n_points, n_annotators, k):
    assert k <= n_annotators, "The redundancy should be smaller than  the number of annotators"
    t_A = np.zeros(n_points * k, dtype=int)
    w_A = np.zeros(n_points * k, dtype=int)
    selected_annotators_indexes = np.argsort(np.random.rand(n_points, n_annotators), axis=1)[:, :k]
    total_annotations = 0
    for j in range(n_annotators):
        j_point_indices = np.argwhere(selected_annotators_indexes == j)[:, 0]
        end_annotations = total_annotations + len(j_point_indices)
        t_A[total_annotations:end_annotations] = j_point_indices
        w_A[total_annotations:end_annotations] = j
        total_annotations = end_annotations
    return t_A, w_A

class AnnotatorSet:
    def __init__(self, annotators: np.ndarray):
        self.annotators = annotators

    @property
    def n_annotators(self):
        return len(self.annotators)

    def annotate(self, ann_index, points):
        self.annotators[ann_index].annotate(points)

    def sample(self, k):
        selected_annotators_indexes = np.random.choice(len(self.annotators), size=k, replace=False)
        return self.annotators[selected_annotators_indexes]

    def random_annotation(self, k, points):
        t_A, w_A = random_assignment(len(points), self.n_annotators, k)
        annotations = self.batch_annotation(t_A, w_A, points)
        return t_A, w_A, annotations

    def batch_annotation(self, t_A, w_A, points):
        n_annotations = len(t_A)
        annotations_per_individual = 0
        annotations = np.zeros(n_annotations, dtype=float)
        for j, ann in enumerate(self.annotators):
            j_point_indices = np.argwhere(w_A == j)[:, 0]
            j_points = points[t_A[j_point_indices]]
            ann_points = ann.annotate(j_points)
            annotations[j_point_indices] = ann_points
        return annotations

# Code below should be moved to another module









class RandomStrategy:
    def location_curve(self, annotator_set, points, max_annotator_pool):
        result = np.zeros((len(points), max_annotator_pool))
        for i, p in enumerate(points):
            # print("p=",p)
            pool = annotator_set.sample(max_annotator_pool)
            pos = None
            for k, annotator in enumerate(pool):
                new_pos_reported = annotator.annotate(np.array([p]))[0]
                # print(new_pos_reported)
                if pos is None:
                    pos = new_pos_reported
                else:
                    pos = (new_pos_reported + pos * k) / (k + 1)
                result[i, k] = pos
        return result


def sigma_average(positions, sigmas):
    num = 0
    denom = 0
    for i in range(len(positions)):
        num += positions[i] * (sigmas[i] ** -2)
        denom += (sigmas[i] ** -2)
    return num / denom


class RandomStrategyIntAverage:

    def location_curve(self, annotator_set, points, max_annotator_pool):
        result = np.zeros((len(points), max_annotator_pool))
        for i, p in enumerate(points):
            # print("p=",p)
            pool = annotator_set.sample(max_annotator_pool)
            positions = []
            sigmas = []
            for k, annotator in enumerate(pool):
                positions.append(annotator.annotate(np.array([p]))[0])
                sigmas.append(annotator.sigma(positions[-1]))
                result[i, k] = sigma_average(positions, sigmas)
        return result


class GreedyStrategyIntAverage:

    def location_curve(self, annotator_set, points, max_annotator_pool):
        result = np.zeros((len(points), max_annotator_pool))
        for i, p in enumerate(points):
            # print("p=",p)
            annotators = annotator_set.annotators.copy()
            sigmas = np.array([ann._sigma for ann in annotators])
            best_annotators_indexes = np.argsort(sigmas)
            positions = []
            sigmas = []
            for k in range(max_annotator_pool):
                annotator = annotators[best_annotators_indexes[k]]
                positions.append(annotator.annotate(np.array([p]))[0])
                sigmas.append(annotator.sigma(positions[-1]))
                result[i, k] = sigma_average(positions, sigmas)
        return result


class MultiSegmentNormalAnnotator:
    def __init__(self, splits, sigmas):
        self.splits = splits
        self.sigmas = sigmas

    def annotate(self, points: np.ndarray):
        partitions = np.digitize(points, self.splits) - 1
        # print(partitions)
        # print(np.max(partitions))
        return points + np.random.normal(loc=0, scale=self.sigmas[partitions])


def generate_annotators(splits):
    num_splits = len(splits) - 1
    sigma_options = np.linspace(0.001, 0.2, 10)
    # print(sigma_options)
    num_annotators = 4
    annotators = []
    for i in range(num_annotators):
        sigmas = np.random.choice(sigma_options, num_splits)
        # print(sigmas)
        a = MultiSegmentNormalAnnotator(splits, sigmas)
        annotators.append(a)
    return annotators


def annotate(points, annotators):
    n = len(points)
    a = len(annotators)
    annotated_points = np.zeros((a, n))
    for i, annotator in enumerate(annotators):
        annotated_points[i] = annotator.annotate(points)
    return annotated_points
