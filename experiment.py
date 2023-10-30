import random
from scipy.stats import uniform
from dataclasses import dataclass
import numpy as np
from .one_d import AnnotatorSet, random_assignment, SimpleNormalAnnotatorPopulation, sample_tokyo_latitudes,FunctionNormalAnnotatorPopulation

@dataclass
class CGPInstance:
    t: int          # Number of points to geolocate
    w: int          # Number of workers
    a: int          # Number of annotations
    t_A: np.ndarray    # Task for each annotation
    w_A: np.ndarray    # Worker for each annotation
    ann: np.ndarray    # Value for each annotation

class TooManyAnnotationsPerIndividual(Exception):
    pass

class TooManyAnnotations(Exception):
    pass

class ActiveAnnotationContest:
    def __init__(self, points: np.ndarray, ans: AnnotatorSet,
                 max_total_annotations: int = 0,
                 max_annotations_per_individual: int = 1e200):
        self.points = points
        self.annotator_set = ans
        self.max_total_annotations = max_total_annotations
        self.max_annotations_per_individual = max_annotations_per_individual
        self.reset()

    @property
    def n_points(self):
        return len(self.points)

    @property
    def n_annotators(self):
        return self.annotator_set.n_annotators

    def reset(self):
        self.annotations_per_individual = np.zeros(self.annotator_set.n_annotators)

    def batch_request(self, t_A, w_A):
        u, u_counts = np.unique(w_A, return_counts=True)
        self.annotations_per_individual[u] += u_counts
        if np.max(self.annotations_per_individual) > self.max_annotations_per_individual:
            self.annotations_per_individual[u] -= u_counts
            raise TooManyAnnotationsPerIndividual()
        if np.sum(self.annotations_per_individual) > self.max_total_annotations:
            print(np.sum(self.annotations_per_individual))
            self.annotations_per_individual[u] -= u_counts
            raise TooManyAnnotations(np.sum(self.annotations_per_individual))
        ann = self.annotator_set.batch_annotation(t_A, w_A, self.points)
        return ann

    def request(self, t: int, w: int):
        if self.annotations_per_individual[w] >= self.max_annotations_per_individual:
            raise TooManyAnnotationsPerIndividual()
        if np.sum(self.annotations_per_individual) >= self.max_total_annotations:
            raise TooManyAnnotations()
        self.annotations_per_individual[w] += 1
        return self.annotator_set[w].annotate(self.points[t:t+1])


class ActiveAnnotationMethod:
    def run(self, exp: ActiveAnnotationContest):
        return None #points


def max_redundancy(exp):
    max_total_annotations = min(exp.max_annotations_per_individual * exp.n_points, exp.max_total_annotations)
    k = max_total_annotations // exp.n_points
    return k

def random_annotation(exp, batch_start=0, batch_size=None, k=None):
    if batch_size is None:
        batch_size = exp.n_points
    if k is None:
        k = max_redundancy(exp)
    t, w = random_assignment(batch_size, exp.n_annotators, k)
    t += batch_start
    ann = exp.batch_request(t, w)
    return t, w, ann


def softmax_stable(x):
    return (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


# Add normalization of the weights
def normalize(tab):
    norm=[]
    norm=[(k-np.min(tab))/(np.max(tab)-np.min(tab)) for k in tab]
    return(np.array(norm))




# +

def sigma_assignment(batch_size, sigmas, k, greediness, previous_anns=None, batch_start=0):

    n_annotators = len(sigmas)

    if previous_anns is not None:
        for i in np.sort(previous_anns)[::-1]:
            sigmas = np.delete(sigmas, i)

    weights = (sigmas ** (-2))
    sum_weights = np.sum(weights)
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    if greediness < 1.00000001:
        greediness_factor = 0
    else:
        greediness_factor = np.log(greediness) / (max_weight - min_weight)
    #print(greediness_factor)
    p = softmax_stable(greediness_factor * weights)

    if previous_anns is not None:
        for i in np.sort(previous_anns):
            p = np.insert(p, i, 0)

    #print("probabilities: ",p)
    # print("p=", p, np.sum(p))
    # pm = softmax_stable(-greedyness * (sigmas ** (-2)))
    # print("pm=", pm, np.sum(pm))
    t_A = np.zeros(batch_size * k, dtype=int)
    w_A = np.zeros(batch_size * k, dtype=int)
    selected_annotators_indexes = np.argsort(p[None, :] * np.random.rand(batch_size, n_annotators), axis=1)[:, -k:]
    # print("sel = ", selected_annotators_indexes)
    total_annotations = 0
    for j in range(n_annotators):
        j_point_indices = np.argwhere(selected_annotators_indexes == j)[:, 0]
        end_annotations = total_annotations + len(j_point_indices)
        t_A[total_annotations:end_annotations] = j_point_indices
        w_A[total_annotations:end_annotations] = j
        total_annotations = end_annotations
    # print("w=", w_A)
    return t_A+batch_start, w_A




# +

def sigma_annotation(exp, sigmas, previous_anns=None, greediness=1., batch_start=0, batch_size=None, k=None):
    if batch_size is None:
        batch_size = exp.n_points
    if k is None:
        k = max_redundancy(exp)
    t, w = sigma_assignment(batch_size, sigmas, k, greediness, previous_anns, batch_start=batch_start)
    ann = exp.batch_request(t, w)
    return t, w, ann


# -

def mse(points, predictions):
    return np.average(np.square(predictions-points))

def mean_norm_error(points, predictions, ord=None):
     return (np.linalg.norm(predictions-points, ord=ord) / len(predictions))
#
def mean_location_norm_error(exp, predictions, ord=1):
    #print(predictions["locations"] - exp.points)
    return (np.linalg.norm(predictions["locations"] - exp.points, ord=ord) / len(exp.points))
#
# def mean_sigma_norm_error(exp, predictions, ord=None):
#     sigmas = np.array([a.sigma(0) for a in exp.annotator_set.annotators])
#     return (np.linalg.norm(predictions["sigmas"] - sigmas, ord=ord) / len(predictions))
#def mean_location_norm_error(exp, predictions, ord=None):
#    return (np.linalg.norm(np.log(np.abs(predictions["locations"] - exp.points)), ord=ord) / len(predictions))

def mean_sigma_norm_error(exp, predictions, ord=None):
    sigmas = np.array([a.sigma(0) for a in exp.annotator_set.annotators])
    return np.linalg.norm(np.log(predictions["sigmas"]/sigmas), ord=ord) / len(sigmas)


# saving experiments setup

import pickle

#tokyo latitude sampling
def tok(n):
    t = np.array(sample_tokyo_latitudes(n)) #we sample the n points
    tok_norm=(t-np.min(t))/(np.max(t)-np.min(t)) #we normalize
    return(tok_norm)

def save_experiment_setup(params):
    # Create a dictionary containing experiment data

    np.random.seed(1234)  # we set a seed to generate each time the same points/sigmas
    random.seed(1234)
    n_points, n_annotators, redundancy = (params[0], params[1], params[2])  # choice of general parameters
    sig_distr = params[3]  # choice of the sigma distrib
    if sig_distr == 'uniform':
        annotator_population = SimpleNormalAnnotatorPopulation(
            uniform(scale=0.1))  # Which value there ?? 0.1 at the beginning
    if sig_distr == 'beta':
        annotator_population = SimpleNormalAnnotatorPopulation()

    point_distr = params[4]  # choice of point distrib

    if point_distr == 'uniform':
        point_distribution = uniform()
        points = point_distribution.rvs(n_points)
    else:
        points = tok(n_points)

    list_tru_sig = []
    ann_set = annotator_population.sample(n_annotators)
    list_true_sig = [ann_set.annotators[k]._sigma for k in range(len(ann_set.annotators))]
    # print(list_true_sig)
    experiment_data = {
        "nb_points": params[0],
        "nb_annotators": params[1],
        "redundancy": params[2],
        "sigma_distrib": params[3],
        "point_distrib": params[4],
        "points": points,
        "sigmas": list_true_sig,
        "random_seed": np.random.randint(0, 10000)
    }

    filename = f"np_{params[0]}_na_{params[1]}_rd_{params[2]}_sd_{params[3]}_pd_{params[4]}_setup.pkl"
    # Save the experiment data to a file using pickle
    with open(filename, "wb") as f:
        pickle.dump(experiment_data, f, protocol=5)


def save_experiment_setup_2(params):
    # Create a dictionary containing experiment data

    np.random.seed(1234)  # we set a seed to generate each time the same points/sigmas
    random.seed(1234)
    n_points, n_annotators, redundancy = (params[0], params[1], params[2])  # choice of general parameters
    sig_distr = params[3]  # choice of the sigma distrib
    if sig_distr == 'uniform':
        annotator_population = FunctionNormalAnnotatorPopulation()
              # Which value there ?? 0.1 at the beginning
    if sig_distr == 'beta':
        annotator_population = FunctionNormalAnnotatorPopulation()

    point_distr = params[4]  # choice of point distrib

    if point_distr == 'uniform':
        point_distribution = uniform()
        points = point_distribution.rvs(n_points)
    else:
        points = tok(n_points)

    list_tru_sig = []
    allx = np.arange(1000) / 1000.
    ann_set = annotator_population.sample(n_annotators)
    list_true_sig = [ann_set.annotators[k].sigma(allx) for k in range(len(ann_set.annotators))]
    # print(list_true_sig)
    experiment_data = {
        "nb_points": params[0],
        "nb_annotators": params[1],
        "redundancy": params[2],
        "sigma_distrib": params[3],
        "point_distrib": params[4],
        "points": points,
        "sigmas": list_true_sig,
        "random_seed": np.random.randint(0, 10000)
    }

    filename = f"np_{params[0]}_na_{params[1]}_rd_{params[2]}_sd_{params[3]}_pd_{params[4]}_setup_not_constant.pkl"
    # Save the experiment data to a file using pickle
    with open(filename, "wb") as f:
        pickle.dump(experiment_data, f, protocol=5)



# saving experiments setup (only the setup, not te results)

import pickle


def save_experiment(list_param, results):
    # Create a dictionary containing experiment data
    experiment_data = {
        "nb_points": list_param[0],
        "nb_annotators": list_param[1],
        "redundancy": list_param[2],
        "sigma_distrib": list_param[3],
        "point_distrib": list_param[4],
        "points": results[0],
        "sigmas": results[1],
        "method_results": {
            "name": list_param[5],
            "points": results[2][0],
            "sigmas": results[3][0]
        }
    }

    # ,
    # "conservative2": {
    #    "points":results[2][1] ,
    #    "sigmas": results[3][1]
    # },
    # "10shot": {
    #    "points": results[2][2],
    #    "sigmas": results[3][2]
    # }
    # }
    #  }

    filename = f"np_{list_param[0]}_na_{list_param[1]}_rd_{list_param[2]}_sd_{list_param[3]}_pd_{list_param[4]}.pkl"
    # Save the experiment data to a file using pickle
    with open(filename, "wb") as f:
        pickle.dump(experiment_data, f, protocol=5)

def load_experiment(path):
    # Load the pickled file
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return(data)

# load an experiment for a pickle file, return list of params
def load_experiment_setup(path):
    # Load the pickled file
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return([data["nb_points"],data["nb_annotators"],data["redundancy"],data["sigma_distrib"],data["point_distrib"],data["points"],data["sigmas"],data["random_seed"]])
