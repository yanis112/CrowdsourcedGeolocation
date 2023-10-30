import numpy as np
import cmdstanpy as cmd

from .experiment import ActiveAnnotationContest, ActiveAnnotationMethod, random_annotation, sigma_annotation
from .cmdstan import resource_filename

def point_average(t_A: np.ndarray, ann: np.ndarray):
    _ann = ann.copy()
    _t_A = t_A.copy()
    _ndx = np.argsort(_t_A)
    tasks, _pos, g_count = np.unique(_t_A[_ndx],
                                   return_index=True,
                                   return_counts=True)
    g_sum = np.add.reduceat(_ann[_ndx], _pos, axis=0)
    g_mean = g_sum / g_count
    return tasks, g_mean

def point_averages_and_sigmas(t, w, ann):
    # Compute means
    _ann = ann.copy()
    _t = t.copy()
    _ndx = np.argsort(_t)
    tasks, _pos, g_count = np.unique(_t[_ndx],
                                   return_index=True,
                                   return_counts=True)
    g_sum = np.add.reduceat(_ann[_ndx], _pos, axis=0)
    g_mean = g_sum / g_count

    # Compute square error of each annotation with respect to the mean of that point

    inv_tasks = np.ones(np.max(tasks)+1, dtype=int)*(-1)
    inv_tasks[tasks] = np.arange(len(tasks))
    sq_errors = (_ann - g_mean[inv_tasks[t]])**2

    # Compute the variances

    _ndxa = np.argsort(w)
    # print(_ndx)
    workers, _posa, g_counta = np.unique(w[_ndxa],
                                      return_index=True,
                                      return_counts=True)
    # print(_pos)
    # print(g_count)
    g_suma = np.add.reduceat(sq_errors[_ndxa], _posa, axis=0)
    g_meana = g_suma / (g_counta)

    return tasks, g_mean, workers, np.sqrt(g_meana)


def imeans_and_sigmas(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    weights = sigmas ** -2
    weights_per_ann = weights[w]
    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                             return_index=True)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    imeans = nums / denoms

    # Compute square error of each annotation with respect to the mean of that point

    inv_tasks = np.ones(np.max(tasks) + 1, dtype=int) * (-1)
    inv_tasks[tasks] = np.arange(len(tasks))
    sq_errors = (ann - imeans[inv_tasks[t]]) ** 2

    # Compute the variances

    ndx_w = np.argsort(w)
    # print(_ndx)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                                        return_index=True,
                                        return_counts=True)
    sum_w = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0)
    sample_variances = sum_w / (count_w)

    return tasks, imeans, workers, np.sqrt(sample_variances)

def make_conservative(weights, max_dif=50):
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    q = max_weight / min_weight
    if q > max_dif:
        #print("q:", q)
        a = np.log(weights) - np.log(min_weight)
        b = np.log(q)
        #print("Before:", weights)
        weights = np.exp((a/b)*np.log(max_dif))*min_weight
        #print("After:", weights)
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        q = max_weight / min_weight
        #print("q after:", q)
    return weights

class OverconfidenceException(Exception):
    pass

def imeans_and_sigmas2(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)
    
    weights = sigmas ** -2
    #print(weights)
    if (np.max(weights) / np.min(weights) > 1000): #MODIF HERE, 50 ORIGINAL VALUE, 
        raise OverconfidenceException()
    #weights = make_conservative(weights)
    weights_per_ann = weights[w]
    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                             return_index=True)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    imeans = nums / denoms

    # Compute square error of each annotation with respect to the mean of that point

    inv_tasks = np.ones(np.max(tasks) + 1, dtype=int) * (-1)
    inv_tasks[tasks] = np.arange(len(tasks))
    sq_errors = (ann - imeans[inv_tasks[t]]) ** 2

    # Compute the variances

    ndx_w = np.argsort(w)
    # print(_ndx)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                                        return_index=True,
                                        return_counts=True)
    sum_w = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0)
    sample_variances = sum_w / (count_w)

    return tasks, imeans, workers, np.sqrt(sample_variances)


def mean_averaging(T, t, w, ann):
    # Compute the means
    _id, mean = point_average(t, ann)
    v = np.ones(T) * 0.5
    v[_id] = mean
    return v

def direct_weights(T, W, t, w, ann):
    tasks, means, workers, st_devs = point_averages_and_sigmas(t, w, ann)
    v = np.ones(T) * 0.5
    v[tasks] = means
    sigmas = np.ones(W) * (-1)
    sigmas[workers] = st_devs
    sigmas[sigmas < 0] = np.max(sigmas)

    return sigmas


def conservative_means_and_sigmas(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    weights = sigmas ** -2
    weights_per_ann = weights[w]

    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                                return_index=True)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)

    nums_per_ann = nums[t] - (weights_per_ann*ann)
    denoms_per_ann = denoms[t] - weights_per_ann
    means_per_ann = nums_per_ann / denoms_per_ann
    sq_errors = (ann - means_per_ann) ** 2

    ndx_w = np.argsort(w)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                            return_index=True,
                            return_counts=True)
    sample_variances = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0) / (count_w - 1)
    sigmas = np.sqrt(sample_variances)
    #print(sigmas)
    return means_per_ann, workers, sigmas


def conservative_means_and_sigmas2(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    #print("sigmas:", sigmas)
    weights = sigmas ** -2
    #print("weights:", weights/np.sum(weights))
    if (np.max(weights) / np.min(weights) > 1000): #MODIF 50 AVANT
        raise OverconfidenceException()
    weights_per_ann = weights[w]
    best_worker = np.argmax(weights)
    best_worker_self_weight = np.partition(weights, -2)[-2] # Assign the second largest weight
    best_worker_annotations = (w == best_worker)

    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                                return_index=True)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)

    nums_per_ann = nums[t]
    denoms_per_ann = denoms[t]

    nums_per_ann[best_worker_annotations] -= (weights_per_ann[best_worker_annotations] * ann[best_worker_annotations])
    nums_per_ann[best_worker_annotations] += (best_worker_self_weight * ann[best_worker_annotations])
    denoms_per_ann[best_worker_annotations] -= weights_per_ann[best_worker_annotations]
    denoms_per_ann[best_worker_annotations] += best_worker_self_weight

    means_per_ann = nums_per_ann / denoms_per_ann
    sq_errors = (ann - means_per_ann) ** 2

    ndx_w = np.argsort(w)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                            return_index=True,
                            return_counts=True)
    sample_variances = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0) / (count_w)

    return means_per_ann, workers, np.sqrt(sample_variances)


def imean_averaging(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)
    weights = sigmas ** -2
    weights_per_ann = weights[w]
    #print(weights_per_ann[:5])
    #print(w[:5])
    _ndx = np.argsort(t)
    tasks, _pos = np.unique(t[_ndx],
                                return_index=True)
    denoms = np.add.reduceat(weights_per_ann[_ndx], _pos, axis=0)
    #print(denoms[:5])
    nums = np.add.reduceat((weights_per_ann * ann)[_ndx], _pos, axis=0)
    #print(nums[:5])
    return tasks, nums/denoms

# +
class OneShotMean(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        tasks, means = imean_averaging(t, w, ann)
        v = np.ones(T) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": np.ones(W)}
    
class OneShotDirect(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        sigmas = direct_weights(T, W, t, w, ann)
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(T) * 0.5
        v[tasks] = means
        return {"locations": v, "sigmas": sigmas}


# -

class OneShotIterative(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        sigmas = np.ones(W)
        means = np.zeros(T)
        eps = 1e-8 #ATTENTION MODIFI2 -4 avant
        difference = 1e99
        reached_overconfidence = False
        while (difference > eps) and not reached_overconfidence :
            try:
                #print("sigmas:",sigmas)
                old_sigmas = sigmas.copy()
                tasks, imeans, workers, partial_sigmas = imeans_and_sigmas2(t, w, ann, sigmas)
                sigmas[workers] = partial_sigmas
                means[tasks] = np.nan_to_num(imeans, nan=0.5)
                difference = np.sum(np.abs(old_sigmas-sigmas))
            except OverconfidenceException:
                reached_overconfidence = True
                #print("overconf")
        return {"locations": means, "sigmas": sigmas}


class OneShotBayesian(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        #sigmas = direct_weights(T, W, t, w, ann)

        n_annotators = exp.n_annotators
        n_points = exp.n_points

        d = {"w": n_annotators,
             "a": len(t),
             "t": n_points,
             "t_A": t + 1,
             "w_A": w + 1,
             "ann": ann
             }
        inits = {"sigmas": [np.ones(n_annotators)]}

        model = cmd.CmdStanModel(stan_file=resource_filename('normal.2.stan'))
        s = model.sample(data=d,inits=inits,show_console=False) 

        rec_sigmas_sample = s.stan_variable("sigmas")
        stan_sigmas = []
        for annotator in range(n_annotators):
            stan_sigmas.append(np.median(rec_sigmas_sample[:, annotator]))

        tasks, means = imean_averaging(t, w, ann, np.array(stan_sigmas))

        v = np.ones(np.max(t)+1) * 0.5
        v[tasks] = means

        rec_points_sample = s.stan_variable("mu")
        stan_points = []
        for p in range(n_points):
            stan_points.append(np.median(rec_points_sample[:, p]))

        return {"locations": v, "sigmas": stan_sigmas, "stan points": stan_points}


class OneShotConservative(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        sigmas = np.ones(W)
        means = np.zeros(T)
        eps = 1e-8 #change here
        difference = 1e99
        while difference > eps:
            old_sigmas = sigmas.copy()
            means_per_ann, workers, partial_sigmas = conservative_means_and_sigmas(t, w, ann, sigmas)
            sigmas[workers] = partial_sigmas
            difference = np.sum(np.abs(old_sigmas-sigmas))
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(T) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": sigmas}

def compute_sigmas_conservative2(t, w, ann):
    sigmas = np.ones(np.max(w) + 1)
    eps = 1e-8 #4 avant
    difference = 1e99
    overconfidence_reached = False
    while (difference > eps) and  not overconfidence_reached :
        old_sigmas = sigmas.copy()
        try:
            means_per_ann, workers, partial_sigmas = conservative_means_and_sigmas2(t, w, ann, sigmas)
            sigmas[workers] = partial_sigmas
            difference = np.sum(np.abs(old_sigmas - sigmas))
        except OverconfidenceException:
            overconfidence_reached = True
            #print("overconfindence")
    return sigmas

class OneShotConservative2(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        # _, counts = np.unique(w,return_counts=True)
        # print(counts)
        sigmas = compute_sigmas_conservative2(t, w, ann)
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(np.max(t) + 1) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": sigmas}


class KShot(ActiveAnnotationMethod):
    #construcor added there to define greedyness
    def __init__(self, greediness=2.0):
        self.greediness = greediness
    
    def run(self, exp: ActiveAnnotationContest):
        n_batches = 10
        batch_start = 0
        batch_size = exp.n_points // n_batches
        t, w, ann = random_annotation(exp, batch_start, batch_size)
        #print("anns:", t, w, ann)
        sigmas = compute_sigmas_conservative2(t, w, ann)
        #print("sigmas", sigmas)
        while (batch_start + batch_size) < exp.n_points:
            batch_start = batch_start + batch_size
            batch_size = min(batch_size, exp.n_points - batch_start)
            t_, w_, ann_ = sigma_annotation(exp, sigmas, greediness=self.greediness, batch_start=batch_start,
                                            batch_size=batch_size)
            #print("anns2:", t_, w_, ann_)
            t = np.concatenate((t, t_))
            w = np.concatenate((w, w_))
            ann = np.concatenate((ann, ann_))
            sigmas = compute_sigmas_conservative2(t, w, ann)
            
            #print("sigmas2", sigmas)
        _, counts = np.unique(t, return_counts=True)
        #print("c/w:", counts)
        #print(np.sum(counts))
        _, counts = np.unique(w, return_counts=True)
        #print("c/w:", counts)
        #print(np.sum(counts))
        tasks, means = imean_averaging(t, w, ann, sigmas)
        #print("final sigmas: ",sigmas)
        v = np.ones(np.max(t) + 1) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"w":w, "locations": v, "sigmas": sigmas}


