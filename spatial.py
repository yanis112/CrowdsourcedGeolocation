import numpy as np
import cmdstanpy as cmd

from .experiment import ActiveAnnotationContest, ActiveAnnotationMethod, random_annotation,\
    sigma_annotation, sigma_assignment, random_assignment

from .cmdstan import resource_filename


def compute_functionsigmas(kappa, y_grid, l=15):

    gridpoints = np.arange(l) / (l - 1.)
    allx = np.arange(1000) / 1000.
    n_annotators = y_grid.shape[0]

    functionsigmas_learned = []
    for i in range(n_annotators):
        functionsigmas_learned.append(compute_functionsigmas_aux(allx, kappa, gridpoints, y_grid[i]))

    return functionsigmas_learned


def compute_functionsigmas_aux(t, kappa, x, y):
    r = len(x)
    tr = np.repeat(t, r)
    tr.shape = (len(t), r)
    d = (tr - x) * (tr - x)
    ed = np.exp(-kappa * d)
    s = np.sum(ed, axis=1)
    res = np.dot(ed, y)/s
    return res


class OneShotSpatialBayesian(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)

        n_annotators = exp.n_annotators
        n_points = exp.n_points

        allx = np.arange(1000) / 1000.
        l = 15

        d = {"w": n_annotators,
             "a": len(ann),
             "t": n_points,
             "t_A": t + 1,
             "w_A": w + 1,
             "ann": ann,
             "l": l
             }
        inits = {"y_grid": np.ones((n_annotators, l)) * 0.1}
        gp = cmd.CmdStanModel(stan_file=resource_filename('gp-learn-variances-ma.stan'))
        s = gp.optimize(data=d, inits=inits, show_console=True, iter=1000, algorithm='lbfgs', tol_rel_grad=10000.)

        kappa = s.stan_variable("kappa")
        y_grid = s.stan_variable("y_grid")
        v = s.stan_variable("x")

        gridpoints = np.arange(l) / (l - 1.)
        functionsigmas_learned = []
        for i in range(n_annotators):
            functionsigmas_learned.append(compute_functionsigmas_aux(allx, kappa, gridpoints, y_grid[i]))

        return {"locations": v, "sigmas": functionsigmas_learned}


def learn_variance_profiles(t, w, ann, l=15):
    n_annotators = np.max(w) + 1
    # print(n_annotators)
    n_points = np.max(t) + 1
    # print(n_points)
    d = {"w": n_annotators,
         "a": len(ann),
         "t": n_points,
         "t_A": t + 1,
         "w_A": w + 1,
         "ann": ann,
         "l": l
         }
    inits = {"y_grid": np.ones((n_annotators, l)) * 0.1}
    gp = cmd.CmdStanModel(stan_file=resource_filename('gp-learn-variances-ma.stan'))
    s = gp.optimize(data=d, inits=inits, show_console=True, iter=10000, algorithm='lbfgs', tol_rel_grad=10000.)
    kappa = s.stan_variable("kappa")
    y_grid = s.stan_variable("y_grid")
    return kappa, y_grid


def compute_mean_sigmas(n_annotators, sigma_functions):
    mean_sigmas = []
    for i in range(n_annotators):
        mean_sigmas.append(np.mean(sigma_functions[i]))
    return np.array(mean_sigmas)


def position_based_round(exp, tasks, positions, annotators_per_task, functionlearned_sigmas=None, greediness=1.):
    # Miramos las sigmas-funciones en cada punto para volver a hacer un request:
    N = len(tasks)
    t3 = np.zeros(N, dtype=int)
    w3 = np.zeros(N, dtype=int)
    ann3 = np.zeros(N)

    for i, p in enumerate(positions):
        # sigma_at_point = np.array([annotator.sigma([p])[0] for annotator in exp.annotator_set.annotators])
        if p < 0.:
            p = 0.
        elif p >= 1.:
            p = 0.999 #we should treat this better... maybe we could assert it when annotating
        sigma_at_point = []
        index_aux = int(p*1000)
        for annotator in range(exp.n_annotators):
            sigma_at_point.append(functionlearned_sigmas[annotator][index_aux])

        previous_anns = []
        for ann_index in annotators_per_task[tasks[i]]:
            sigma_at_point[ann_index] = 1e3
            previous_anns.append(ann_index)
            #sigma_at_point.pop(ann_index)
        # print("sigma=", sigma_at_point)
        t3[i:i + 1], w3[i:i + 1], ann3[i:i + 1] = sigma_annotation(exp, np.array(sigma_at_point),
                                                                   previous_anns=np.array(previous_anns),
                                                                   batch_start=tasks[i],
                                                                   batch_size=1, k=1, greediness=greediness)
                                            #podemos tener problemas si otorgamos una probabilidad pequeña al anotador que ha anotado anteriormente

        # t_aux, w_aux = sigma_assignment(1, np.array(sigma_point), 1, batch_start=t2[i])
        # print(t_aux)
        # t_aux = np.array([t2[i]])
        # print(t_aux)
        # ann_aux = exp.batch_request(t_aux, w_aux)
        # print(p,t_aux,w_aux)
        # t3.append(t_aux[0])
        # w3.append(w_aux[0])
        # ann3.append(ann_aux[0])
    return t3, w3, ann3


def compute_annotators_per_task(t, w):
    from collections import defaultdict
    annotators_per_task = defaultdict(set)
    for i, w in enumerate(w):
        annotators_per_task[t[i]].add(w)
    return annotators_per_task


def diagnose_errors(t, w, ann):
    annotators_per_task = compute_annotators_per_task(t, w)
    for t in annotators_per_task.values():
        if len(t) < 2:
            print("There is a task with less than two annotators")
            raise Exception("Task with less than two annotators")
    print("All tasks in the dataset contain at least two annotators. GREAT!")


def spatial_imean_averaging(t, w, ann, sigma_functions):

    tasks = []
    means = []
    for i, task in enumerate(np.unique(t)):
        sigmas_at_point = []
        for j, annotator in enumerate(w[t == task]):
            annotation_aux = ann[t == task][j]
            if annotation_aux < 0.:
                annotation_aux = 0.
            elif annotation_aux >= 1.:
                annotation_aux = 0.999
            index_aux = int(annotation_aux * 1000)
            sigmas_at_point.append(sigma_functions[annotator][index_aux])

        tasks.append(
            imean_averaging(t[t == task], np.arange(j + 1), ann[t == task], sigmas=np.array(sigmas_at_point))[0][0])
        means.append(
            imean_averaging(t[t == task], np.arange(j + 1), ann[t == task], sigmas=np.array(sigmas_at_point))[1][0])

    return tasks, means


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


class KShotSpatial(ActiveAnnotationMethod):

    def __init__(self, greediness):
        self.greediness = greediness

    def run(self, exp: ActiveAnnotationContest):
        # 1. 10% of the points are annotated randomly
        initial_percentage = 0.1
        initial_batch_size = int(exp.n_points*initial_percentage)
        t1, w1, ann1 = random_annotation(exp, 0, initial_batch_size)

        # 2. We learn the sigma functions of all annotators
        kappa1, y_grid1 = learn_variance_profiles(t1, w1, ann1)
        functionsigmas_learned1 = compute_functionsigmas(kappa1, y_grid1)

        # 3. We compute the mean sigma for each annotator
        mean_sigmas = compute_mean_sigmas(len(functionsigmas_learned1), functionsigmas_learned1)

        # 4. The other 90% of the points are annotated just once taking into account the mean sigmas
        second_batch_size = exp.n_points - initial_batch_size
        second_batch_start = initial_batch_size
        t2, w2, ann2 = sigma_annotation(exp, mean_sigmas, greediness=self.greediness, batch_start=second_batch_start,
                                        batch_size=second_batch_size, k=1) #should greediness follow a certain criteria?

        last_t, last_w, last_consensus = t2, w2, ann2
        last_sigmafunctions = functionsigmas_learned1
        t_historic, w_historic, ann_historic = t2, w2, ann2

        iter = 0
        while iter < (exp.max_total_annotations/exp.n_points - 1):
            # 5. The same 90% of points are annotated just once using the learned sigma values at the previous points
            annotators_per_task = compute_annotators_per_task(t_historic, w_historic)
            t3, w3, ann3 = position_based_round(exp, last_t, last_consensus, annotators_per_task, last_sigmafunctions, greediness=self.greediness)
            t_historic = np.concatenate((t_historic, t3))
            w_historic = np.concatenate((w_historic, w3))
            ann_historic = np.concatenate((ann_historic, ann3))

            # 6. We recompute the sigma profiles
            t = np.concatenate((t1, t_historic))
            w = np.concatenate((w1, w_historic))
            ann = np.concatenate((ann1, ann_historic))
            kappa2, y_grid2 = learn_variance_profiles(t, w, ann)
            functionsigmas_learned2 = compute_functionsigmas(kappa2, y_grid2)
            last_sigmafunctions = functionsigmas_learned2

            # 7. We perform an intelligent mean between the last two annotations taking into account the new sigmas
            tasks, means = spatial_imean_averaging(t_historic, w_historic, ann_historic, last_sigmafunctions)
            last_consensus = means
            last_t = tasks
            iter += 1

        initial_tasks, initial_means = spatial_imean_averaging(t1, w1, ann1, last_sigmafunctions)
        locations = np.concatenate((initial_means, last_consensus))

        return {'locations': locations, 'sigmas': last_sigmafunctions}
