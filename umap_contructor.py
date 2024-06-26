import scipy.sparse.csgraph
import scipy.sparse
import numba
import time
import numpy as np
from pynndescent import NNDescent
from warnings import warn
from sklearn.utils import check_random_state

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf



def ts():
    return time.ctime(time.time())

@numba.jit(nopython=True)
def covariance_numba(arr1, arr2):
    """
    Calculate the covariance between two numpy arrays of float32 type.
    
    Parameters:
    arr1 (numpy array): First input array.
    arr2 (numpy array): Second input array.
    
    Returns:
    float32: Covariance of arr1 and arr2.
    """
    covariance_dist = 1.0 - np.sum(arr1  * arr2 ) / (arr1.shape[0] - 1)
    # covariance_dist = 1.0 - np.sum(arr1  * arr2 ) / arr1.shape[0]
    
    return covariance_dist


@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """A fast computation of knn indices.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor indices of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices




def raise_disconnected_warning(
    vertices_disconnected,
    total_rows,
    threshold=0.1,
    verbose=False,
):
    if vertices_disconnected > threshold * total_rows:
        warn(
            f"A large number of your vertices are disconnected from the manifold.\n"
            f"It has fully disconnected {vertices_disconnected} vertices.\n"
            f"You might consider using find_disconnected_points() to find and remove these points from your data.\n",
            # f"Use umap.utils.disconnected_vertices() to identify them.",
        )


# @numba.njit(
#     locals={
#         "psum": numba.types.float32,
#         "lo": numba.types.float32,
#         "mid": numba.types.float32,
#         "hi": numba.types.float32,
#     },
#     fastmath=True,
# )  # benchmarking `parallel=True` shows it to *decrease* performance
# def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
#     """Compute a continuous version of the distance to the kth nearest
#     neighbor. That is, this is similar to knn-distance but allows continuous
#     k values rather than requiring an integral k. In essence we are simply
#     computing the distance such that the cardinality of fuzzy set we generate
#     is k.

#     Parameters
#     ----------
#     distances: array of shape (n_samples, n_neighbors)
#         Distances to nearest neighbors for each samples. Each row should be a
#         sorted list of distances to a given samples nearest neighbors.

#     k: float
#         The number of nearest neighbors to approximate for.

#     n_iter: int (optional, default 64)
#         We need to binary search for the correct distance value. This is the
#         max number of iterations to use in such a search.

#     local_connectivity: int (optional, default 1)
#         The local connectivity required -- i.e. the number of nearest
#         neighbors that should be assumed to be connected at a local level.
#         The higher this value the more connected the manifold becomes
#         locally. In practice this should be not more than the local intrinsic
#         dimension of the manifold.

#     bandwidth: float (optional, default 1)
#         The target bandwidth of the kernel, larger values will produce
#         larger return values.

#     Returns
#     -------
#     knn_dist: array of shape (n_samples,)
#         The distance to kth nearest neighbor, as suitably approximated.

#     nn_dist: array of shape (n_samples,)
#         The distance to the 1st nearest neighbor for each point.
#     """
#     target = np.log2(k) * bandwidth
#     rho = np.zeros(distances.shape[0], dtype=np.float32)
#     result = np.zeros(distances.shape[0], dtype=np.float32)

#     mean_distances = np.mean(distances)

#     for i in range(distances.shape[0]):
#         lo = 0.0
#         hi = NPY_INFINITY
#         mid = 1.0

#         # TODO: This is very inefficient, but will do for now. FIXME
#         ith_distances = distances[i]
#         non_zero_dists = ith_distances[ith_distances > 0.0]
#         if non_zero_dists.shape[0] >= local_connectivity:
#             index = int(np.floor(local_connectivity))
#             interpolation = local_connectivity - index
#             if index > 0:
#                 rho[i] = non_zero_dists[index - 1]
#                 if interpolation > SMOOTH_K_TOLERANCE:
#                     rho[i] += interpolation * (
#                         non_zero_dists[index] - non_zero_dists[index - 1]
#                     )
#             else:
#                 rho[i] = interpolation * non_zero_dists[0]
#         elif non_zero_dists.shape[0] > 0:
#             rho[i] = np.max(non_zero_dists)

#         for n in range(n_iter):

#             psum = 0.0
#             for j in range(1, distances.shape[1]):
#                 d = distances[i, j] - rho[i]
#                 if d > 0:
#                     psum += np.exp(-(d / mid))
#                 else:
#                     psum += 1.0

#             if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
#                 break

#             if psum > target:
#                 hi = mid
#                 mid = (lo + hi) / 2.0
#             else:
#                 lo = mid
#                 if hi == NPY_INFINITY:
#                     mid *= 2
#                 else:
#                     mid = (lo + hi) / 2.0

#         result[i] = mid

#         # TODO: This is very inefficient, but will do for now. FIXME
#         if rho[i] > 0.0:
#             mean_ith_distances = np.mean(ith_distances)
#             if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
#                 result[i] = MIN_K_DIST_SCALE * mean_ith_distances
#         else:
#             if result[i] < MIN_K_DIST_SCALE * mean_distances:
#                 result[i] = MIN_K_DIST_SCALE * mean_distances

#     return result, rho


def nearest_neighbors(
    X,
    n_neighbors,
    metric,
    metric_kwds,
    angular,
    random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=-1,
    verbose=False,
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metric: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    low_memory: bool (optional, default True)
        Whether to pursue lower memory NNdescent.

    verbose: bool (optional, default False)
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    rp_forest: list of trees
        The random projection forest used for searching (if used, None otherwise)
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    # if metric == "precomputed":
    #     # Note that this does not support sparse distance matrices yet ...
    #     # Compute indices of n nearest neighbors
    #     knn_indices = fast_knn_indices(X, n_neighbors)
    #     # knn_indices = np.argsort(X)[:, :n_neighbors]
    #     # Compute the nearest neighbor distances
    #     #   (equivalent to np.sort(X)[:,:n_neighbors])
    #     knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    #     # Prune any nearest neighbours that are infinite distance apart.
    #     disconnected_index = knn_dists == np.inf
    #     knn_indices[disconnected_index] = -1

    #     knn_search_index = None
    if metric != "correlation":
        raise ValueError("Invalid metric. This function only supports 'correlation'.")

    else:
        # TODO: Hacked values for now
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        knn_search_index = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=covariance_numba,
            metric_kwds=metric_kwds,
            random_state=random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            compressed=False,
        )
        knn_indices, knn_dists = knn_search_index.neighbor_graph

    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, knn_search_index

# @numba.njit(
#     locals={
#         "knn_dists": numba.types.float32[:, ::1],
#         "sigmas": numba.types.float32[::1],
#         "rhos": numba.types.float32[::1],
#         "val": numba.types.float32,
#     },
#     parallel=True,
#     fastmath=True,
# )
# def compute_membership_strengths(
#     knn_indices,
#     knn_dists,
#     sigmas,
#     rhos,
#     return_dists=False,
#     bipartite=False,
# ):
#     """Construct the membership strength data for the 1-skeleton of each local
#     fuzzy simplicial set -- this is formed as a sparse matrix where each row is
#     a local fuzzy simplicial set, with a membership strength for the
#     1-simplex to each other data point.

#     Parameters
#     ----------
#     knn_indices: array of shape (n_samples, n_neighbors)
#         The indices on the ``n_neighbors`` closest points in the dataset.

#     knn_dists: array of shape (n_samples, n_neighbors)
#         The distances to the ``n_neighbors`` closest points in the dataset.

#     sigmas: array of shape(n_samples)
#         The normalization factor derived from the metric tensor approximation.

#     rhos: array of shape(n_samples)
#         The local connectivity adjustment.

#     return_dists: bool (optional, default False)
#         Whether to return the pairwise distance associated with each edge

#     bipartite: bool (optional, default False)
#         Does the nearest neighbour set represent a bipartite graph?  That is are the
#         nearest neighbour indices from the same point set as the row indices?

#     Returns
#     -------
#     rows: array of shape (n_samples * n_neighbors)
#         Row data for the resulting sparse matrix (coo format)

#     cols: array of shape (n_samples * n_neighbors)
#         Column data for the resulting sparse matrix (coo format)

#     vals: array of shape (n_samples * n_neighbors)
#         Entries for the resulting sparse matrix (coo format)

#     dists: array of shape (n_samples * n_neighbors)
#         Distance associated with each entry in the resulting sparse matrix
#     """
#     n_samples = knn_indices.shape[0]
#     n_neighbors = knn_indices.shape[1]

#     rows = np.zeros(knn_indices.size, dtype=np.int32)
#     cols = np.zeros(knn_indices.size, dtype=np.int32)
#     vals = np.zeros(knn_indices.size, dtype=np.float32)
#     if return_dists:
#         dists = np.zeros(knn_indices.size, dtype=np.float32)
#     else:
#         dists = None

#     for i in range(n_samples):
#         for j in range(n_neighbors):
#             if knn_indices[i, j] == -1:
#                 continue  # We didn't get the full knn for i
#             # If applied to an adjacency matrix points shouldn't be similar to themselves.
#             # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
#             if (bipartite == False) & (knn_indices[i, j] == i):
#                 val = 0.0
#             elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
#                 val = 1.0
#             else:
#                 val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

#             rows[i * n_neighbors + j] = i
#             cols[i * n_neighbors + j] = knn_indices[i, j]
#             vals[i * n_neighbors + j] = val
#             if return_dists:
#                 dists[i * n_neighbors + j] = knn_dists[i, j]

#     return rows, cols, vals, dists


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths_custom(
    knn_indices,
    knn_dists,
    return_dists=False,
    bipartite=False,
):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    return_dists: bool (optional, default False)
        Whether to return the pairwise distance associated with each edge

    bipartite: bool (optional, default False)
        Does the nearest neighbour set represent a bipartite graph?  That is are the
        nearest neighbour indices from the same point set as the row indices?

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)

    dists: array of shape (n_samples * n_neighbors)
        Distance associated with each entry in the resulting sparse matrix
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            # if (bipartite == False) & (knn_indices[i, j] == i):
            #     val = 0.0
            # elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
            #     val = 1.0
            # else:
            #     val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
            
            val = 1 - knn_dists[i, j]
            if val > 1:
                val = 1
            elif val < -1:
                val = -1

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists



# def fuzzy_simplicial_set(
#     X,
#     n_neighbors,
#     random_state,
#     metric,
#     metric_kwds={},
#     knn_indices=None,
#     knn_dists=None,
#     angular=False,
#     set_op_mix_ratio=1.0,
#     local_connectivity=1.0,
#     apply_set_operations=True,
#     verbose=False,
#     return_dists=None,
# ):
#     """Given a set of data X, a neighborhood size, and a measure of distance
#     compute the fuzzy simplicial set (here represented as a fuzzy graph in
#     the form of a sparse matrix) associated to the data. This is done by
#     locally approximating geodesic distance at each point, creating a fuzzy
#     simplicial set for each such point, and then combining all the local
#     fuzzy simplicial sets into a global one via a fuzzy union.

#     Parameters
#     ----------
#     X: array of shape (n_samples, n_features)
#         The data to be modelled as a fuzzy simplicial set.

#     n_neighbors: int
#         The number of neighbors to use to approximate geodesic distance.
#         Larger numbers induce more global estimates of the manifold that can
#         miss finer detail, while smaller values will focus on fine manifold
#         structure to the detriment of the larger picture.

#     random_state: numpy RandomState or equivalent
#         A state capable being used as a numpy random state.

#     metric: string or function (optional, default 'euclidean')
#         The metric to use to compute distances in high dimensional space.
#         If a string is passed it must match a valid predefined metric. If
#         a general metric is required a function that takes two 1d arrays and
#         returns a float can be provided. For performance purposes it is
#         required that this be a numba jit'd function. Valid string metrics
#         include:
#             * euclidean (or l2)
#             * manhattan (or l1)
#             * cityblock
#             * braycurtis
#             * canberra
#             * chebyshev
#             * correlation
#             * cosine
#             * dice
#             * hamming
#             * jaccard
#             * kulsinski
#             * ll_dirichlet
#             * mahalanobis
#             * matching
#             * minkowski
#             * rogerstanimoto
#             * russellrao
#             * seuclidean
#             * sokalmichener
#             * sokalsneath
#             * sqeuclidean
#             * yule
#             * wminkowski

#         Metrics that take arguments (such as minkowski, mahalanobis etc.)
#         can have arguments passed via the metric_kwds dictionary. At this
#         time care must be taken and dictionary elements must be ordered
#         appropriately; this will hopefully be fixed in the future.

#     metric_kwds: dict (optional, default {})
#         Arguments to pass on to the metric, such as the ``p`` value for
#         Minkowski distance.

#     knn_indices: array of shape (n_samples, n_neighbors) (optional)
#         If the k-nearest neighbors of each point has already been calculated
#         you can pass them in here to save computation time. This should be
#         an array with the indices of the k-nearest neighbors as a row for
#         each data point.

#     knn_dists: array of shape (n_samples, n_neighbors) (optional)
#         If the k-nearest neighbors of each point has already been calculated
#         you can pass them in here to save computation time. This should be
#         an array with the distances of the k-nearest neighbors as a row for
#         each data point.

#     angular: bool (optional, default False)
#         Whether to use angular/cosine distance for the random projection
#         forest for seeding NN-descent to determine approximate nearest
#         neighbors.

#     set_op_mix_ratio: float (optional, default 1.0)
#         Interpolate between (fuzzy) union and intersection as the set operation
#         used to combine local fuzzy simplicial sets to obtain a global fuzzy
#         simplicial sets. Both fuzzy set operations use the product t-norm.
#         The value of this parameter should be between 0.0 and 1.0; a value of
#         1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
#         intersection.

#     local_connectivity: int (optional, default 1)
#         The local connectivity required -- i.e. the number of nearest
#         neighbors that should be assumed to be connected at a local level.
#         The higher this value the more connected the manifold becomes
#         locally. In practice this should be not more than the local intrinsic
#         dimension of the manifold.

#     verbose: bool (optional, default False)
#         Whether to report information on the current progress of the algorithm.

#     return_dists: bool or None (optional, default None)
#         Whether to return the pairwise distance associated with each edge.

#     Returns
#     -------
#     fuzzy_simplicial_set: coo_matrix
#         A fuzzy simplicial set represented as a sparse matrix. The (i,
#         j) entry of the matrix represents the membership strength of the
#         1-simplex between the ith and jth sample points.
#     """
#     if knn_indices is None or knn_dists is None:
#         knn_indices, knn_dists, _ = nearest_neighbors(
#             X,
#             n_neighbors,
#             metric,
#             metric_kwds,
#             angular,
#             random_state,
#             verbose=verbose,
#         )

#     knn_dists = knn_dists.astype(np.float32)

#     sigmas, rhos = smooth_knn_dist(
#         knn_dists,
#         float(n_neighbors),
#         local_connectivity=float(local_connectivity),
#     )

#     rows, cols, vals, dists = compute_membership_strengths(
#         knn_indices, knn_dists, sigmas, rhos, return_dists
#     )

#     # min_ = 

#     result = scipy.sparse.coo_matrix(
#         (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
#     )
#     result.eliminate_zeros()

#     if apply_set_operations:
#         transpose = result.transpose()

#         prod_matrix = result.multiply(transpose)

#         result = (
#             set_op_mix_ratio * (result + transpose - prod_matrix)
#             + (1.0 - set_op_mix_ratio) * prod_matrix
#         )

#     result.eliminate_zeros()

#     if return_dists is None:
#         return result, sigmas, rhos
#     else:
#         if return_dists:
#             dmat = scipy.sparse.coo_matrix(
#                 (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
#             )

#             dists = dmat.maximum(dmat.transpose()).todok()
#         else:
#             dists = None

#         return result, sigmas, rhos, dists
    
    
def fuzzy_simplicial_set_custom(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
    return_dists=None,
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean (or l2)
            * manhattan (or l1)
            * cityblock
            * braycurtis
            * canberra
            * chebyshev
            * correlation
            * cosine
            * dice
            * hamming
            * jaccard
            * kulsinski
            * ll_dirichlet
            * mahalanobis
            * matching
            * minkowski
            * rogerstanimoto
            * russellrao
            * seuclidean
            * sokalmichener
            * sokalsneath
            * sqeuclidean
            * yule
            * wminkowski

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    return_dists: bool or None (optional, default None)
        Whether to return the pairwise distance associated with each edge.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            n_neighbors,
            metric,
            metric_kwds,
            angular,
            random_state,
            verbose=verbose,
        )

    knn_dists = knn_dists.astype(np.float32)

    # sigmas, rhos = smooth_knn_dist(
    #     knn_dists,
    #     float(n_neighbors),
    #     local_connectivity=float(local_connectivity),
    # )

    rows, cols, vals, dists = compute_membership_strengths_custom(
        knn_indices, knn_dists, return_dists
    )


    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    # result.data = np.around(result.data, 2)
    result.setdiag(1)
    result.eliminate_zeros()

    # if return_dists is None:
    #     return result
    # else:
    #     if return_dists:
    #         dmat = scipy.sparse.coo_matrix(
    #             (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
    #         )

    #         dists = dmat.maximum(dmat.transpose()).todok()
    #     else:
    #         dists = None
            
        # result.data = np.around(result.data, 2)

    return result#, dists


@numba.njit
def custom_intersect1d(arr1, arr2):
    """
    Custom intersection function compatible with Numba.
    Assumes unique elements within each array but not between arrays.
    """
    set1 = set(arr1)
    return np.array([item for item in arr2 if item in set1])

@numba.njit
def compute_updates(indices, indptr, data, shape):
    new_vals = []
    for i in range(shape[0]):
        start_i, end_i = indptr[i], indptr[i+1]
        n_i = indices[start_i:end_i]
        
        for j in n_i:
            start_j, end_j = indptr[j], indptr[j+1]
            n_j = indices[start_j:end_j]
            
            section = custom_intersect1d(n_i, n_j)
            if len(section) <= 1:
                new_vals.append(data[start_i:end_i][np.searchsorted(n_i, j)])
                # new_vals.append(0)
                continue
            
            # Extract ps values
            ps_vals = np.zeros(len(section))
            for idx, sec in enumerate(section):
                ps_idx = np.searchsorted(n_i, sec)
                if ps_idx < len(n_i) and n_i[ps_idx] == sec:
                    ps_vals[idx] = data[start_i:end_i][ps_idx]
            
            if len(ps_vals) == 0:
                continue
            
            # update_value = (1/(1+np.exp(-0.5*np.sum(ps_vals))) - 0.5) * (1 - np.mean(ps_vals))/(1 - 0.5) + np.mean(ps_vals)
            
            val = np.mean(ps_vals)
            if val < data[start_i:end_i][np.searchsorted(n_i, j)]:
                
                update_value = np.min(ps_vals)
            else:
                update_value = np.max(ps_vals)
            new_vals.append(update_value)
    
    return new_vals


def build_graph(X, metric='correlation', mesosim = False, n_neighbors = 15,
                verbose=False, angular=False,n_jobs=-1,
                random_state = None, ):
    
    
    random_state = check_random_state(random_state)
    _knn_indices, _knn_dists, _ = nearest_neighbors(
                                    X,
                                    n_neighbors = n_neighbors,
                                    metric = metric,
                                    metric_kwds={},
                                    angular = angular,
                                    random_state = random_state,
                                    low_memory=True,
                                    use_pynndescent=True,
                                    n_jobs=n_jobs,
                                    verbose=verbose)
    
    
    
    
    # graph_, sigmas, rhos = fuzzy_simplicial_set(
    graph_ = fuzzy_simplicial_set_custom(
        X,
        n_neighbors,
        random_state,
        metric,
        metric_kwds={},
        knn_indices=_knn_indices,
        knn_dists=_knn_dists,
        angular=angular,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=verbose,
        return_dists=None,)
    
    if mesosim == True:
        g_csr = graph_.copy()
        g_csr = g_csr.tocsr()
        shape = g_csr.shape
        g_csr.setdiag(0)
        g_csr.eliminate_zeros()
        indices, indptr, vals_ = g_csr.indices, g_csr.indptr, g_csr.data

        # Compute updates
        new_vals = compute_updates(indices, indptr, vals_, shape)

        n_rows = len(indptr) - 1  # Number of rows
        row = np.repeat(np.arange(n_rows), np.diff(indptr))
        col = indices

        mesograph = scipy.sparse.csr_matrix((new_vals,(row,col)),shape=g_csr.shape)
        mesograph.setdiag(1)
        return mesograph

        


    vertices_disconnected = np.sum(
        np.array(graph_.sum(axis=1)).flatten() == 0
    )    
    
    raise_disconnected_warning(
        vertices_disconnected,
        X.shape[0],
        verbose=verbose,
    )
    
    return graph_