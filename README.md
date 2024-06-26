# graphalc
v0 code for graph agglomerative likelihood clustering. a 1st implementation for a sparse solution to ALC.
Significantly faster clustering algorithm which first maps data to a sparse graph and merges clusters according to rules and quality Giada-Marsili Likelihood Function.
Capable of processing large data-sets 1M in 20mins on an i9 processor.
## Requires
Numpy, sklearn, numba, networkx, graphblas_algorithms, graphblas, pynndescent, scipy
(will provide versioning later, but latest versions of these libraries work without issues at the time November 2023)

# Graph ALC: April 14th 2024 Update:
A revamped version of the original ALC (https://github.com/lyelibi/ALC) is now available
The revamped code is significantl streamlined and shows significant decrease in runtime with no apparent degradation to performance. It however remains to be properly tested and demonstrated.
The revamped code has been adapted, so to speak, to work with sparse similarity matrices, which means clustering weighted graphs is now a possibility. Again, this needs to be tested with synthetic and real data for various noise, signal, dimensionality AND sparsity levels.
The Sparse algorithm uses a built-in graph construction method which is borrowed (copied) from UMAP. The graph construction method is an open question at this time but it is not primordial for the time being. It will be revisited in future work.

## New files:
"alc_dense_v1.py" -> Revamped ALC for dense matrices.
"alc_sparse_v1.py" -> Revamped Graph ALC for Sparse Matrices.

## Dependencies:
dense: Numpy, Numba
sparse: Numpy, Numba, Scipy, Pynndescent, sklearn (again the versioning will have to wait)
