# graphalc
v0 code for graph agglomerative likelihood clustering. a 1st implementation for a sparse solution to ALC.
Significantly faster clustering algorithm which first maps data to a sparse graph and merges clusters according to rules and quality Giada-Marsili Likelihood Function.
Capable of processing large data-sets 1M in 20mins on an i9 processor.
## Requires
Numpy, sklearn, numba, networkx, graphblas_algorithms, graphblas, pynndescent, scipy
(will provide versioning later, but latest versions of these libraries work without issues at the time November 2023)
