# Multiple target tracing (MTT) for Python

A Python implementation of the multiple-target tracing (MTT) algorithm to localize and track fluorophores in temporal super-resolution microscopy data. Takes Nikon ND2 files (and soon TIF files) as input and generates localization and tracking data readable by MATLAB and other programs. Requires the [https://pypi.org/project/nd2reader/](nd2reader) package.

The algorithm was originally developed for the problem of tracking fluorescently labeled proteins at the cell membrane by Arnauld Sergé, Nicolas Bertaux, Hervé Rigneault, and Didier Marguet in their paper

Sergé *et. al.* "Dynamic multiple-target tracing to probe spatiotemporal cartography of cell membranes." *Nature Methods* **5**, 687-694 (2008).

It has since been adapted for various biological tracking experiments. The algorithm sequentially performs the following steps:
1. Detect particles.
2. Localize particles with subpixel accuracy.
3. Start trajectories out of localizations from the first frame.
4. Iterate through the rest of the frames, reconnecting existing trajectories with new localizations according to a probabilistic model ("tracking").
5. Save data to a MAT file, which is readable by MATLAB or Python's scipy.io.loadmat function.

## Tracking implementation 

In the present implementation, the tracking step is performed by generating a semigraph between existing trajectories and new localizations in each frame - that is, a matrix with x-indices corresponding to individual trajectories and y-indices corresponding to individual localizations. A given element is nonzero if the corresponding localization lies within the maximum search radius of the corresponding trajectory. The goal of tracking is to permute the semigraph until the assignment of trajectories to localizations maximizes a likelihood function.

To reduce the complexity of the problem, semigraphs for each frame-frame comparison are split into independent sets of connected components. Arrows in the semigraph of each connected component are then weighted according to a probabilistic model of diffusion that incorporates that past history of each particle, and the semigraph is permuted until the matrix trace is maximized. Assignment of trajectory indices and localization indices along the diagonal gives the maximum-likelihood reconnection between trajectories and localizations.
