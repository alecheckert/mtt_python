# mtt_python

A Python implementation of the multiple-target tracing (MTT) algorithm to localize and track of fluorophores in temporal super-resolution microscopy data. Takes Nikon ND2 files (and soon TIF files) as input and generates localization and tracking data readable by MATLAB and other programs.

The algorithm was originally developed for the problem of tracking fluorescently labeled proteins at the cell membrane by Arnauld Sergé, Nicolas Bertaux, Hervé Rigneault, and Didier Marguet in

Sergé *et. al.* "Dynamic multiple-target tracing to probe spatiotemporal cartography of cell membranes." *Nature Methods* **5**, 687-694 (2008).

Requires the [https://pypi.org/project/nd2reader/](nd2reader) package.

There are five basic steps in the algorithm.
1. Detect particles.
2. Localize particles with subpixel accuracy.
3. Start trajectories out of localizations from the first frame.
4. Iterate through the rest of the frames, reconnecting existing trajectories with new localizations.
5. Save data to a MAT file, which is readable by MATLAB or Python's scipy.io.loadmat function.

Detection is accomplished with a generalized
