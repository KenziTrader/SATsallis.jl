# SATsallis.jl

**Simulated Annealing** global optimization in Julia with a generalization of Gibbs startistics (C. Tsallis, `Physica A` **233** (1996) 395-406). Incluides an exact random number generator for the q-Gaussian distribution (T. Schanze, `Comp. Phys. Comm.` **175** (2006) 708–712).

The backbone of this Simulated Annealing code is based on [Optim.jl](https://github.com/JuliaOpt/Optim.jl).

Generalized (Tsallis) simulated annealing may sometime outperform certain Differential Evolution heuristics (Xiang et al. `R Journal` 5/1 (2013) 13-28). A weak point of the current code is handling box constraints - this aspect has ample room for improvement.

(2015) Multi-disciplinary Insights LLC.

*MIT license*.

