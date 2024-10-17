# JDQMCFramework.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SmoQySuite.github.io/JDQMCFramework.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SmoQySuite.github.io/JDQMCFramework.jl/dev/)
[![Build Status](https://github.com/SmoQySuite/JDQMCFramework.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SmoQySuite/JDQMCFramework.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/SmoQySuite/JDQMCFramework.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SmoQySuite/JDQMCFramework.jl)

This package exports a suite of types and routines that significantly simplify the process of writing a determinant quantum
Monte Carlo (DQMC) code.

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Installation
To install [JDQMCFramework.jl](https://github.com/SmoQySuite/JDQMCFramework.jl.git),
simply open the Julia REPL and run the commands
```julia
julia> ]
pkg> add JDQMCFramework
```
or equivalently via `Pkg` do
```julia
julia> using Pkg; Pkg.add("JDQMCFramework")
```

## Documentation

- [STABLE](https://SmoQySuite.github.io/JDQMCFramework.jl/stable/): Documentation for the latest version of the code published to the Julia [`General`](https://github.com/JuliaRegistries/General.git) registry.
- [DEV](https://SmoQySuite.github.io/JDQMCFramework.jl/dev/): Documentation for the latest commit to the `master` branch.

## Citation

If you found this library to be useful in the course of academic work, please consider citing us:

```bibtex
@misc{SmoQyDQMC,
      title={SmoQyDQMC.jl: A flexible implementation of determinant quantum Monte Carlo for Hubbard and electron-phonon interactions}, 
      author={Benjamin Cohen-Stead and Sohan Malkaruge Costa and James Neuhaus and Andy Tanjaroon Ly and Yutan Zhang and Richard Scalettar and Kipton Barros and Steven Johnston},
      year={2023},
      eprint={2311.09395},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el},
      url={https://arxiv.org/abs/2311.09395}
}
```