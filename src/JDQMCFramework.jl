module JDQMCFramework

using LinearAlgebra
using FastLapackInterface
using Checkerboard
using StableLinearAlgebra

# importing function to be overloaded
import Base: size, copyto!, iterate, eltype, resize!
import LinearAlgebra: mul!, lmul!, rmul!, ldiv!, rdiv!, ishermitian

# import functions for multiplying/dividing by diagonal matrix represented by a vector
import StableLinearAlgebra: mul_D!, div_D!, lmul_D!, rmul_D!, ldiv_D!, rdiv_D!

@doc raw"""
    Continuous = Union{AbstractFloat,Complex{<:AbstractFloat}}
    
An abstract type to represent continuous real and complex numbers.
"""
Continuous = Union{AbstractFloat,Complex{<:AbstractFloat}}

# various utility function
include("utility_functions.jl")
export eval_length_imaginary_axis # calculate length of imaginary time axis Lτ given β and Δτ
export exp! # exponentiate matrix while avoiding dynamic memory allocations
export build_hopping_matrix! # construct the hopping matrix given a neighbor table and vector or hopping amplitudes

# define types to represent propagator matrices, allowing for both symmetric and asymmetric propagators,
# and also the use of either the exact exponentiated hopping matrix or the checkerboard approximation
include("Propagators.jl")
export AbstractPropagator
export AbstractExactPropagator, AbstractChkbrdPropagator
export SymExactPropagator, AsymExactPropagator
export SymChkbrdPropagator, AsymChkbrdPropagator
export SymPropagators

# defines FermionGreensCalculator type to simplify the process of calculating
# single-particle fermion green's funciton matrices
include("FermionGreensCalculator.jl")
export FermionGreensCalculator
export resize!

# implements core routines that are useful in writing a DQMC code
include("dqmc_routines.jl")
export propagate_equaltime_greens!, stabilize_equaltime_greens!, calculate_equaltime_greens!
export initialize_unequaltime_greens!, propagate_unequaltime_greens!, stabilize_unequaltime_greens!
export local_update_det_ratio, local_update_greens!

end