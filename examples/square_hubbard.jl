# # Tutorial 1: Square Lattice Hubbard Model DQMC Simulation
# 
# This tutorial implements a determinant quantum Monte Carlo (DQMC) simulation from "scratch"
# using the [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git) package, along with several others.
# The purpose of this tutorial is to empower researchers to write their own lightweight DQMC codes
# in order to address specific research needs that fall outside the scope of existing high-level
# DQMC packages like [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl.git), and to enable
# rapid prototyping of algorithmic improvements to existing DQMC methods.
#
# This tutorial is relatively long as a lot goes into writing a full DQMC code.
# However, in spite of the length, each step is relatively straightforward.
# This is made possible by leveraging the functionality exported by
# [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git) and other packages.
# For instance, the [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git) package takes care of all the
# numerical stabilization nonsense that is one of the most challenging parts of writing a DQMC code.
# Also, implementing various correlation measurements in a DQMC simulation is typically very time consuming and challening,
# as it requires working through arduous Wick's contractions, and then implementing each term.
# Once again, this hurdle is largely avoided by leveraging the functionality exported by the
# [`JDQMCMeasurements.jl`](https://github.com/SmoQySuite/JDQMCMeasurements.jl.git) package,
# which implements a variety of standard correlation function measurements for arbitary lattice geometries.
#
# The repulsive Hubbard model Hamiltonian on a square lattice considered in this tutorial is given by
# ```math
# \hat{H} = -t \sum_{\sigma,\langle i,j\rangle} (\hat{c}^{\dagger}_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
#   + U \sum_i (\hat{n}_{\uparrow,i}-\tfrac{1}{2})(\hat{n}_{\downarrow,i}-\tfrac{1}{2}) - \mu \sum_{\sigma,i} \hat{n}_{\sigma,i},
# ```
# where ``\hat{c}^\dagger_{\sigma,i} (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
# electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
# is the spin-``\sigma`` electron number operator for site ``i``. In the above Hamiltonian, ``t`` is the nearest neighbor hopping integral,
# ``\mu`` is the chemical potential, and ``U > 0`` controls the strength of the on-site Hubbard repulsion.
# Lastly, if ``\mu = 0.0,`` then the Hamiltonian is particle-hole symmetric, ensuring the system is half-filled ``(\langle n_\sigma \rangle = \tfrac{1}{2})``
# and that there is no sign problem. In the case of ``\mu \ne 0`` there will be a sign problem.
#
# The script version of this tutorial, which can be downloaded using the link found at the top of this page,
# can be run with the command
# ```
# julia square_hubbard.jl
# ```
# in a terminal. This tutorial can also be downloaded as a notebook at the top of this page.
#
# We begin by importing the relevant packages we will need to use in this example.
# Note that to run this tutorial you will need to install all the required Julia packages.
# However, this is straightforward as all the packages used in this tutorial are registered
# with the Julia [General](https://github.com/JuliaRegistries/General.git)
# package registry. This means they can all be easily installed with the Julia package manager
# using the `add` command in the same way that the [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git)
# package is installed.

## Import relevant standard template libraries.
using Random
using LinearAlgebra

## Provides framework for implementing DQMC code.
import JDQMCFramework as jdqmcf

## Exports methods for measuring various correlation functions in a DQMC simulation.
import JDQMCMeasurements as jdqmcm

## Exports types and methods for representing lattice geometries.
import LatticeUtilities as lu

## Exports the checkerboard approximation for representing an exponentiated hopping matrix.
import Checkerboard as cb

## Package for performing Fast Fourier Transforms (FFTs).
using FFTW

# The next incantations are included for annoying technical reasons.
# Without going into too much detail, the default multithreading behavior used by BLAS/LAPACK in Julia is somewhat sub-optimal.
# As a result, it is typically a good idea to include these commands in Julia DQMC codes, as they ensure
# that BLAS/LAPACK (and FFTW) run in a single-threaded fashion. For more information on this issue,
# I refer readers to [this discussion](https://carstenbauer.github.io/ThreadPinning.jl/stable/explanations/blas/),
# which is found in the documentation for the [`ThreadPinning.jl`](https://github.com/carstenbauer/ThreadPinning.jl.git) package.

## Set number of threads used by BLAS/LAPACK to one.
BLAS.set_num_threads(1)

## Set number of threads used by FFTW to one.
FFTW.set_num_threads(1)

# Now we define the relevant Hamiltonian parameter values that we want to simulate.
# In this example we will stick to a relatively small system size ``(4 \times 4)`` and
# inverse temperature ``(\beta = 4)`` to ensure that this tutorial
# can be run quickly on a personal computer.
# Also, in this tutorial I will include many print statements so that when
# the tutorial is run users can keep track of what is going on. That said, for a DQMC code
# that will be used in actual research you will want to replace the print statements with code
# that writes relevant information and measurement results to file.

## Nearest-neighbor hopping amplitude.
t = 1.0
println("Nearest-neighbor hopping amplitude, t = ", t)

## Hubbard interaction.
U = 6.0
println("Hubbard interaction, U = ", U)

## Chemical potential.
μ = 2.0
println("Chemical potential, mu = ", μ)

## Inverse temperature.
β = 4.0
println("Inverse temperature, beta = ", β)

## Lattice size.
L = 4
println("Linear lattice size, L = ", L)
#jl println()

# Next we define the relevant DQMC simulation parameters.

## Discretization in imaginary time.
Δτ = 0.05
println("Disretization in imaginary time, dtau = ", Δτ)

## Length of imaginary time axis.
Lτ = round(Int, β/Δτ)
println("Length of imaginary time axis, Ltau = ", Lτ)

## Whether or not to use a symmetric or asymmetric definition for the propagator matrices.
symmetric = false
println("Whether symmetric or asymmetric propagator matrices are used, symmetric = ", symmetric)

## Whether or not to use the checkerboard approximation to represent the
## exponentiated electron kinetic energy matrix exp(-Δτ⋅K).
checkerboard = false
println("Whether the checkerboard approximation is used, checkerboard = ", checkerboard)

## Period with which numerical stabilization is performed i.e.
## how many imaginary time slices separate more expensive recomputations
## of the Green's function matrix using numerically stable routines.
n_stab = 10
println("Numerical stabilization period, n_stab = ", n_stab)

## The number of burnin sweeps through the lattice performing local updates that
## are performed to thermalize the system.
N_burnin = 2_500
println("Number of burnin sweeps, N_burnin = ", N_burnin)

## The number of measurements made once the system is thermalized.
N_measurements = 10_000
println("Number of measurements, N_measurements = ", N_measurements)

## Number of local update sweeps separating sequential measurements.
N_sweeps = 1
println("Number of local update sweeps seperating measurements, n_sweeps = ", N_sweeps)

## Number of bins used to performing a binning analysis when calculating final error bars
## for measured observables.
N_bins = 50
println("Number of measurement bins, N_bins = ", N_bins)

## Number of measurements averaged over per measurement bin.
N_binsize = N_measurements ÷ N_bins
println("Number of measurements per bin, N_binsize = ", N_binsize)
#jl println()

# Now we initialize the random number generator (RNG) that will be used in the rest of the simulation.

## Initialize random number generator.
seed = abs(rand(Int))
rng = Xoshiro(seed)
println("Random seed used to initialize RNG, seed = ", seed)
#jl println()

# Next, we define our square lattice geometry using the [`LatticeUtilities.jl`](https://github.com/SmoQySuite/LatticeUtilities.jl.git) package.

## Define the square lattice unit cell.
unit_cell = lu.UnitCell(
    lattice_vecs = [[1.0, 0.0],
                    [0.0, 1.0]],
    basis_vecs   = [[0.0, 0.0]]
)

## Define the size of the periodic square lattice.
lattice = lu.Lattice(
    L = [L, L],
    periodic = [true, true]
)

## Define nearest-neighbor bond in +x direction
bond_px = lu.Bond(
    orbitals = (1,1),
    displacement = [1,0]
)

## Define nearest-neighbor bond in +y direction
bond_py = lu.Bond(
    orbitals = (1,1),
    displacement = [0,1]
)

## Build the neighbor table corresponding to all nearest-neighbor bonds.
neighbor_table = lu.build_neighbor_table([bond_px, bond_py], unit_cell, lattice)
println("The neighbor table, neighbor_table =")
show(stdout, "text/plain", neighbor_table)
println("\n")

## The total number of sites/orbitals in the lattice.
N = lu.nsites(unit_cell, lattice) # For square lattice this is simply N = L^2
#md println("Total number of sites in lattice, N = ", N)
#nb println("Total number of sites in lattice, N = ", N)

## Total number of bonds in lattice.
N_bonds = size(neighbor_table, 2)
#md println("Total number of bonds in lattice, N_bonds = ", N_bonds)
#nb println("Total number of bonds in lattice, N_bonds = ", N_bonds)

# Now we define a few other bonds that are needed to measure
# the local s-wave, extended s-wave and d-wave pair susceptibilities.

## Define a "trivial" bond that maps a site back onto itself.
bond_trivial = lu.Bond(
    orbitals = (1,1),
    displacement = [0,0]
)

## Define bond in -x direction.
bond_nx = lu.Bond(
    orbitals = (1,1),
    displacement = [-1,0]
)

## Define bond in -y direction.
bond_ny = lu.Bond(
    orbitals = (1,1),
    displacement = [0,-1]
);

# Now let us calculated the exponentiated electron kinetic energy matrix ``e^{-\Delta\tau^\prime K}``,
# where
# ```math
# \hat{K} = -t \sum_{\sigma,\langle i,j\rangle} (\hat{c}^{\dagger}_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
#         = \sum_\sigma \hat{\mathbf{c}}^\dagger_\sigma K \hat{\mathbf{c}}^{\phantom\dagger}_\sigma
# ```
# and ``\hat{\mathbf{c}}^{\dagger}_{\sigma,i} = \left[ \hat{c}^{\dagger}_{\sigma,1} \ , \ \dots \ , \ \hat{c}^{\dagger}_{\sigma,N} \right]``
# is a row vector of electron creation operators.
# Note that if `symmetric = true`, i.e. the symmetric definition for the propagator matrices
# ```math
# B_{\sigma,l} = e^{-\Delta\tau^\prime K} \cdot e^{-\Delta\tau V_{\sigma,l}} \cdot e^{-\Delta\tau^\prime K}
# ```
# is being used, then ``\Delta\tau^\prime = \tfrac{1}{2} \Delta\tau``.
# If the asymmetric definition
# ```math
# B_{\sigma,l} = e^{-\Delta\tau V_{\sigma,l}} \cdot e^{-\Delta\tau^\prime K}
# ```
# is used (`symmetric = false`), then ``\Delta\tau^\prime = \Delta\tau.``
#
# Note the branching logic below associated with whether or not the
# matrix ``e^{-\Delta\tau^\prime K}`` is calculated exactly, or represented by the sparse checkerboard approximation
# using the package [`Checkerboard.jl`](https://github.com/SmoQySuite/Checkerboard.jl.git).

## Define Δτ′=Δτ/2 if symmetric = true, otherwise Δτ′=Δτ
Δτ′ = symmetric ? Δτ/2 : Δτ

## If the matrix exp(Δτ′⋅K) is represented by the checkerboard approximation.
if checkerboard

    ## Construct the checkerboard approximation to the matrix exp(-Δτ′⋅K).
    expnΔτ′K = cb.CheckerboardMatrix(neighbor_table, fill(t, N_bonds), Δτ′)

## If the matrix exp(Δτ′⋅K) is NOT represented by the checkerboard approximation.
else

    ## Construct the electron kinetic energy matrix.
    K = zeros(typeof(t), N, N)
    for bond in 1:N_bonds
        i, j = neighbor_table[1, bond], neighbor_table[2, bond]
        K[i,j] = -t
        K[j,i] = -conj(t)
    end

    ## Calculate the exponentiated kinetic energy matrix, exp(-Δτ⋅K).
    ## Note that behind the scenes Julia is diagonalizing the matrix K in order to exponentiate it.
    expnΔτ′K = exp(-Δτ′*K)

    ## Calculate the inverse of the exponentiated kinetic energy matrix, exp(+Δτ⋅K).
    exppΔτ′K = exp(+Δτ′*K)
end;

# In this example we are going to introduce an Ising Hubbard-Stratonovich (HS) field to decouple
# the Hubbard interaction. The Ising HS transformation
# ```math
# e^{-\Delta\tau U (\hat{n}_{\uparrow,i,l}-\tfrac{1}{2})(\hat{n}_{\downarrow,i,l}-\tfrac{1}{2})} = 
#   \frac{1}{2} e^{-\tfrac{1}{4} \Delta\tau U} \sum_{s_{i,l} = \pm 1} e^{\alpha s_{i,l}(\hat{n}_{\uparrow,i,l}-\hat{n}_{\downarrow,i,l})}
# ```
# is introduced for all imaginary time slices ``l \in [1, L_\tau]`` and sites ``i \in [1, N]`` in the lattice, where
# ```math
# \alpha = \cosh^{-1}\left( e^{\tfrac{1}{2}\Delta\tau U} \right)
# ```
# is a constant. We start the simulation from a random ``s_{i,l}`` Ising HS field configuration.

## Define constant associated Ising Hubbard-Stratonovich (HS) transformation.
α = acosh(exp(Δτ*U/2))

## Initialize a random Ising HS configuration.
s = rand(rng, -1:2:1, N, Lτ)
#md println("Random initial Ising HS configuration, s =")
#md show(stdout, "text/plain", s)
#nb println("Random initial Ising HS configuration, s =")
#nb show(stdout, "text/plain", s)

# Next we initialize a propagator matrix ``B_{\sigma,l}`` for each imaginary time slice ``l \in [1,L_\tau]``.
# We first initialize a pair of vectors `Bup` and `Bdn` that will contain the ``L_\tau`` propagators associated with each time slice.
# The branching logic below enforces the correct propagator matrix definition is used based on the boolean flags
# `symmetric` and `checkerboard` defined above.

## Matrix element type for exponentiated electron kinetic energy matrix exp{-Δτ′⋅K}
T_expnΔτK = eltype(t)

## Matrix element type for diagonal exponentiated electron potential energy matrix exp{-Δτ⋅V[σ,l]}
T_expnΔτV = typeof(α)

## Initialize empty vector to contain propagator matrices for each imaginary time slice.
if checkerboard && symmetric

    ## Propagator defined as B[σ,l] = exp{-Δτ⋅K/2}⋅exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K/2},
    ## where the dense matrix exp{-Δτ⋅K/2} is approximated by the sparse checkerboard matrix.
    Bup = jdqmcf.SymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.SymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]

elseif checkerboard && !symmetric

    ## Propagator defined as B[σ,l] = exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K},
    ## where the dense matrix exp{-Δτ⋅K} is approximated by the sparse checkerboard matrix.
    Bup = jdqmcf.AsymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.AsymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]

elseif !checkerboard && symmetric

    ## Propagator defined as B[σ,l] = exp{-Δτ⋅K/2}⋅exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K/2},
    ## where the dense matrix exp{-Δτ⋅K/2} is exactly calculated.
    Bup = jdqmcf.SymExactPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.SymExactPropagator{T_expnΔτK, T_expnΔτV}[]

elseif !checkerboard && !symmetric

    ## Propagator defined as B[σ,l] = exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K},
    ## where the dense matrix exp{-Δτ⋅K} is exactly calculated.
    Bup = jdqmcf.AsymExactPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.AsymExactPropagator{T_expnΔτK, T_expnΔτV}[]
end;

# Having an initialized the vector `Bup` and `Bdn` that will contain the propagator matrices, we now construct the
# propagator matrices for each time-slice based on the initial HS field configuration `s`.

## Iterate over time-slices.
for l in 1:Lτ

    ## Get the HS fields associated with the current time-slice l.
    s_l = @view s[:,l]

    ## Calculate the spin-up diagonal exponentiated potential energy
    ## matrix exp{-Δτ⋅V[↑,l]} = exp{-Δτ⋅(-α/Δτ⋅s[i,l]-μ)} = exp{+α⋅s[i,l] + Δτ⋅μ}.
    expnΔτVup = zeros(T_expnΔτV, N)
    @. expnΔτVup = exp(+α * s_l + Δτ*μ)

    ## Calculate the spin-down diagonal exponentiated potential energy
    ## matrix exp{-Δτ⋅V[↓,l]} = exp{-Δτ⋅(+α/Δτ⋅s[i,l]-μ)} = exp{-α⋅s[i,l] + Δτ⋅μ}.
    expnΔτVdn = zeros(T_expnΔτV, N)
    @. expnΔτVdn = exp(-α * s_l + Δτ*μ)
    
    ## Initialize spin-up and spin-down propagator matrix for the current time-slice l.
    if checkerboard && symmetric

        push!(Bup, jdqmcf.SymChkbrdPropagator(expnΔτVup, expnΔτ′K))
        push!(Bdn, jdqmcf.SymChkbrdPropagator(expnΔτVdn, expnΔτ′K))

    elseif checkerboard && !symmetric

        push!(Bup, jdqmcf.AsymChkbrdPropagator(expnΔτVup, expnΔτ′K))
        push!(Bdn, jdqmcf.AsymChkbrdPropagator(expnΔτVdn, expnΔτ′K))

    elseif !checkerboard && symmetric

        push!(Bup, jdqmcf.SymExactPropagator(expnΔτVup, expnΔτ′K, exppΔτ′K))
        push!(Bdn, jdqmcf.SymExactPropagator(expnΔτVdn, expnΔτ′K, exppΔτ′K))

    elseif !checkerboard && !symmetric

        push!(Bup, jdqmcf.AsymExactPropagator(expnΔτVup, expnΔτ′K, exppΔτ′K))
        push!(Bdn, jdqmcf.AsymExactPropagator(expnΔτVdn, expnΔτ′K, exppΔτ′K))
    end
end

# Now we instantiate two instances for the [`FermionGreensCalculator`](@ref) type, one for each spin
# species, spin up and spin down. This object enables the efficient and numerically stable calculation
# of the Green's functions behind-the-scenes, so that we do not need to concern ourselves with
# implementing numerical stablization routines ourselves.

## Initialize a FermionGreensCalculator for both spin up and down electrons.
fermion_greens_calc_up = jdqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
fermion_greens_calc_dn = jdqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab);

# Next we calculate the equal-time Green's function matrices
# ```math
# G_\sigma(0,0) = [1 + B_\sigma(\beta,0)]^{-1} = [1 + B_{\sigma,L_\tau} \dots B_{\sigma,1}]^{-1}
# ```
# for both electron spin species, ``\sigma = (\uparrow, \downarrow).``

## Calculate spin-up equal-time Green's function matrix.
Gup = zeros(typeof(t), N, N)
logdetGup, sgndetGup = jdqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calc_up)

## Calculate spin-down equal-time Green's function matrix.
Gdn = zeros(typeof(t), N, N)
logdetGdn, sgndetGdn = jdqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calc_dn);

# In order to perform the DQMC simulation all we need are the equal-time Green's function matrices
# ``G_\sigma(0,0)`` calculated above. However, in order to make time-displaced correlation function
# measurements we also need to initialize six more matrices, which correspond to ``G_\sigma(\tau,\tau),``
# ``G_\sigma(\tau,0)`` and ``G_\sigma(0,\tau).``

## Allcoate time-displaced Green's functions.
Gup_τ0 = zero(Gup) # Gup(τ,0)
Gup_0τ = zero(Gup) # Gup(0,τ)
Gup_ττ = zero(Gup) # Gup(τ,τ)
Gdn_τ0 = zero(Gdn) # Gdn(τ,0)
Gdn_0τ = zero(Gdn) # Gdn(0,τ)
Gdn_ττ = zero(Gdn); # Gdn(τ,τ)

# Now we will allocate arrays to contain the various measurements we will make during the simulation,
# including various correlation functions. Note that the definition for each measurement will be
# supplied later in the tutorial when we begin processing the data to calculate the final statistics
# for each measured observable.

## Vector to contain binned average sign measurement.
avg_sign = zeros(eltype(Gup), N_bins)

## Vector to contain binned density measurement.
density = zeros(eltype(Gup), N_bins)

## Vector to contain binned double occupancy measurement.
double_occ = zeros(eltype(Gup), N_bins)

## Array to contain binned position-space time-displaced Green's function measurements.
C_greens = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

## Array to contain binned position-space time-displaced Spin-Z correlation function measurements.
C_spinz = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

## Array to contain binned position-space time-displaced density correlation function measurements.
C_density = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

## Array to contain binned position-space local s-wave pair correlation function.
C_loc_swave = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

## Array to contain binned position-space extended s-wave pair correlation function.
C_ext_swave = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

## Array to contain binned position-space d-wave pair correlation function.
C_dwave = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

## Array to contain binned momentum-space d-wave pair susceptibility.
P_d_q = zeros(Complex{Float64}, N_bins, L, L);

# Below we implement a function that sweeps through all time-slices and sites in the lattice,
# attempting an update to each Ising HS field ``s_{i,l}``.

## Function to perform local updates to all Ising HS fields.
function local_update!(
    Gup::Matrix{T}, logdetGup, sgndetGup, Bup, fermion_greens_calc_up,
    Gdn::Matrix{T}, logdetGdn, sgndetGdn, Bdn, fermion_greens_calc_dn,
    s, μ, α, Δτ, δG, rng
) where {T<:Number}

    ## Length of imaginary time axis.
    Lτ = length(Bup)

    ## Number of sites in lattice.
    N = size(Gup,1)

    ## Allocate temporary arrays that will be used to avoid dynamic memory allocation.
    A = zeros(T, N, N)
    u = zeros(T, N)
    v = zeros(T, N)

    ## Allocate vector of integers to contain random permutation specifying the order in which
    ## sites are iterated over at each imaginary time slice when performing local updates.
    perm = collect(1:size(Gup,1))

    ## Variable to keep track of the acceptance rate.
    acceptance_rate = 0.0

    ## Iterate over imaginary time slices.
    for l in fermion_greens_calc_up

        ## Propagate equal-time Green's function matrix to current imaginary time
        ## G(τ±Δτ,τ±Δτ) ==> G(τ,τ) depending on whether iterating over imaginary
        ## time in the forward or reverse direction
        jdqmcf.propagate_equaltime_greens!(Gup, fermion_greens_calc_up, Bup)
        jdqmcf.propagate_equaltime_greens!(Gdn, fermion_greens_calc_dn, Bdn)

        ## If using symmetric propagator definition (symmetric = true), then apply
        ## the transformation G ==> G̃ = exp{+Δτ⋅K/2}⋅G⋅exp{-Δτ⋅K/2}.
        ## If asymmetric propagator definition is used (symmetric = false),
        ## then this does nothing.
        jdqmcf.partially_wrap_greens_forward!(Gup, Bup[l], A)
        jdqmcf.partially_wrap_greens_forward!(Gdn, Bdn[l], A)

        ## Get the HS fields associated with the current imaginary time-slice.
        s_l = @view s[:,l]

        ## Perform local updates HS fields associated with the current imaginary time slice.
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, acceptance_rate_l) = _local_update!(
            Gup, logdetGup, sgndetGup, Bup[l], Gdn, logdetGdn, sgndetGdn, Bdn[l],
            s_l, μ, α, Δτ, rng, perm, u, v
        )

        ## Record the acceptance rate
        acceptance_rate += acceptance_rate_l / Lτ

        ## If using symmetric propagator definition (symmetric = true), then apply
        ## the transformation G̃ ==> G = exp{-Δτ⋅K/2}⋅G̃⋅exp{+Δτ⋅K/2}.
        ## If asymmetric propagator definition is used (symmetric = false),
        ## then this does nothing.
        jdqmcf.partially_wrap_greens_reverse!(Gup, Bup[l], A)
        jdqmcf.partially_wrap_greens_reverse!(Gdn, Bdn[l], A)

        ## Periodically re-calculate the Green's function matrix for numerical stability.
        logdetGup, sgndetGup, δGup, δθup = jdqmcf.stabilize_equaltime_greens!(
            Gup, logdetGup, sgndetGup, fermion_greens_calc_up, Bup, update_B̄ = true
        )
        logdetGdn, sgndetGdn, δGdn, δθdn = jdqmcf.stabilize_equaltime_greens!(
            Gdn, logdetGdn, sgndetGdn, fermion_greens_calc_dn, Bdn, update_B̄ = true
        )

        ## Record largest numerical error corrected by numerical stabilization.
        δG = max(δG, δGup, δGdn)

        ## Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fermion_greens_calc_dn, fermion_greens_calc_up.forward)
    end

    return logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, acceptance_rate
end

## Iterate over all sites for single imaginary time-slice, attempting a local
## update to each corresponding Ising HS field.
function _local_update!(
    Gup, logdetGup, sgndetGup, Bup, Gdn, logdetGdn, sgndetGdn, Bdn,
    s, μ, α, Δτ, rng, perm, u, v
)

    ## Randomize the order in which the sites are iterated over.
    shuffle!(rng, perm)

    ## Counter for number of accepted updates.
    accepted = 0

    ## Iterate over sites in lattice.
    for i in perm

        ## Calculate the new value of the diagonal potential energy matrix element
        ## assuming the sign of the Ising HS field is changed.
        Vup′ = -α/Δτ * (-s[i]) - μ
        Vdn′ = +α/Δτ * (-s[i]) - μ

        ## Calculate the determinant ratio associated with the proposed update.
        Rup, Δup = jdqmcf.local_update_det_ratio(Gup, Bup, Vup′, i, Δτ)
        Rdn, Δdn = jdqmcf.local_update_det_ratio(Gdn, Bdn, Vdn′, i, Δτ)

        ## Calculate the acceptance probability based on the Metropolis accept/reject criteria.
        P = min(1.0, abs(Rup * Rdn))

        ## Randomly Accept or reject the proposed update with the specified probability.
        if rand(rng) < P

            ## Increment the accepted update counter.
            accepted += 1

            ## Flip the appropriate Ising HS field.
            s[i] = -s[i]

            ## Update the Green's function matrices.
            logdetGup, sgndetGup = jdqmcf.local_update_greens!(
                Gup, logdetGup, sgndetGup, Bup, Rup, Δup, i, u, v
            )
            logdetGdn, sgndetGdn = jdqmcf.local_update_greens!(
                Gdn, logdetGdn, sgndetGdn, Bdn, Rdn, Δdn, i, u, v
            )
        end
    end

    ## Calculate the acceptance rate.
    acceptance_rate = accepted / N

    return logdetGup, sgndetGup, logdetGdn, sgndetGdn, acceptance_rate
end;

# Next we implement a function to make measurements during the simulation, including time-displaced measurements.
# Note that if we want to calculate the expectation value for some observable ``\langle \mathcal{O} \rangle``,
# then during the simulation we actually measure ``\langle \mathcal{S O} \rangle_{\mathcal{W}}``, where
# ``\langle \bullet \rangle_{\mathcal{W}}`` denotes an average with respect to states sampled according
# to the DQMC weights
# ```math
# \mathcal{W} = | \det G_\uparrow^{-1} \det G_\downarrow^{-1} |,
# ```
# such that
# ```math
# \mathcal{S} = \text{sign}(\det G_\uparrow^{-1} \det G_\downarrow^{-1})
# ```
# is the sign associated with each state. The reweighting method is then used at the end of a simulation
# to recover the correct expectation value according to
# ```math
# \langle \mathcal{O} \rangle = \frac{ \langle \mathcal{SO} \rangle_{\mathcal{W}} }{ \langle \mathcal{S} \rangle_{\mathcal{W}} },
# ```
# where ``\langle \mathcal{S} \rangle_{\mathcal{W}}`` is the average sign measured over the course of the simulation.

## Make measurements.
function make_measurements!(
    Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
    Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    bin, avg_sign, density, double_occ, C_greens, C_spinz, C_density,
    C_loc_swave, C_ext_swave, C_dwave
)


    ## Initialize time-displaced Green's function matrices for both spin species:
    ## G(τ=0,τ=0) = G(0,0)
    ## G(τ=0,0)   = G(0,0)
    ## G(0,τ=0)   = -(I-G(0,0))
    jdqmcf.initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup)
    jdqmcf.initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn)

    ## Calculate the current sign.
    sgn = sign(sgndetGup * sgndetGdn)

    ## Measure the average sign.
    avg_sign[bin] += sgn

    ## Measure the density.
    nup = jdqmcm.measure_n(Gup)
    ndn = jdqmcm.measure_n(Gdn)
    density[bin] += sgn * (nup + ndn)

    ## Measure the double occupancy.
    double_occ[bin] += sgn * jdqmcm.measure_double_occ(Gup, Gdn)

    ## Measure equal-time correlation functions.
    make_correlation_measurements!(
        Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
        unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
        bin, 0, sgn, C_greens, C_spinz, C_density, C_loc_swave, C_ext_swave, C_dwave
    )

    ## Iterate over imaginary time slices.
    for l in fermion_greens_calc_up

        ## Propagate equal-time Green's function matrix to current imaginary time
        ## G(τ±Δτ,τ±Δτ) ==> G(τ,τ) depending on whether iterating over imaginary
        ## time in the forward or reverse direction
        jdqmcf.propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fermion_greens_calc_up, Bup)
        jdqmcf.propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fermion_greens_calc_dn, Bdn)

        ## Measure time-displaced correlation function measurements for τ = l⋅Δτ.
        make_correlation_measurements!(
            Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
            bin, l, sgn, C_greens, C_spinz, C_density, C_loc_swave, C_ext_swave, C_dwave,
        )

        ## Periodically re-calculate the Green's function matrix for numerical stability.
        logdetGup, sgndetGup, δGup, δθup = jdqmcf.stabilize_unequaltime_greens!(
            Gup_τ0, Gup_0τ, Gup_ττ, logdetGup, sgndetGup, fermion_greens_calc_up, Bup, update_B̄=false
        )
        logdetGdn, sgndetGdn, δGdn, δθdn = jdqmcf.stabilize_unequaltime_greens!(
            Gdn_τ0, Gdn_0τ, Gdn_ττ, logdetGdn, sgndetGdn, fermion_greens_calc_dn, Bdn, update_B̄=false
        )

        ## Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fermion_greens_calc_dn, fermion_greens_calc_up.forward)
    end

    return nothing
end

## Make time-displaced measurements.
function make_correlation_measurements!(
    Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    bin, l, sgn, C_greens, C_spinz, C_density, C_loc_swave, C_ext_swave, C_dwave,
    tmp = zeros(eltype(C_greens), lattice.L...)
)

    ## Get a view into the arrays accumulating the correlation measurements
    ## for the current imaginary time-slice and bin.
    C_greens_bin_l  = @view C_greens[bin,:,:,l+1]
    C_spinz_bin_l = @view C_spinz[bin,:,:,l+1]
    C_density_bin_l = @view C_density[bin,:,:,l+1]
    C_loc_swave_bin_l = @view C_loc_swave[bin,:,:,l+1]
    C_ext_swave_bin_l = @view C_ext_swave[bin,:,:,l+1]
    C_dwave_bin_l = @view C_dwave[bin,:,:,l+1]

    ## Measure Green's function for both spin-up and spin-down.
    jdqmcm.greens!(C_greens_bin_l, 1, 1, unit_cell, lattice, Gup_τ0, sgn)
    jdqmcm.greens!(C_greens_bin_l, 1, 1, unit_cell, lattice, Gdn_τ0, sgn)

    ## Measure spin-z spin-spin correlation.
    jdqmcm.spin_z_correlation!(
        C_spinz_bin_l, 1, 1, unit_cell, lattice,
        Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn
    )

    ## Measure density-density correlation.
    jdqmcm.density_correlation!(
        C_density_bin_l, 1, 1, unit_cell, lattice,
        Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn
    )

    ## Measure local s-wave correlation measurement.
    jdqmcm.pair_correlation!(
        C_loc_swave_bin_l, bond_trivial, bond_trivial, unit_cell, lattice, Gup_τ0, Gdn_τ0, sgn
    )

    ## Group the nearest-neighbor bonds together.
    bonds = (bond_px, bond_nx, bond_py, bond_ny)

    ## d-wave correlation phases.
    dwave_phases = (+1, +1, -1, -1)

    ## Iterate over all pairs of nearest-neigbbor bonds.
    for i in eachindex(bonds)
        for j in eachindex(bonds)
            ## Measure pair correlation associated with bond pair.
            fill!(tmp, 0)
            jdqmcm.pair_correlation!(
                tmp, bonds[i], bonds[j], unit_cell, lattice, Gup_τ0, Gdn_τ0, sgn
            )
            ## Add contribution to extended s-wave and d-wave pair correlation.
            @. C_ext_swave_bin_l += tmp / 4
            @. C_dwave_bin_l += dwave_phases[i] * dwave_phases[j] * tmp / 4
        end
    end

    return nothing
end;

# Now we will write a top-level function to run the simulation, including both the thermalization
# and measurement portions of the simulation.

## High-level function to run the DQMC simulation.
function run_simulation!(
    s, μ, α, Δτ, rng, N_burnin, N_bins, N_binsize, N_sweeps,
    Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
    Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    avg_sign, density, double_occ, C_greens, C_spinz, C_density,
    C_loc_swave, C_ext_swave, C_dwave
)

    ## Initialize variable to keep track of largest corrected numerical error.
    δG = 0.0

    ## The acceptance rate on local updates.
    acceptance_rate = 0.0

#jl     println("Beginning Thermalization Updates")

    ## Perform burnin updates to thermalize system.
    for n in 1:N_burnin

        ## Attempt local update to every Ising HS field.
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG′, ac) = local_update!(
            Gup, logdetGup, sgndetGup, Bup, fermion_greens_calc_up,
            Gdn, logdetGdn, sgndetGdn, Bdn, fermion_greens_calc_dn,
            s, μ, α, Δτ, δG, rng
        )

        ## Record max numerical error.
        δG = max(δG, δG′)

        ## Update acceptance rate.
        acceptance_rate += ac
    end

#jl     println()
#jl     println("Beginning Measurements.")

    ## Iterate over measurement bins.
    for bin in 1:N_bins

        ## Iterate over updates and measurements in bin.
        for n in 1:N_binsize

            ## Iterate over number of local update sweeps per measurement.
            for sweep in 1:N_sweeps
                ## Attempt local update to every Ising HS field.
                (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG′, ac) = local_update!(
                    Gup, logdetGup, sgndetGup, Bup, fermion_greens_calc_up,
                    Gdn, logdetGdn, sgndetGdn, Bdn, fermion_greens_calc_dn,
                    s, μ, α, Δτ, δG, rng
                )

                ## Record max numerical error.
                δG = max(δG, δG′)

                ## Update acceptance rate.
                acceptance_rate += ac
            end

            ## Make measurements.
            make_measurements!(
                Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
                Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
                unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
                bin, avg_sign, density, double_occ, C_greens, C_spinz, C_density,
                C_loc_swave, C_ext_swave, C_dwave
            )
        end

        ## Normalize accumulated measurements by the bin size.
        avg_sign[bin] /= N_binsize
        density[bin] /= N_binsize
        double_occ[bin] /= N_binsize
        C_greens[bin,:,:,:] /= (2 * N_binsize)
        C_spinz[bin,:,:,:] /= N_binsize
        C_density[bin,:,:,:] /= N_binsize
        C_loc_swave[bin,:,:,:] /= N_binsize
        C_ext_swave[bin,:,:,:] /= N_binsize
        C_dwave[bin,:,:,:] /= N_binsize
    end

    ## Calculate the final acceptance rate for local updates.
    acceptance_rate /= (N_burnin + N_bins * N_binsize * N_sweeps)

#jl     println()
#jl     println("Simuilation Complete.")
#jl     println()

    return acceptance_rate, δG
end;

# Now let us run our DQMC simulation.

## Run the DQMC simulation.
acceptance_rate, δG = run_simulation!(
    s, μ, α, Δτ, rng, N_burnin, N_bins, N_binsize, N_sweeps,
    Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
    Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    avg_sign, density, double_occ, C_greens, C_spinz, C_density,
    C_loc_swave, C_ext_swave, C_dwave
)
println("Acceptance Rate = ", acceptance_rate)
println("Largest Numerical Error = ", δG)

# Having completed the DQMC simulation, the next step is the analyze the results,
# calculating the mean and error for various measuremed observables.
# We will first calculate the relevant global measurements, including the 
# average density ``\langle n \rangle = \langle n_\uparrow + n_\downarrow \rangle``
# and double occupancy ``\langle n_\uparrow n_\downarrow \rangle.``
# Note that the binning method is used to calculate the error bar for the correlated data.
# The Jackknife algorithm is also used to propagate error and correct for bias when evaluating
# ```math
# \langle \mathcal{O} \rangle = \frac{ \langle \mathcal{S O} \rangle_\mathcal{W} }{ \langle \mathcal{S} \rangle_\mathcal{W} }
# ```
# to account for the sign problem.

## Calculate the average sign for the simulation.
sign_avg, sign_std = jdqmcm.jackknife(identity, avg_sign)
println("Avg Sign, S = ", real(sign_avg), " +/- ", sign_std)

## Calculate the average density.
density_avg, density_std = jdqmcm.jackknife(/, density, avg_sign)
println("Density, n = ", real(density_avg), " +/- ", density_std)

## Calculate the average double occupancy.
double_occ_avg, double_occ_std = jdqmcm.jackknife(/, double_occ, avg_sign)
println("Double occupancy, nup_ndn = ", real(double_occ_avg), " +/- ", double_occ_std)

# Now we move onto processing the measured correlation function data.
# We define two functions to assist with this process.
# The first function integrates the binned time-displaced correlation function data
# over the imaginary time axis in order to generate binned susceptibility data.
# Note that the integration over the imaginary time axis is performed using Simpson's
# rule, which is accurate to order ``\mathcal{O}(\Delta\tau^4)``.

## Given the binned time-displaced correlation function/structure factor data,
## calculate and return the corresponding binned susceptibility data.
function susceptibility(S::AbstractArray{T}, Δτ) where {T<:Number}

    ## Allocate array to contain susceptibility.
    χ = zeros(T, size(S)[1:3])

    ## Iterate over bins.
    for bin in axes(S,1)

        ## Calculate the susceptibility for the current bin by integrating the correlation
        ## data over the imaginary time axis using Simpson's rule.
        S_bin = @view S[bin,:,:,:]
        χ_bin = @view χ[bin,:,:]
        jdqmcm.susceptibility!(χ_bin, S_bin, Δτ, 3)
    end

    return χ
end

# We also define a function to calculate the average and error of a correlation function
# measurement based on the input binned correlation function data.

## Calculate average correlation function values based on binned data.
function correlation_stats(
    S::AbstractArray{Complex{T}},
    avg_sign::Vector{E}
) where {T<:AbstractFloat, E<:Number}

    ## Allocate arrays to contain the mean and standard deviation of
    ## measured correlation function.
    S_avg = zeros(Complex{T}, size(S)[2:end])
    S_std = zeros(T, size(S)[2:end])

    ## Number of bins.
    N_bins = length(avg_sign)

    ## Preallocate arrays to make the jackknife error analysis faster.
    jackknife_samples = (zeros(Complex{T}, N_bins), zeros(E, N_bins))
    jackknife_g       = zeros(Complex{T}, N_bins)

    ## Iterate over correlation functions.
    for n in CartesianIndices(S_avg)
        ## Use the jackknife method to calculage average and error.
        vals = @view S[:,n]
        S_avg[n], S_std[n] = jdqmcm.jackknife(
            /, vals, avg_sign,
            jackknife_samples = jackknife_samples,
            jackknife_g = jackknife_g
        )
    end

    return S_avg, S_std
end

# First, let us compute the average and error for the time-displaced electron
# Green's function
# ```math
# G_\sigma(\mathbf{r},\tau) = \langle \hat{c}^{\phantom \dagger}_{\sigma,\mathbf{i}+\mathbf{r}}(\tau) \hat{c}^\dagger_{\sigma,\mathbf{i}}(0) \rangle
# ```
# in position space, and
# ```math
# G_\sigma(\mathbf{k},\tau) = \langle \hat{c}^{\phantom \dagger}_{\sigma,\mathbf{k}}(\tau) \hat{c}^\dagger_{\sigma,\mathbf{k}}(0) \rangle
# ```
# in momentum space, where ``\tau \in [0, \Delta\tau, \dots, \beta-\Delta\tau, \beta].``

## Fourier transform Green's function from position to momentum space.
S_greens = copy(C_greens)
jdqmcm.fourier_transform!(S_greens, 1, 1, (1,4), unit_cell, lattice)

## Calculate average Green's function in position space.
C_greens_avg, C_greens_std = correlation_stats(C_greens, avg_sign)

## Calculate average Green's function in momentum space.
S_greens_avg, S_greens_std = correlation_stats(S_greens, avg_sign)

## Verify that the position space G(r=0,τ=0) measurement agrees with the
## average density measurement.
agreement = (2*(1-C_greens_avg[1,1,1]) ≈ density_avg)
#md println("(2*[1-G(r=0,tau=0)] == <n>) = ", agreement)
#nb println("(2*[1-G(r=0,tau=0)] == <n>) = ", agreement)

# Now we will calculate the spin susceptibility
# ```math
# \chi_z(\mathbf{q}) = \int_0^\beta S_z(\mathbf{q},\tau) \ d\tau
# ```
# where the time-displaced spin structure
# ```math
# S_z(\mathbf{q},\tau) = \sum_\mathbf{r} e^{-{\rm i} \mathbf{q}\cdot\mathbf{r}} \ C_z(\mathbf{r},\tau)
# ```
# is given by the fourier transform of the spin-``z`` correlation function
# ```math
# C_z(\mathbf{r},\tau) = \frac{1}{N} \sum_\mathbf{i} \langle \hat{S}_{z,\mathbf{i}+\mathbf{r}}(\tau) \hat{S}_{z,\mathbf{i}}(0) \rangle
# ```
# in position space. Then we report the spin-suscpetibility ``\chi_{\rm afm} = \chi_z(\pi,\pi)`` corresponding to
# antiferromagnetism.

## Fourier transform the binned Cz(r,τ) position space spin-z correlation function
## data to get the binned Sz(q,τ) spin-z structure factor data.
S_spinz = copy(C_spinz)
jdqmcm.fourier_transform!(S_spinz, 1, 1, (1,4), unit_cell, lattice)

## Integrate the binned Sz(q,τ) spin-z structure factor data over the imaginary
## time axis to get the binned χz(q) spin susceptibility.
χ_spinz = susceptibility(S_spinz, Δτ)

## Calculate the average spin correlation functions in position space.
C_spinz_avg, C_spinz_std = correlation_stats(C_spinz, avg_sign)

## Calculate the average spin structure factor in momentum space.
S_spinz_avg, S_spinz_std = correlation_stats(S_spinz, avg_sign)

## Calculate the average spin susceptibility for all scattering momentum q.
χ_spinz_avg, χ_spinz_std = correlation_stats(χ_spinz, avg_sign)

## Report the spin susceptibility χafm = χz(π,π) corresponding to antiferromagnetism.
χafm_avg = real(χ_spinz_avg[L÷2+1, L÷2+1])
χafm_std = χ_spinz_std[L÷2+1, L÷2+1]
println("Antiferromagentic Spin Susceptibility, chi_afm = ", χafm_avg, " +/- ", χafm_std)

# Given the measured time-displaced density correlation function
# ```math
# C_{\rho}(\mathbf{r},\tau) = \sum_{\mathbf{i}}
#   \langle \hat{n}_{\mathbf{i}+\mathbf{r}}(\tau) \hat{n}_{\mathbf{i}}(0) \rangle,
# ```
# where ``\hat{n}_{\mathbf{i}} = (\hat{n}_{\mathbf{i}, \uparrow} + \hat{n}_{\mathbf{i}, \downarrow}),``
# we can compute the time-displaced charge structure factor
# ```math
# S_{\rho}(\mathbf{q},\tau) = \sum_\mathbf{r} e^{-{\rm i}\mathbf{q}\cdot\mathbf{r}} \ C_{\rho}(\mathbf{r},\tau)
# ```
# and corresponding charge susceptibility
# ```math
# \chi_{\rho}(\mathbf{q}) \int_0^\beta S_{\rho}(\mathbf{q},\tau) \ d\tau.
# ```

## Fourier transform the binned Cρ(r,τ) position space density correlation
## data to get the time-dispaced charge structure factor Sρ(q,τ) in
## momentum space.
S_density = copy(C_density)
jdqmcm.fourier_transform!(S_density, 1, 1, (1,4), unit_cell, lattice)

## Integrate the binned Sρ(q,τ) density structure factor data over the imaginary
## time axis to get the binned χρ(q) density susceptibility.
χ_density = susceptibility(S_density, Δτ)

## Calculate the average charge correlation functions in position space.
C_density_avg, C_density_std = correlation_stats(C_density, avg_sign)

## Calculate the average charge structure factor in momentum space.
S_density_avg, S_density_std = correlation_stats(S_density, avg_sign)

## Calculate the average charge susceptibility for all scattering momentum q.
χ_density_avg, χ_density_std = correlation_stats(χ_spinz, avg_sign);

# Now we calculate the local s-wave pair susceptibility
# ```math
# P_{s} = \frac{1}{N} \int_0^\beta \langle \hat{\Delta}_{s}(\tau) \hat{\Delta}_{s}(0) \rangle \ d\tau,
# ```
# where ``\hat{\Delta}_{s} = \sum_\mathbf{i} \hat{c}_{\downarrow,\mathbf{i}} \hat{c}_{\uparrow,\mathbf{i}}.``

## Fourier transform binned position space local s-wave correlation function data to get
## the binned momentum space local s-wave structure factor data.
S_loc_swave = copy(C_loc_swave)
jdqmcm.fourier_transform!(S_loc_swave, 1, 1, (1,4), unit_cell, lattice)

## Integrate the binned local s-wave structure factor data to get the
## binned local s-wave pair susceptibility data.
P_loc_swave = susceptibility(S_loc_swave, Δτ)

## Calculate the average local s-wave pair susceptibility for all scattering momentum q.
P_loc_swave_avg, P_loc_swave_std = correlation_stats(P_loc_swave, avg_sign)

## Report the local s-wave pair suspcetibility.
Ps_avg = real(P_loc_swave_avg[1,1])
Ps_std = P_loc_swave_std[1,1]
println("Local s-wave pair susceptibility, P_s = ", Ps_avg, " +/- ", Ps_std)

# Next, we calculate the local s-wave pair susceptibility
# ```math
# P_{\textrm{ext-}s} = \frac{1}{N} \int_0^\beta \langle \hat{\Delta}_{\textrm{ext-}s}(\tau) \hat{\Delta}_{\textrm{ext-}s}(0) \rangle \ d\tau,
# ```
# where
# ```math
# \hat{\Delta}_{\textrm{ext-}s} = \frac{1}{2} \sum_\mathbf{i}
#   (\hat{c}_{\downarrow,\mathbf{i}+\mathbf{x}}
#   +\hat{c}_{\downarrow,\mathbf{i}-\mathbf{x}}
#   +\hat{c}_{\downarrow,\mathbf{i}+\mathbf{y}}
#   +\hat{c}_{\downarrow,\mathbf{i}-\mathbf{y}})
#   \hat{c}_{\uparrow,\mathbf{i}}.
# ```

## Fourier transform binned position space extended s-wave correlation function data to get
## the binned momentum space extended s-wave structure factor data.
S_ext_swave = copy(C_ext_swave)
jdqmcm.fourier_transform!(S_ext_swave, 1, 1, (1,4), unit_cell, lattice)

## Integrate the binned extended s-wave structure factor data to get the
## binned extended s-wave pair susceptibility data.
P_ext_swave = susceptibility(S_ext_swave, Δτ)

## Calculate the average extended s-wave pair susceptibility for all scattering momentum q.
P_ext_swave_avg, P_ext_swave_std = correlation_stats(P_ext_swave, avg_sign)

## Report the local s-wave pair suspcetibility.
Pexts_avg = real(P_ext_swave_avg[1,1])
Pexts_std = P_ext_swave_std[1,1]
println("Extended s-wave pair susceptibility, P_ext-s = ", Pexts_avg, " +/- ", Pexts_std)

# Lastly, we calculate the d-wave pair susceptibility
# ```math
# P_{d} = \frac{1}{N} \int_0^\beta \langle \hat{\Delta}_{d}(\tau) \hat{\Delta}_{d}(0) \rangle \ d\tau,
# ```
# where
# ```math
# \hat{\Delta}_{d} = \frac{1}{2} \sum_\mathbf{i}
#   (\hat{c}_{\downarrow,\mathbf{i}+\mathbf{x}}
#   +\hat{c}_{\downarrow,\mathbf{i}-\mathbf{x}}
#   -\hat{c}_{\downarrow,\mathbf{i}+\mathbf{y}}
#   -\hat{c}_{\downarrow,\mathbf{i}-\mathbf{y}})
#   \hat{c}_{\uparrow,\mathbf{i}}.
# ```

## Fourier transform binned position space d-wave correlation function data to get
## the binned momentum space d-wave structure factor data.
S_dwave = copy(C_dwave)
jdqmcm.fourier_transform!(S_dwave, 1, 1, (1,4), unit_cell, lattice)

## Integrate the binned d-wave structure factor data to get the
## binned d-wave pair susceptibility data.
P_dwave = susceptibility(S_dwave, Δτ)

## Calculate the average d-wave pair susceptibility for all scattering momentum q.
P_dwave_avg, P_dwave_std = correlation_stats(P_dwave, avg_sign)

## Report the d-wave pair susceptibility.
Pd_avg = real(P_dwave_avg[1,1])
Pd_std = P_dwave_std[1,1]
println("Extended d-wave pair susceptibility, P_d = ", Pd_avg, " +/- ", Pd_std)