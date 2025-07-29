# Import relevant standard template libraries.
using Random
using LinearAlgebra

# Provides framework for implementing DQMC code.
import JDQMCFramework as jdqmcf

# Exports methods for measuring various correlation functions in a DQMC simulation.
import JDQMCMeasurements as jdqmcm

# Exports types and methods for representing lattice geometries.
import LatticeUtilities as lu

# Exports the checkerboard approximation for representing an exponentiated hopping matrix.
import Checkerboard as cb

# Package for performing Fast Fourier Transforms (FFTs).
using FFTW

# Set number of threads used by BLAS/LAPACK to one.
BLAS.set_num_threads(1)

# Set number of threads used by FFTW to one.
FFTW.set_num_threads(1)

# Nearest-neighbor hopping amplitude.
t = 1.0
println("Nearest-neighbor hopping amplitude, t = ", t)

# Hubbard interaction.
U = 6.0
println("Hubbard interaction, U = ", U)

# Chemical potential.
μ = 2.0
println("Chemical potential, mu = ", μ)

# Inverse temperature.
β = 4.0
println("Inverse temperature, beta = ", β)

# Lattice size.
L = 4
println("Linear lattice size, L = ", L)
println()

# Discretization in imaginary time.
Δτ = 0.05
println("Disretization in imaginary time, dtau = ", Δτ)

# Length of imaginary time axis.
Lτ = round(Int, β/Δτ)
println("Length of imaginary time axis, Ltau = ", Lτ)

# Whether or not to use a symmetric or asymmetric definition for the propagator matrices.
symmetric = false
println("Whether symmetric or asymmetric propagator matrices are used, symmetric = ", symmetric)

# Whether or not to use the checkerboard approximation to represent the
# exponentiated electron kinetic energy matrix exp(-Δτ⋅K).
checkerboard = false
println("Whether the checkerboard approximation is used, checkerboard = ", checkerboard)

# Period with which numerical stabilization is performed i.e.
# how many imaginary time slices separate more expensive recomputations
# of the Green's function matrix using numerically stable routines.
n_stab = 10
println("Numerical stabilization period, n_stab = ", n_stab)

# The number of burnin sweeps through the lattice performing local updates that
# are performed to thermalize the system.
N_burnin = 2_500
println("Number of burnin sweeps, N_burnin = ", N_burnin)

# The number of measurements made once the system is thermalized.
N_measurements = 10_000
println("Number of measurements, N_measurements = ", N_measurements)

# Number of local update sweeps separating sequential measurements.
N_sweeps = 1
println("Number of local update sweeps seperating measurements, n_sweeps = ", N_sweeps)

# Number of bins used to performing a binning analysis when calculating final error bars
# for measured observables.
N_bins = 50
println("Number of measurement bins, N_bins = ", N_bins)

# Number of measurements averaged over per measurement bin.
N_binsize = N_measurements ÷ N_bins
println("Number of measurements per bin, N_binsize = ", N_binsize)
println()

# Initialize random number generator.
seed = abs(rand(Int))
rng = Xoshiro(seed)
println("Random seed used to initialize RNG, seed = ", seed)
println()

# Define the square lattice unit cell.
unit_cell = lu.UnitCell(
    lattice_vecs = [[1.0, 0.0],
                    [0.0, 1.0]],
    basis_vecs   = [[0.0, 0.0]]
)

# Define the size of the periodic square lattice.
lattice = lu.Lattice(
    L = [L, L],
    periodic = [true, true]
)

# Define nearest-neighbor bond in +x direction
bond_px = lu.Bond(
    orbitals = (1,1),
    displacement = [1,0]
)

# Define nearest-neighbor bond in +y direction
bond_py = lu.Bond(
    orbitals = (1,1),
    displacement = [0,1]
)

# Build the neighbor table corresponding to all nearest-neighbor bonds.
neighbor_table = lu.build_neighbor_table([bond_px, bond_py], unit_cell, lattice)
println("The neighbor table, neighbor_table =")
show(stdout, "text/plain", neighbor_table)
println("\n")

# The total number of sites/orbitals in the lattice.
N = lu.nsites(unit_cell, lattice) # For square lattice this is simply N = L^2

# Total number of bonds in lattice.
N_bonds = size(neighbor_table, 2)

# Define a "trivial" bond that maps a site back onto itself.
bond_trivial = lu.Bond(
    orbitals = (1,1),
    displacement = [0,0]
)

# Define bond in -x direction.
bond_nx = lu.Bond(
    orbitals = (1,1),
    displacement = [-1,0]
)

# Define bond in -y direction.
bond_ny = lu.Bond(
    orbitals = (1,1),
    displacement = [0,-1]
);

# Define Δτ′=Δτ/2 if symmetric = true, otherwise Δτ′=Δτ
Δτ′ = symmetric ? Δτ/2 : Δτ

# If the matrix exp(Δτ′⋅K) is represented by the checkerboard approximation.
if checkerboard

    # Construct the checkerboard approximation to the matrix exp(-Δτ′⋅K).
    expnΔτ′K = cb.CheckerboardMatrix(neighbor_table, fill(t, N_bonds), Δτ′)

# If the matrix exp(Δτ′⋅K) is NOT represented by the checkerboard approximation.
else

    # Construct the electron kinetic energy matrix.
    K = zeros(typeof(t), N, N)
    for bond in 1:N_bonds
        i, j = neighbor_table[1, bond], neighbor_table[2, bond]
        K[i,j] = -t
        K[j,i] = -conj(t)
    end

    # Calculate the exponentiated kinetic energy matrix, exp(-Δτ⋅K).
    # Note that behind the scenes Julia is diagonalizing the matrix K in order to exponentiate it.
    expnΔτ′K = exp(-Δτ′*K)

    # Calculate the inverse of the exponentiated kinetic energy matrix, exp(+Δτ⋅K).
    exppΔτ′K = exp(+Δτ′*K)
end;

# Define constant associated Ising Hubbard-Stratonovich (HS) transformation.
α = acosh(exp(Δτ*U/2))

# Initialize a random Ising HS configuration.
s = rand(rng, -1:2:1, N, Lτ)

# Matrix element type for exponentiated electron kinetic energy matrix exp{-Δτ′⋅K}
T_expnΔτK = eltype(t)

# Matrix element type for diagonal exponentiated electron potential energy matrix exp{-Δτ⋅V[σ,l]}
T_expnΔτV = typeof(α)

# Initialize empty vector to contain propagator matrices for each imaginary time slice.
if checkerboard && symmetric

    # Propagator defined as B[σ,l] = exp{-Δτ⋅K/2}⋅exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K/2},
    # where the dense matrix exp{-Δτ⋅K/2} is approximated by the sparse checkerboard matrix.
    Bup = jdqmcf.SymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.SymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]

elseif checkerboard && !symmetric

    # Propagator defined as B[σ,l] = exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K},
    # where the dense matrix exp{-Δτ⋅K} is approximated by the sparse checkerboard matrix.
    Bup = jdqmcf.AsymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.AsymChkbrdPropagator{T_expnΔτK, T_expnΔτV}[]

elseif !checkerboard && symmetric

    # Propagator defined as B[σ,l] = exp{-Δτ⋅K/2}⋅exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K/2},
    # where the dense matrix exp{-Δτ⋅K/2} is exactly calculated.
    Bup = jdqmcf.SymExactPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.SymExactPropagator{T_expnΔτK, T_expnΔτV}[]

elseif !checkerboard && !symmetric

    # Propagator defined as B[σ,l] = exp{-Δτ⋅V[σ,l]}⋅exp{-Δτ⋅K},
    # where the dense matrix exp{-Δτ⋅K} is exactly calculated.
    Bup = jdqmcf.AsymExactPropagator{T_expnΔτK, T_expnΔτV}[]
    Bdn = jdqmcf.AsymExactPropagator{T_expnΔτK, T_expnΔτV}[]
end;

# Iterate over time-slices.
for l in 1:Lτ

    # Get the HS fields associated with the current time-slice l.
    s_l = @view s[:,l]

    # Calculate the spin-up diagonal exponentiated potential energy
    # matrix exp{-Δτ⋅V[↑,l]} = exp{-Δτ⋅(-α/Δτ⋅s[i,l]-μ)} = exp{+α⋅s[i,l] + Δτ⋅μ}.
    expnΔτVup = zeros(T_expnΔτV, N)
    @. expnΔτVup = exp(+α * s_l + Δτ*μ)

    # Calculate the spin-down diagonal exponentiated potential energy
    # matrix exp{-Δτ⋅V[↓,l]} = exp{-Δτ⋅(+α/Δτ⋅s[i,l]-μ)} = exp{-α⋅s[i,l] + Δτ⋅μ}.
    expnΔτVdn = zeros(T_expnΔτV, N)
    @. expnΔτVdn = exp(-α * s_l + Δτ*μ)

    # Initialize spin-up and spin-down propagator matrix for the current time-slice l.
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

# Initialize a FermionGreensCalculator for both spin up and down electrons.
fermion_greens_calc_up = jdqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
fermion_greens_calc_dn = jdqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab);

# Calculate spin-up equal-time Green's function matrix.
Gup = zeros(typeof(t), N, N)
logdetGup, sgndetGup = jdqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calc_up)

# Calculate spin-down equal-time Green's function matrix.
Gdn = zeros(typeof(t), N, N)
logdetGdn, sgndetGdn = jdqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calc_dn);

# Allcoate time-displaced Green's functions.
Gup_τ0 = zero(Gup) # Gup(τ,0)
Gup_0τ = zero(Gup) # Gup(0,τ)
Gup_ττ = zero(Gup) # Gup(τ,τ)
Gdn_τ0 = zero(Gdn) # Gdn(τ,0)
Gdn_0τ = zero(Gdn) # Gdn(0,τ)
Gdn_ττ = zero(Gdn); # Gdn(τ,τ)

# Vector to contain binned average sign measurement.
avg_sign = zeros(eltype(Gup), N_bins)

# Vector to contain binned density measurement.
density = zeros(eltype(Gup), N_bins)

# Vector to contain binned double occupancy measurement.
double_occ = zeros(eltype(Gup), N_bins)

# Array to contain binned position-space time-displaced Green's function measurements.
C_greens = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

# Array to contain binned position-space time-displaced Spin-Z correlation function measurements.
C_spinz = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

# Array to contain binned position-space time-displaced density correlation function measurements.
C_density = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

# Array to contain binned position-space local s-wave pair correlation function.
C_loc_swave = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

# Array to contain binned position-space extended s-wave pair correlation function.
C_ext_swave = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

# Array to contain binned position-space d-wave pair correlation function.
C_dwave = zeros(Complex{Float64}, N_bins, L, L, Lτ+1)

# Array to contain binned momentum-space d-wave pair susceptibility.
P_d_q = zeros(Complex{Float64}, N_bins, L, L);

# Function to perform local updates to all Ising HS fields.
function local_update!(
    Gup::Matrix{T}, logdetGup, sgndetGup, Bup, fermion_greens_calc_up,
    Gdn::Matrix{T}, logdetGdn, sgndetGdn, Bdn, fermion_greens_calc_dn,
    s, μ, α, Δτ, δG, rng
) where {T<:Number}

    # Length of imaginary time axis.
    Lτ = length(Bup)

    # Number of sites in lattice.
    N = size(Gup,1)

    # Allocate temporary arrays that will be used to avoid dynamic memory allocation.
    A = zeros(T, N, N)
    u = zeros(T, N)
    v = zeros(T, N)

    # Allocate vector of integers to contain random permutation specifying the order in which
    # sites are iterated over at each imaginary time slice when performing local updates.
    perm = collect(1:size(Gup,1))

    # Variable to keep track of the acceptance rate.
    acceptance_rate = 0.0

    # Iterate over imaginary time slices.
    for l in fermion_greens_calc_up

        # Propagate equal-time Green's function matrix to current imaginary time
        # G(τ±Δτ,τ±Δτ) ==> G(τ,τ) depending on whether iterating over imaginary
        # time in the forward or reverse direction
        jdqmcf.propagate_equaltime_greens!(Gup, fermion_greens_calc_up, Bup)
        jdqmcf.propagate_equaltime_greens!(Gdn, fermion_greens_calc_dn, Bdn)

        # If using symmetric propagator definition (symmetric = true), then apply
        # the transformation G ==> G̃ = exp{+Δτ⋅K/2}⋅G⋅exp{-Δτ⋅K/2}.
        # If asymmetric propagator definition is used (symmetric = false),
        # then this does nothing.
        jdqmcf.partially_wrap_greens_reverse!(Gup, Bup[l], A)
        jdqmcf.partially_wrap_greens_reverse!(Gdn, Bdn[l], A)

        # Get the HS fields associated with the current imaginary time-slice.
        s_l = @view s[:,l]

        # Perform local updates HS fields associated with the current imaginary time slice.
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, acceptance_rate_l) = _local_update!(
            Gup, logdetGup, sgndetGup, Bup[l], Gdn, logdetGdn, sgndetGdn, Bdn[l],
            s_l, μ, α, Δτ, rng, perm, u, v
        )

        # Record the acceptance rate
        acceptance_rate += acceptance_rate_l / Lτ

        # If using symmetric propagator definition (symmetric = true), then apply
        # the transformation G̃ ==> G = exp{-Δτ⋅K/2}⋅G̃⋅exp{+Δτ⋅K/2}.
        # If asymmetric propagator definition is used (symmetric = false),
        # then this does nothing.
        jdqmcf.partially_wrap_greens_forward!(Gup, Bup[l], A)
        jdqmcf.partially_wrap_greens_forward!(Gdn, Bdn[l], A)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetGup, sgndetGup, δGup, δθup = jdqmcf.stabilize_equaltime_greens!(
            Gup, logdetGup, sgndetGup, fermion_greens_calc_up, Bup, update_B̄ = true
        )
        logdetGdn, sgndetGdn, δGdn, δθdn = jdqmcf.stabilize_equaltime_greens!(
            Gdn, logdetGdn, sgndetGdn, fermion_greens_calc_dn, Bdn, update_B̄ = true
        )

        # Record largest numerical error corrected by numerical stabilization.
        δG = max(δG, δGup, δGdn)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fermion_greens_calc_dn, fermion_greens_calc_up.forward)
    end

    return logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, acceptance_rate
end

# Iterate over all sites for single imaginary time-slice, attempting a local
# update to each corresponding Ising HS field.
function _local_update!(
    Gup, logdetGup, sgndetGup, Bup, Gdn, logdetGdn, sgndetGdn, Bdn,
    s, μ, α, Δτ, rng, perm, u, v
)

    # Randomize the order in which the sites are iterated over.
    shuffle!(rng, perm)

    # Counter for number of accepted updates.
    accepted = 0

    # Iterate over sites in lattice.
    for i in perm

        # Calculate the change in the diagonal potential energy matrix element
        # assuming the sign of the Ising HS field is changed.
        ΔVup = -α/Δτ * (-2*s[i])
        ΔVdn = +α/Δτ * (-2*s[i])

        # Calculate the determinant ratio associated with the proposed update.
        Rup, Δup = jdqmcf.local_update_det_ratio(Gup, ΔVup, i, Δτ)
        Rdn, Δdn = jdqmcf.local_update_det_ratio(Gdn, ΔVdn, i, Δτ)

        # Calculate the acceptance probability based on the Metropolis accept/reject criteria.
        P = min(1.0, abs(Rup * Rdn))

        # Randomly Accept or reject the proposed update with the specified probability.
        if rand(rng) < P

            # Increment the accepted update counter.
            accepted += 1

            # Flip the appropriate Ising HS field.
            s[i] = -s[i]

            # Update the Green's function matrices.
            logdetGup, sgndetGup = jdqmcf.local_update_greens!(
                Gup, logdetGup, sgndetGup, Bup, Rup, Δup, i, u, v
            )
            logdetGdn, sgndetGdn = jdqmcf.local_update_greens!(
                Gdn, logdetGdn, sgndetGdn, Bdn, Rdn, Δdn, i, u, v
            )
        end
    end

    # Calculate the acceptance rate.
    acceptance_rate = accepted / N

    return logdetGup, sgndetGup, logdetGdn, sgndetGdn, acceptance_rate
end;

# Make measurements.
function make_measurements!(
    Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
    Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    bin, avg_sign, density, double_occ, C_greens, C_spinz, C_density,
    C_loc_swave, C_ext_swave, C_dwave
)


    # Initialize time-displaced Green's function matrices for both spin species:
    # G(τ=0,τ=0) = G(0,0)
    # G(τ=0,0)   = G(0,0)
    # G(0,τ=0)   = -(I-G(0,0))
    jdqmcf.initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup)
    jdqmcf.initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn)

    # Calculate the current sign.
    sgn = sign(inv(sgndetGup) * inv(sgndetGdn))

    # Measure the average sign.
    avg_sign[bin] += sgn

    # Measure the density.
    nup = jdqmcm.measure_n(Gup)
    ndn = jdqmcm.measure_n(Gdn)
    density[bin] += sgn * (nup + ndn)

    # Measure the double occupancy.
    double_occ[bin] += sgn * jdqmcm.measure_double_occ(Gup, Gdn)

    # Measure equal-time correlation functions.
    make_correlation_measurements!(
        Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
        unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
        bin, 0, sgn, C_greens, C_spinz, C_density, C_loc_swave, C_ext_swave, C_dwave
    )

    # Iterate over imaginary time slices.
    for l in fermion_greens_calc_up

        # Propagate equal-time Green's function matrix to current imaginary time
        # G(τ±Δτ,τ±Δτ) ==> G(τ,τ) depending on whether iterating over imaginary
        # time in the forward or reverse direction
        jdqmcf.propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fermion_greens_calc_up, Bup)
        jdqmcf.propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fermion_greens_calc_dn, Bdn)

        # Measure time-displaced correlation function measurements for τ = l⋅Δτ.
        make_correlation_measurements!(
            Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
            bin, l, sgn, C_greens, C_spinz, C_density, C_loc_swave, C_ext_swave, C_dwave,
        )

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetGup, sgndetGup, δGup, δθup = jdqmcf.stabilize_unequaltime_greens!(
            Gup_τ0, Gup_0τ, Gup_ττ, logdetGup, sgndetGup, fermion_greens_calc_up, Bup, update_B̄=false
        )
        logdetGdn, sgndetGdn, δGdn, δθdn = jdqmcf.stabilize_unequaltime_greens!(
            Gdn_τ0, Gdn_0τ, Gdn_ττ, logdetGdn, sgndetGdn, fermion_greens_calc_dn, Bdn, update_B̄=false
        )

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fermion_greens_calc_dn, fermion_greens_calc_up.forward)
    end

    return nothing
end

# Make time-displaced measurements.
function make_correlation_measurements!(
    Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    bin, l, sgn, C_greens, C_spinz, C_density, C_loc_swave, C_ext_swave, C_dwave,
    tmp = zeros(eltype(C_greens), lattice.L...)
)

    # Get a view into the arrays accumulating the correlation measurements
    # for the current imaginary time-slice and bin.
    C_greens_bin_l  = @view C_greens[bin,:,:,l+1]
    C_spinz_bin_l = @view C_spinz[bin,:,:,l+1]
    C_density_bin_l = @view C_density[bin,:,:,l+1]
    C_loc_swave_bin_l = @view C_loc_swave[bin,:,:,l+1]
    C_ext_swave_bin_l = @view C_ext_swave[bin,:,:,l+1]
    C_dwave_bin_l = @view C_dwave[bin,:,:,l+1]

    # Measure Green's function for both spin-up and spin-down.
    jdqmcm.greens!(C_greens_bin_l, 1, 1, unit_cell, lattice, Gup_τ0, sgn)
    jdqmcm.greens!(C_greens_bin_l, 1, 1, unit_cell, lattice, Gdn_τ0, sgn)

    # Measure spin-z spin-spin correlation.
    jdqmcm.spin_z_correlation!(
        C_spinz_bin_l, 1, 1, unit_cell, lattice,
        Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn
    )

    # Measure density-density correlation.
    jdqmcm.density_correlation!(
        C_density_bin_l, 1, 1, unit_cell, lattice,
        Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn
    )

    # Measure local s-wave correlation measurement.
    jdqmcm.pair_correlation!(
        C_loc_swave_bin_l, bond_trivial, bond_trivial, unit_cell, lattice, Gup_τ0, Gdn_τ0, sgn
    )

    # Group the nearest-neighbor bonds together.
    bonds = (bond_px, bond_nx, bond_py, bond_ny)

    # d-wave correlation phases.
    dwave_phases = (+1, +1, -1, -1)

    # Iterate over all pairs of nearest-neigbbor bonds.
    for i in eachindex(bonds)
        for j in eachindex(bonds)
            # Measure pair correlation associated with bond pair.
            fill!(tmp, 0)
            jdqmcm.pair_correlation!(
                tmp, bonds[i], bonds[j], unit_cell, lattice, Gup_τ0, Gdn_τ0, sgn
            )
            # Add contribution to extended s-wave and d-wave pair correlation.
            @. C_ext_swave_bin_l += tmp / 4
            @. C_dwave_bin_l += dwave_phases[i] * dwave_phases[j] * tmp / 4
        end
    end

    return nothing
end;

# High-level function to run the DQMC simulation.
function run_simulation!(
    s, μ, α, Δτ, rng, N_burnin, N_bins, N_binsize, N_sweeps,
    Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
    Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    avg_sign, density, double_occ, C_greens, C_spinz, C_density,
    C_loc_swave, C_ext_swave, C_dwave
)

    # Get start time for simulation.
    t_start = time()

    # Initialize variable to keep track of largest corrected numerical error.
    δG = 0.0

    # The acceptance rate on local updates.
    acceptance_rate = 0.0

    println("Beginning Thermalization Updates")

    # Perform burnin updates to thermalize system.
    for n in 1:N_burnin

        # Attempt local update to every Ising HS field.
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG′, ac) = local_update!(
            Gup, logdetGup, sgndetGup, Bup, fermion_greens_calc_up,
            Gdn, logdetGdn, sgndetGdn, Bdn, fermion_greens_calc_dn,
            s, μ, α, Δτ, δG, rng
        )

        # Record max numerical error.
        δG = max(δG, δG′)

        # Update acceptance rate.
        acceptance_rate += ac
    end

    println()
    println("Beginning Measurements.")

    # Iterate over measurement bins.
    for bin in 1:N_bins

        # Iterate over updates and measurements in bin.
        for n in 1:N_binsize

            # Iterate over number of local update sweeps per measurement.
            for sweep in 1:N_sweeps
                # Attempt local update to every Ising HS field.
                (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG′, ac) = local_update!(
                    Gup, logdetGup, sgndetGup, Bup, fermion_greens_calc_up,
                    Gdn, logdetGdn, sgndetGdn, Bdn, fermion_greens_calc_dn,
                    s, μ, α, Δτ, δG, rng
                )

                # Record max numerical error.
                δG = max(δG, δG′)

                # Update acceptance rate.
                acceptance_rate += ac
            end

            # Make measurements.
            make_measurements!(
                Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
                Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
                unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
                bin, avg_sign, density, double_occ, C_greens, C_spinz, C_density,
                C_loc_swave, C_ext_swave, C_dwave
            )
        end

        # Normalize accumulated measurements by the bin size.
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

    # Calculate the final acceptance rate for local updates.
    acceptance_rate /= (N_burnin + N_bins * N_binsize * N_sweeps)

    println()
    println("Simuilation Complete.")
    println()

    # Get simulation runtime
    runtime = time() - t_start

    return acceptance_rate, δG, runtime
end;

# Run the DQMC simulation.
acceptance_rate, δG, runtime = run_simulation!(
    s, μ, α, Δτ, rng, N_burnin, N_bins, N_binsize, N_sweeps,
    Gup, logdetGup, sgndetGup, Gup_ττ, Gup_τ0, Gup_0τ, Bup, fermion_greens_calc_up,
    Gdn, logdetGdn, sgndetGdn, Gdn_ττ, Gdn_τ0, Gdn_0τ, Bdn, fermion_greens_calc_dn,
    unit_cell, lattice, bond_trivial, bond_px, bond_nx, bond_py, bond_ny,
    avg_sign, density, double_occ, C_greens, C_spinz, C_density,
    C_loc_swave, C_ext_swave, C_dwave
)
println("Acceptance Rate = ", acceptance_rate)
println("Largest Numerical Error = ", δG)
println("Simulation Run Time (sec) = ", runtime)

# Calculate the average sign for the simulation.
sign_avg, sign_std = jdqmcm.jackknife(identity, avg_sign)
println("Avg Sign, S = ", real(sign_avg), " +/- ", sign_std)

# Calculate the average density.
density_avg, density_std = jdqmcm.jackknife(/, density, avg_sign)
println("Density, n = ", real(density_avg), " +/- ", density_std)

# Calculate the average double occupancy.
double_occ_avg, double_occ_std = jdqmcm.jackknife(/, double_occ, avg_sign)
println("Double occupancy, nup_ndn = ", real(double_occ_avg), " +/- ", double_occ_std)

# Given the binned time-displaced correlation function/structure factor data,
# calculate and return the corresponding binned susceptibility data.
function susceptibility(S::AbstractArray{T}, Δτ) where {T<:Number}

    # Allocate array to contain susceptibility.
    χ = zeros(T, size(S)[1:3])

    # Iterate over bins.
    for bin in axes(S,1)

        # Calculate the susceptibility for the current bin by integrating the correlation
        # data over the imaginary time axis using Simpson's rule.
        S_bin = @view S[bin,:,:,:]
        χ_bin = @view χ[bin,:,:]
        jdqmcm.susceptibility!(χ_bin, S_bin, Δτ, 3)
    end

    return χ
end

# Calculate average correlation function values based on binned data.
function correlation_stats(
    S::AbstractArray{Complex{T}},
    avg_sign::Vector{E}
) where {T<:AbstractFloat, E<:Number}

    # Allocate arrays to contain the mean and standard deviation of
    # measured correlation function.
    S_avg = zeros(Complex{T}, size(S)[2:end])
    S_std = zeros(T, size(S)[2:end])

    # Number of bins.
    N_bins = length(avg_sign)

    # Preallocate arrays to make the jackknife error analysis faster.
    jackknife_sample_means = (zeros(Complex{T}, N_bins), zeros(E, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # Iterate over correlation functions.
    for n in CartesianIndices(S_avg)
        # Use the jackknife method to calculage average and error.
        vals = @view S[:,n]
        S_avg[n], S_std[n] = jdqmcm.jackknife(
            /, vals, avg_sign,
            jackknife_sample_means = jackknife_sample_means,
            jackknife_g = jackknife_g
        )
    end

    return S_avg, S_std
end

# Fourier transform Green's function from position to momentum space.
S_greens = copy(C_greens)
jdqmcm.fourier_transform!(S_greens, 1, 1, (1,4), unit_cell, lattice)

# Calculate average Green's function in position space.
C_greens_avg, C_greens_std = correlation_stats(C_greens, avg_sign)

# Calculate average Green's function in momentum space.
S_greens_avg, S_greens_std = correlation_stats(S_greens, avg_sign)

# Verify that the position space G(r=0,τ=0) measurement agrees with the
# average density measurement.
agreement = (2*(1-C_greens_avg[1,1,1]) ≈ density_avg)

# Fourier transform the binned Cz(r,τ) position space spin-z correlation function
# data to get the binned Sz(q,τ) spin-z structure factor data.
S_spinz = copy(C_spinz)
jdqmcm.fourier_transform!(S_spinz, 1, 1, (1,4), unit_cell, lattice)

# Integrate the binned Sz(q,τ) spin-z structure factor data over the imaginary
# time axis to get the binned χz(q) spin susceptibility.
χ_spinz = susceptibility(S_spinz, Δτ)

# Calculate the average spin correlation functions in position space.
C_spinz_avg, C_spinz_std = correlation_stats(C_spinz, avg_sign)

# Calculate the average spin structure factor in momentum space.
S_spinz_avg, S_spinz_std = correlation_stats(S_spinz, avg_sign)

# Calculate the average spin susceptibility for all scattering momentum q.
χ_spinz_avg, χ_spinz_std = correlation_stats(χ_spinz, avg_sign)

# Report the spin susceptibility χafm = χz(π,π) corresponding to antiferromagnetism.
χafm_avg = real(χ_spinz_avg[L÷2+1, L÷2+1])
χafm_std = χ_spinz_std[L÷2+1, L÷2+1]
println("Antiferromagentic Spin Susceptibility, chi_afm = ", χafm_avg, " +/- ", χafm_std)

# Fourier transform the binned Cρ(r,τ) position space density correlation
# data to get the time-dispaced charge structure factor Sρ(q,τ) in
# momentum space.
S_density = copy(C_density)
jdqmcm.fourier_transform!(S_density, 1, 1, (1,4), unit_cell, lattice)

# Integrate the binned Sρ(q,τ) density structure factor data over the imaginary
# time axis to get the binned χρ(q) density susceptibility.
χ_density = susceptibility(S_density, Δτ)

# Calculate the average charge correlation functions in position space.
C_density_avg, C_density_std = correlation_stats(C_density, avg_sign)

# Calculate the average charge structure factor in momentum space.
S_density_avg, S_density_std = correlation_stats(S_density, avg_sign)

# Calculate the average charge susceptibility for all scattering momentum q.
χ_density_avg, χ_density_std = correlation_stats(χ_spinz, avg_sign);

# Fourier transform binned position space local s-wave correlation function data to get
# the binned momentum space local s-wave structure factor data.
S_loc_swave = copy(C_loc_swave)
jdqmcm.fourier_transform!(S_loc_swave, 1, 1, (1,4), unit_cell, lattice)

# Integrate the binned local s-wave structure factor data to get the
# binned local s-wave pair susceptibility data.
P_loc_swave = susceptibility(S_loc_swave, Δτ)

# Calculate the average local s-wave pair susceptibility for all scattering momentum q.
P_loc_swave_avg, P_loc_swave_std = correlation_stats(P_loc_swave, avg_sign)

# Report the local s-wave pair suspcetibility.
Ps_avg = real(P_loc_swave_avg[1,1])
Ps_std = P_loc_swave_std[1,1]
println("Local s-wave pair susceptibility, P_s = ", Ps_avg, " +/- ", Ps_std)

# Fourier transform binned position space extended s-wave correlation function data to get
# the binned momentum space extended s-wave structure factor data.
S_ext_swave = copy(C_ext_swave)
jdqmcm.fourier_transform!(S_ext_swave, 1, 1, (1,4), unit_cell, lattice)

# Integrate the binned extended s-wave structure factor data to get the
# binned extended s-wave pair susceptibility data.
P_ext_swave = susceptibility(S_ext_swave, Δτ)

# Calculate the average extended s-wave pair susceptibility for all scattering momentum q.
P_ext_swave_avg, P_ext_swave_std = correlation_stats(P_ext_swave, avg_sign)

# Report the local s-wave pair suspcetibility.
Pexts_avg = real(P_ext_swave_avg[1,1])
Pexts_std = P_ext_swave_std[1,1]
println("Extended s-wave pair susceptibility, P_ext-s = ", Pexts_avg, " +/- ", Pexts_std)

# Fourier transform binned position space d-wave correlation function data to get
# the binned momentum space d-wave structure factor data.
S_dwave = copy(C_dwave)
jdqmcm.fourier_transform!(S_dwave, 1, 1, (1,4), unit_cell, lattice)

# Integrate the binned d-wave structure factor data to get the
# binned d-wave pair susceptibility data.
P_dwave = susceptibility(S_dwave, Δτ)

# Calculate the average d-wave pair susceptibility for all scattering momentum q.
P_dwave_avg, P_dwave_std = correlation_stats(P_dwave, avg_sign)

# Report the d-wave pair susceptibility.
Pd_avg = real(P_dwave_avg[1,1])
Pd_std = P_dwave_std[1,1]
println("Extended d-wave pair susceptibility, P_d = ", Pd_avg, " +/- ", Pd_std)
