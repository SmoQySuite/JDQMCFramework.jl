```@meta
CurrentModule = JDQMCFramework
```

# JDQMCFramework.jl

Documentation for [JDQMCFramework](https://github.com/cohensbw/JDQMCFramework.jl).
This is a utility package that exports a suite of types and routines to simplify the
process of writing a DQMC code.

Matrix stabilization routines are supplied by the
[`StableLinearAlgebra.jl`](https://github.com/cohensbw/StableLinearAlgebra.jl.git)
package.

The checkerboard decomposition functionality supported here is provided by the
[`Checkerboard.jl`](https://github.com/cohensbw/Checkerboard.jl.git) package.

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Installation
To install [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git) run following in the Julia REPL:

```julia
] add JDQMCFramework
```

## Formalism and Definitions

This section describes the formalism and definitions adopted by the
JDQMCFramework package. The following discussion assumes an existing
familiarity with the determinant quantum Monte Carlo (DQMC) algorithm,
a method for simulating systems of itinerant fermions on a lattice
at finite temperature in the grand canonical ensemble. The DQMC formalism
starts by representing the partition funciton as a path integral
```math
\begin{align*}
Z= & \textrm{Tr}\,e^{-\beta\hat{H}}=\textrm{Tr}\left[\prod_{l=1}^{L_{\tau}}e^{-\Delta\tau\hat{H}}\right]
\end{align*}
```
in imaginary time ``\tau=l\cdot\Delta\tau``, at inverse temperature
``\beta=L_{\tau}\cdot\Delta\tau``. Next, the Suzuki-Trotter approximation
is applied, and Hubbard-Stratonivich transformations are used as needed
to render the Hamiltonian quadratic in fermion creation and annihilation
operators. Lastly, the fermionic degrees of freedom are integrated
out.

The resulting approximate expression for the partition function allows
for the definition of Monte Carlo weights of the form
```math
W(\mathbf{x})=e^{-S_{B}}\prod_{\sigma}\det M_{\sigma},
```
where ``\mathbf{x}`` signifies all the relevant degrees of freedom
that need to be sampled. While not written explicitly, the bosonic
action ``S_{B}`` and each fermion determinant matrix ``M_{\sigma}``
depend on ``\mathbf{x}``. In the absence of a mean field pairing term
or some similarly exotic interaction, the index ``\sigma`` typically
corresponds to the fermion spin species. In the case of electrons
this means ``\sigma=\{\uparrow,\downarrow\}``.

Each fermion determinant matrix is of the form
```math
\begin{align*}
M_{\sigma}(\tau)= & I+B_{\sigma}(\tau,0)B_{\sigma}(\beta,\tau)\\
= & I+B_{\sigma,l}B_{\sigma,l-1}\dots B_{\sigma,1}B_{\sigma,L_{\tau}}\dots B_{\sigma,l+2}B_{\sigma,l+1}
\end{align*}
```
where
```math
B_{\sigma}(\tau,\tau')=B_{\sigma,l}B_{\sigma,l-1}\dots B_{\sigma,l'+1}
```
such that ``\det M_{\sigma}(\tau)=\det M_{\sigma}(\tau')`` for any
pair ``(l,l')\in[1,L_{\tau}]``.

Each propagator matrix ``B_{\sigma,l}=B_{\sigma}(\tau,\tau-\Delta\tau)`` may be represented in either the symmetric form
```math
B_{\sigma,l}=e^{-\tfrac{\Delta\tau}{2}K_{l}} e^{-\Delta\tau V_{l}} e^{-\tfrac{\Delta\tau}{2}K_{l}}
```
or the asymmetric form
```math
B_{\sigma,l} = e^{-\Delta\tau V_{l}} e^{-\Delta\tau K_{l}},
```
where ``V_{l}`` is a diagonal matrix corresponding to the on-site energy
for each site in the lattice, and ``K_{l}`` is the strictly off-diagonal
hopping matrix.

The single-particle fermion Green's function is given by 
```math
G_{\sigma,i,j}(\tau,\tau')=\langle\hat{\mathcal{T}}\hat{c}_{\sigma,i}(\tau)\hat{c}_{\sigma,j}^{\dagger}(\tau')\rangle=\begin{cases}
\langle\hat{c}_{\sigma,i}(\tau)\hat{c}_{\sigma,j}^{\dagger}(\tau')\rangle & \tau\ge\tau'\\
-\langle\hat{c}_{\sigma,j}^{\dagger}(\tau')\hat{c}_{\sigma,i}(\tau)\rangle & \tau<\tau',
\end{cases}
```
where ``\hat{c}_{\sigma,i}^{\dagger}\,(\hat{c}_{\sigma,i})`` is the
fermion creation (annihilation) operator for a fermion with spin ``\sigma``
on site ``i`` on the lattice, and ``\hat{\mathcal{T}}`` is the time-ordering
operator. The equal-time Green's function is related to the fermion
determinant matrix by
```math
G_{\sigma,i,j}(\tau,\tau)=M_{\sigma,i,j}^{-1}(\tau),
```
where again ``\tau=l\cdot\Delta\tau``. The equal-time Green's function
matrix can be advanced to the next imaginary time slice using the
relationship
```math
G_{\sigma}(\tau+\Delta\tau,\tau+\Delta\tau)=B_{\sigma,l+1}G_{\sigma}(\tau,\tau)B_{\sigma,l+1}^{-1}
```
and
```math
G_{\sigma}(\tau-\Delta\tau,\tau-\Delta\tau)=B_{\sigma,l}^{-1}G_{\sigma}(\tau,\tau)B_{\sigma,l}.
```
The unequal-time Green's function is accessible using the relation
```math
\begin{align*}
G_{\sigma}(\tau,0)= & B_{\sigma}(\tau,0)G_{\sigma}(0,0)\\
= & [B_{\sigma}^{-1}(\tau,0)+B_{\sigma}(\beta,\tau)]^{-1},
\end{align*}
```
which also implies
```math
\begin{align*}
G_{\sigma}(\tau,0)= & B_{\sigma}^{-1}(\tau',\tau)G_{\sigma}(\tau',0),
\end{align*}
```
for ``\tau\in[0,\beta-\Delta\tau]`` and ``\tau<\tau'<\beta``. By applying
the anti-periodic boundary conditions of the single-particle Green's
function in imaginary time it immediately follows that
```math
G_{\sigma}(\beta,0)=I-G_{\sigma}(0,0),
```
where
```math
G_{\sigma}(0,0)=[I+B_{\sigma}(\beta,0)]^{-1}=[I+B_{\sigma,L_{\tau}}\dots B_{\sigma,1}]^{-1},
```
subject to the boundary condition ``G_{\sigma}(0,0)=G_{\sigma}(\beta,\beta)``.

The DQMC method also requires periodic re-calculation of the fermion
Green's function matrix as ``G_{\sigma}(\tau,\tau)`` is propagated
to later or ealier imaginary times to maintain numerical stability.
Therefore, we introduce a parameter ``n_{s},`` which describes the
frequency with which numerical stabilization needs to occur. The number
of "stabilization intervals" in imaginary time is then given by
``N_{s}=\left\lceil L_{\tau}/n_{s}\right \rceil``, and we introduce
the notation
```math
\bar{B}_{\sigma,n}=\prod_{l=(n-1)\cdot n_{s}+1}^{\min(n\cdot n_{s},N_{s})}B_{\sigma,l},
```
where ``n\in[1,N_{s}]``, to represent the product of propagator matrices
over a single stabilization interval. Using this definition we may
express ``G_{\sigma}(0,0)`` as
```math
G_{\sigma}(0,0) = (I + B_{\sigma,L_\tau} B_{\sigma,L_\tau-1} \dots B_{\sigma, 2} B_{\sigma,1})^{-1}
                = (I + \bar{B}_{\sigma,N_s} \bar{B}_{\sigma,N_s-1} \dots \bar{B}_{\sigma, 2} \bar{B}_{\sigma,1})^{-1}.
```

## Basic Usage

In this section we introduce some of the basics of using this package by setting up the framework
for a DQMC simulations in the case of a simple non-interacting square lattice tight binding model,
assuming two electron spin species, spin up and spin down.
While this is a "tivial" example, it is instructive.

```@example square
using LinearAlgebra
using LatticeUtilities
using JDQMCFramework
```

First let us define the relevant model parameters.

```@example square
# hopping amplitude
t = 1.0

# chemical potential
?? = 0.0

# lattice size
L = 4

# inverse temperature
?? = 3.7

# discretization in imaginary time
???? = 0.1

# frequency of numerical stabilization
n??? = 10
nothing; # hide
```

Next we calculate the length of the imaginary time axis ``L_\tau`` using the
[`eval_length_imaginary_axis`](@ref) method.

```@example square
L?? = eval_length_imaginary_axis(??, ????)
```

Using functionality imported from the [`LatticeUtilities.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git)
package, we construct the neighbor table for a square lattice.

```@example square
# define unit cell
unit_cell = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])

# define size of lattice
lattice = Lattice(L = [L,L], periodic = [true,true])

# define bonds/hoppings in x and y directions
bond_x = Bond(orbitals = (1,1), displacement = [1,0])
bond_y = Bond(orbitals = (1,1), displacement = [0,1])

# construct neighbor table
neighbor_table = build_neighbor_table([bond_x, bond_y], unit_cell, lattice)

# calculate number of sites in lattice
N = nsites(unit_cell, lattice)

# calculate number of bonds in lattice
Nbonds = size(neighbor_table, 2)

(N, Nbonds)
```

Next we construct the strictly off-diagonal hopping matrix ``K,`` and a vector to represent
the diagonal on-site energy matrix ``V.``

```julia
# build hopping matrix
K = zeros(typeof(t), N, N)
build_hopping_matrix!(K, neighbor_table, fill(t, Nbonds))

# build vector representing diagonal on-site energy matrix
V = fill(-??, N)
```

Now we define a the propagator ``B_{\sigma,l}`` for each spin species ``\sigma`` and
imaginary time slice ``\tau = \Delta\tau \cdot l.``
Of course, in the non-interacting limit considered here all the propagators matrices are identical.
This will no longer be the case if interactions are introduced, in which case each ``B_{\sigma,l}``
matrix will in general be unique.

```julia
expm????V = exp.(-????*V)
expm????K = exp(-????*K)
expp????K = exp(+????*K)

# null vector spin up propagators to fill
Bup = AsymExactPropagator{eltype(expm????K),eltype(expm????V)}[]

# null vecotr of spin down propagators to fill
Bdn = AsymExactPropagator{eltype(expm????K),eltype(expm????V)}[]

# construct propagator for each spin species and append to appropriate vector
for l in 1:L??
    B_l = AsymExactPropagator(expm????V, expm????K, expp????K)
    push!(Bup, B_l)
    push!(Bdn, B_l)
end
```

In the above we chose to represent the propagator matrices using the [`AsymExactPropagator`](@ref)
type, which assumes the ``B_l = e^{-\Delta\tau K_l} e^{-\Delta\tau V_l}`` definition,
where the ``K_l`` hopping matrix is exactly exponentiated.
This package includes the other possible definitions [`AsymChkbrdPropagator`](@ref),
[`SymExactPropagator`](@ref) and [`SymChkbrdPropagator`](@ref).

Next we instantiate two instances of the [`FermionGreensCalculator`](@ref) type, one for each of the two
electron spin species, spin up and spin down.

```julia
fgc_up = FermionGreensCalculator(Bup, ??, ????, n???)
fgc_dn = FermionGreensCalculator(Bdn, ??, ????, n???)
```

Now we initialize the spin up and spin down equal time Green's function matrices ``G_\uparrow(0,0)``
and ``G_\downarrow(0,0).``

```julia
Gup = zeros(N,N)
logdetGup, sgndetGup = calculate_equaltime_greens!(Gup, fgc_up)

Gdn = zeros(N,N)
logdetGdn, sgndetGdn = calculate_equaltime_greens!(Gdn, fgc_dn)
```

Now we will demonstrate how to synchronously iterate over the imaginary time slices
for both the spin up and spin down sectors.

```julia
# Iterate over imaginary time ??=???????l.
for l in fgc_up

    # Propagate equal-time Green's function matrix to current imaginary time G(????????,????????) ==> G(??,??)
    # depending on whether iterating over imaginary time in the forward or reverse direction
    propagate_equaltime_greens!(Gup, fgc_up, Bup)
    propagate_equaltime_greens!(Gdn, fgc_dn, Bdn)

    # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION FOR THE CURRENT
    # IMAGINARY TIME SLICE WOULD GO HERE

    # Periodically re-calculate the Green's function matrix for numerical stability.
    # Comment: if not performing updates, but just evaluating the derivative of the action, then
    # set update_B??=false to avoid wasting cpu time re-computing B????? matrices.
    logdetGup, sgndetGup, ??Gup, ????up = stabilize_equaltime_greens!(Gup, logdetGup, sgndetGup, fgc_up, Bup, update_B??=true)
    logdetGdn, sgndetGdn, ??Gdn, ????dn = stabilize_equaltime_greens!(Gdn, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B??=true)

    # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
    iterate(fgc_dn, fgc_up.forward)
end
```

Note that if we iterate over imaginary time again, it will iterate in the opposite direction.
This is expected behavior. Each time you iterate over imaginary time the direction of iteration reverses.
While not immediately obvious, this allows for a reduction in the number of required matrix factorizations.

This package also exports two routines, [`local_update_det_ratio`](@ref) and [`local_update_greens!`](@ref),
that are useful for implementing local updates in a DQMC simulation.

Lastly, we will will calculate the unequal-time Green's functions ``G_{\sigma}(\tau,0)`` and the
equal-time Green's function ``G_{\sigma}(\tau,\tau)`` for all imaginary time slices.

```julia
G??0_up = zeros(N, N, L??+1)
G????_up = zeros(N, N, L??+1)
calculate_unequaltime_greens!(G??0_up, G????_up, fgc_up, Bup)

G??0_dn = zeros(N, N, L??+1)
G????_dn = zeros(N, N, L??+1)
calculate_unequaltime_greens!(G??0_dn, G????_dn, fgc_dn, Bdn)
```

Calling the [`calculate_unequaltime_greens!`](@ref) method also reverses the direction of iteration the next time
imaginary time is iterated over.