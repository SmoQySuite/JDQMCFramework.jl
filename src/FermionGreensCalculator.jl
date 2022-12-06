@doc raw"""
    FermionGreensCalculator{T<:Continuous, E<:AbstractFloat}

A type to facilitate calculating the single-particle fermion Green's function matrix.

# Fields

- `forward::Bool`: If `true` then iterate over imaginary time slices from ``l=1`` to ``l=L_\tau``, if `false` then iterate over imaginary time slices from ``l=L_\tau`` to ``l=1``.
- `l::Int`: The current imaginary time slice ``\tau = l \cdot \Delta\tau``.
- `N::Int`: Orbitals in system.
- `β::E`: The inverse temperature ``\beta=1/T,`` where ``T`` is temperature.
- `Δτ::E`: Discretization in imaginary time.
- `Lτ::Int`: Length of imaginary time axis, ``L_\tau = \beta / \Delta\tau.``
- `nₛ::Int`: Frequency with which numerical stabilization is performed, i.e. every ``n_s`` imaginary time slices the equal-time Green's function is recomputed from scratch.
- `Nₛ::Int`: Number of numerical stabilization intervals, ``N_s = \left\lceil L_\tau / n_s \right\rceil.``
- `B̄::Array{T,3}`: A multidimensional array where the matrix `B̄[:,:,n]` represents ``\bar{B}_n.``
- `F::Vector{LDR{T,E}}`: A vector of ``N_s`` LDR factorizations to represent the matrices ``B(0,\tau)`` and ``B(\tau,\beta)``.
- `G′::Matrix{T}`: Matrix used for calculating the error corrected by numerical stabilization of the equal time Green's function.
- `ldr_ws::LDRWorkspace{T}`: Workspace for performing LDR factorization while avoiding dynamic memory allocations.
"""
mutable struct FermionGreensCalculator{T<:Continuous, E<:AbstractFloat}

    forward::Bool
    l::Int
    const N::Int
    const β::E
    const Δτ::E
    const Lτ::Int
    const nₛ::Int
    const Nₛ::Int
    const B̄::Array{T, 3}
    const F::Vector{LDR{T,E}}
    const G′::Matrix{T}
    const ldr_ws::LDRWorkspace{T}
end


@doc raw"""
    fermion_greens_calculator(B::AbstractVector{P}, N::Int, β::E, Δτ::E,
                              nₛ::Int) where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator{T}}

Initialize and return [`FermionGreensCalculator`](@ref) struct based on the vector of propagators `B` passed to the function.
"""
function fermion_greens_calculator(B::AbstractVector{P}, N::Int, β::E, Δτ::E,
                                   nₛ::Int) where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # get length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # check that there is a propagator for each time slice
    @assert length(B) == Lτ

    # make sure that the discretization in imaginary time is valid
    @assert (Lτ*Δτ) ≈ β

    # calculate the number of numerical stabalization intervals
    Nₛ = ceil(Int, Lτ/nₛ)

    # allocate array to represent partical products of B matrices,
    # setting each equal to the identity matrix
    B̄ = zeros(T, N, N, Nₛ)
    for n in axes(B̄,3)
        B̄ₙ = @view B̄[:,:,n]
        copyto!(B̄ₙ, I)
    end

    # calculate scratch matrix
    G′ = zeros(T, N, N)

    # construct vector of LDR factorization to represent B(τ,0) and B(β,τ) matrices
    copyto!(G′, I)
    ldr_ws = ldr_workspace(G′)
    F = ldrs(G′, Nₛ)

    # current imaginary time slice
    l = Lτ

    # whether to iterate over imaginary time τ=Δτ⋅l in forward (l=1 ==> l=Lτ) or reverse order (l=Lτ ==> l=1)
    forward = false

    # allocate FermionGreensCalculator struct
    fgc = FermionGreensCalculator(forward, l, N, β, Δτ, Lτ, nₛ, Nₛ, B̄, F, G′, ldr_ws)

    # initialize FermionGreensCalculator struct
    for l in fgc
        update_factorizations!(fgc, B)
    end

    return fgc
end


@doc raw"""
    fermion_greens_calculator(fgc::FermionGreensCalculator{T,E}) where {T,E}

Return a new [`FermionGreensCalculator`](@ref) that is a copy of `fgc`.
"""
function fermion_greens_calculator(fgc::FermionGreensCalculator{T,E}) where {T,E}

    (; forward, l, N, β, Δτ, Lτ, nₛ, Nₛ, B̄, F, G′, ldr_ws) = fgc

    B̄_new = copy(B̄)
    F_new = deepcopy(F)
    G′_new = copy(G′)
    copyto!(G′_new,I)
    ldr_ws_new = ldr_workspace(G′_new)

    return FermionGreensCalculator(forward, l, N, β, Δτ, Lτ, nₛ, Nₛ, B̄_new, F_new, G′_new, ldr_ws_new)
end


########################
## OVERLOADED METHODS ##
########################


@doc raw"""
    eltype(fgc::FermionGreensCalculator{T,E}) where {T,E}

Return matrix element type `T` associated with an instance of [`FermionGreensCalculator`](ref).
"""
function eltype(fgc::FermionGreensCalculator{T,E}) where {T,E}

    return T
end


@doc raw"""
    copyto!(fgc_out::FermionGreensCalculator{T,E}, fgc_in::FermionGreensCalculator{T,E}) where {T,E}

Copy the contents of `fgc_in` to `fgc_out`.
"""
function copyto!(fgc_out::FermionGreensCalculator{T,E}, fgc_in::FermionGreensCalculator{T,E}) where {T,E}

    fgc_out.forward = fgc_in.forward
    fgc_out.l = fgc_in.l
    copyto!(fgc_out.B̄::Array{T, 3}, fgc_in.B̄::Array{T, 3})
    for i in eachindex(fgc_in.F)
        copyto!(fgc_out.F[i]::LDR{T,E}, fgc_in.F[i]::LDR{T,E})
    end

    return nothing
end


@doc raw"""
    iterate(iter::FermionGreensCalculator)

    iterate(iter::FermionGreensCalculator, state)

Iterate over imaginary time slices, alternating between iterating in the forward direction from ``l=1`` to ``l=L_\tau``
and in the reverse direction from ``l=L_\tau`` to ``l=1``. The `iter.forward` boolean field in the
[`FermionGreensCalculator`](@ref) type determines whether the imaginary time slices are iterated over in forward
or reverse order. The `iter.forward` field is updated as needed automatically and *should not* be adjusted manually.
"""
function iterate(iter::FermionGreensCalculator)::Tuple{Int,Bool}

    state = iter.forward
    # set initial time slice l
    if state # iterating from l=1 ==> l=Lτ
        item = 1
    else # iterating from l=Lτ ==> l=1
        item = iter.Lτ
    end

    return (item, state)
end

function iterate(iter::FermionGreensCalculator, state::Bool)::Union{Tuple{Int,Bool},Nothing}

    if state # iterating from l=1 ==> l=Lτ
        iter.l += 1
        # terminiation criteria
        if iter.l == iter.Lτ+1
            iter.l = iter.Lτ
            iter.forward = false
            return nothing
        else
            return (iter.l, iter.forward)
        end
    else # iterating from l=Lτ ==> l=1
        iter.l -= 1
        # termination criteria
        if iter.l == 0
            iter.l = 1
            iter.forward = true
            return nothing
        else
            return (iter.l, iter.forward)
        end
    end

    return next
end


######################################
## DEVELOPER METHODS (NOT EXPORTED) ##
######################################


@doc raw"""
    update_factorizations!(fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T,E}}

If current imaginary time slice `fgc.l` corresponds to the boundary of a stabilization interval,
calculate a LDR factorization to represent ``B(0, \tau)`` or ``B(\tau-\Delta\tau, \beta)``
if iterating over imaginary time in the forward (`fgc.forward = true`)
or reverse (`fgc.forward = false`) directions respectively.
This method should be called *after* all changes to the current time slice propagator matrix
``B_l`` have been made
This method will also recompute ``\bar{B}_n`` as needed.
"""
function update_factorizations!(fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T,E}}

    # update B̄ matrix when necessary
    update_B̄!(fgc, B)

    # update the factorization
    update_factorizations!(fgc)

    return nothing
end

@doc raw"""
    update_factorizations!(fgc::FermionGreensCalculator{T,E}) where {T, E}

If current imaginary time slice `fgc.l` corresponds to the boundary of a stabilization interval,
calculate a LDR factorization to represent ``B(\tau, 0)`` or ``B(\beta, \tau-\Delta\tau)``
if iterating over imaginary time in the forward (`fgc.forward = true`)
or reverse (`fgc.forward = false`) directions respectively.
This method should be called *after* all changes to the current time slice propagator matrix
``B_l`` have been made, and any required updates to a ``\bar{B}_n`` matrix have
been performed using the [`JDQMCFramework.update_B̄!`](@ref) routine.
"""
function update_factorizations!(fgc::FermionGreensCalculator{T,E}) where {T, E}

    (; l, Lτ, nₛ, Nₛ, forward) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}
    B̄ = fgc.B̄::Array{T,3}

    # get stabalizaiton interval
    n, l′ = stabilization_interval(fgc)

    # if iterating from l=1 => l=Lτ
    if forward
        # if at boundary of first stabilization interval (l=nₛ)
        if l′==nₛ && n==1
            # calculate B(τ=nₛ⋅Δτ,0) = B̄[1]
            B̄₁ = @view B̄[:,:,1]
            B_τ0 = F[1]::LDR{T,E}
            ldr!(B_τ0, B̄₁, ldr_ws)
        # if at the end of a stabilization interval
        elseif l′==nₛ || l==Lτ && Nₛ > 1
            # calculate B(τ=n⋅nₛ⋅Δτ,0) = B̄[n]⋅B(τ=(n-1)⋅nₛ⋅Δτ,0)
            B̄ₙ = @view B̄[:,:,n]
            B_τ0_new = F[n]::LDR{T,E}
            B_τ0_prev = F[n-1]::LDR{T,E}
            mul!(B_τ0_new, B̄ₙ, B_τ0_prev, ldr_ws)
        end
    # if iterating from l=Lτ => l=1
    else
        # if at boundary of last stabilization interval (l=Lτ-nₛ+1)
        if l′==1 && n==Nₛ
            # calculate B(β,τ=β-(nₛ+1)Δτ) = B̄[Nₛ]
            B̄_Nₛ = @view B̄[:,:,n]
            B_βτ = F[n]::LDR{T,E}
            ldr!(B_βτ, B̄_Nₛ, ldr_ws)
        # if at boundary of stabilization interval
        elseif l′==1 && Nₛ > 1
            # calculate B(β,τ=β-n⋅nₛ⋅Δτ-Δτ) = B(β,τ=β-n⋅nₛ⋅Δτ)⋅B̄[n]
            B̄ₙ = @view B̄[:,:,n]
            B_βτ_new = F[n]::LDR{T,E}
            B_βτ_prev = F[n+1]::LDR{T,E}
            mul!(B_βτ_new, B_βτ_prev, B̄ₙ, ldr_ws)
        end
    end

    return nothing
end


@doc raw"""
    update_B̄!(fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P}) where {T,E,P<:AbstractPropagator{T,E}}

Recalculate ``\bar{B}_n`` if the current timeslice `fgc.l` corresponds to the boundary of a stabilization interval,
accounting for whether imaginary time is being iterated over in the forward (`fgc.forward = true`) or
reverse (`fgc.forward = false`) direction.
"""
function update_B̄!(fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P}) where {T,E,P<:AbstractPropagator{T,E}}

    (; forward, nₛ, l, Lτ) = fgc

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # if iterating over imaginary time in the forward direction l=1 ==> l=Lτ
    if forward
        # if at boundary of stabilization interval
        if l′ == nₛ || l==Lτ
            # update B̄[n]
            calculate_B̄!(fgc, B, n)
        end
    # if iterating over imaginary time in the reviews direction l=Lτ ==> l=1
    else
        # if at boundary of stabilization interval
        if l′ == 1
            # update B̄[n]
            calculate_B̄!(fgc, B, n)
        end
    end

    return nothing
end


@doc raw"""
    calculate_B̄!(fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P}, n::Int) where {T,E,P<:AbstractPropagator{T,E}}

Given `B`, a vector of all the propagator matrices ``B_l``, calculate the matrix product
```math
\bar{B}_{\sigma,n}=\prod_{l=(n-1)\cdot n_{s}+1}^{\min(n\cdot n_{s},L_{\tau})}B_{\sigma,l},
```
with the result getting written to `fgc.B̄[:,:,n]`.
"""
function calculate_B̄!(fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P}, n::Int) where {T,E,P<:AbstractPropagator{T,E}}

    (; B̄, nₛ, Lτ, Nₛ, ldr_ws) = fgc
    @assert 1 <= n <= Nₛ
    B̄ₙ = @view B̄[:,:,n]
    copyto!(B̄ₙ, I) # B̄ₙ := I
    # iterate over imaginary time slices associated with stabilization interval
    for l in min(n*nₛ,Lτ):-1:(n-1)*nₛ+1
        B_l = B[l]::P
        rmul!(B̄ₙ, B_l, M = ldr_ws.M) # B̄ₙ := B̄ₙ⋅B[l]
    end

    return nothing
end


@doc raw"""
    stabilization_interval(fgc::FermionGreensCalculator)::Tuple{Int,Int}

Given the current imaginary time slice `fgc.l`, return the corresponding
stabilization interval `n = ceil(Int, fgc.l/fgc.nₛ)`, and the relative location
within that stabilization interval `l′ = mod1(fgc.l, fgc.nₛ)`, such that `l′∈[1,nₛ]`. 
"""
function stabilization_interval(fgc::FermionGreensCalculator)::Tuple{Int,Int}

    # calculate stabilization interval
    n = ceil(Int, fgc.l/fgc.nₛ)
    # location in stabilization interval l′∈[1,nₛ]
    l′ = mod1(fgc.l, fgc.nₛ)

    return n, l′
end