@doc raw"""
    FermionGreensCalculator{T<:Continuous, E<:AbstractFloat}

A type to facilitate calculating the single-particle fermion Green's function matrix.

# Fields

- `forward::Bool`: If `true` then iterate over imaginary time slices from ``l=1`` to ``l=L_\tau``, if `false` then iterate over imaginary time slices from ``l=L_\tau`` to ``l=1``.
- `l::Int`: The current imaginary time slice ``\tau = l \cdot \Delta\tau``.
- `n_stab::Int`: Frequency with which numerical stabilization is performed, i.e. every ``n_s`` imaginary time slices the equal-time Green's function is recomputed from scratch.
- `N_stab::Int`: Number of numerical stabilization intervals, ``N_s = \left\lceil L_\tau / n_s \right\rceil.``
- `N::Int`: Orbitals in system.
- `β::E`: The inverse temperature ``\beta=1/T,`` where ``T`` is temperature.
- `Δτ::E`: Discretization in imaginary time.
- `Lτ::Int`: Length of imaginary time axis, ``L_\tau = \beta / \Delta\tau.``
- `B_bar::Vector{Matrix{T}}`: A multidimensional array where the matrix `B_bar[:,:,n]` represents ``\bar{B}_n.``
- `F::Vector{LDR{T,E}}`: A vector of ``N_s`` LDR factorizations to represent the matrices ``B(0,\tau)`` and ``B(\tau,\beta)``.
- `G′::Matrix{T}`: Matrix used for calculating the error corrected by numerical stabilization of the equal time Green's function.
- `ldr_ws::LDRWorkspace{T}`: Workspace for performing LDR factorization while avoiding dynamic memory allocations.
"""
mutable struct FermionGreensCalculator{T<:Continuous, E<:AbstractFloat}

    forward::Bool
    l::Int
    n_stab::Int
    N_stab::Int
    const N::Int
    const β::E
    const Δτ::E
    const Lτ::Int
    const B_bar::Vector{Matrix{T}}
    const F::Vector{LDR{T,E}}
    const G′::Matrix{T}
    const ldr_ws::LDRWorkspace{T}
end


@doc raw"""
    FermionGreensCalculator(
        B::AbstractVector{P},
        β::R, Δτ::R, n_stab::Int
    ) where {T, E, R<:AbstractFloat, P<:AbstractPropagator{T,E}}

Initialize and return [`FermionGreensCalculator`](@ref) struct based on the vector of propagators `B` passed to the function.
"""
function FermionGreensCalculator(
    B::AbstractVector{P},
    β::R, Δτ::R, n_stab::Int
) where {T, E, R<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # get length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get the number of orbitals
    N = size(B[1],1)

    # check that there is a propagator for each time slice
    @assert length(B) == Lτ

    # make sure that the discretization in imaginary time is valid
    @assert (Lτ*Δτ) ≈ β

    # determine type of propagator matrix elements
    H = (T<:Complex || E<:Complex) ? Complex{R} : R

    # calculate the number of numerical stabalization intervals
    N_stab = ceil(Int, Lτ/n_stab)

    # allocate array to represent partical products of B matrices,
    # setting each equal to the identity matrix
    B_bar = Matrix{H}[]
    for n in 1:N_stab
        push!(B_bar, Matrix{H}(I, N, N))
    end

    # calculate scratch matrix
    G′ = Matrix{H}(I, N, N)

    # construct vector of LDR factorization to represent B(τ,0) and B(β,τ) matrices
    ldr_ws = ldr_workspace(G′)
    F = ldrs(G′, N_stab)

    # current imaginary time slice
    l = Lτ

    # whether to iterate over imaginary time τ=Δτ⋅l in forward (l=1 ==> l=Lτ) or reverse order (l=Lτ ==> l=1)
    forward = false

    # allocate FermionGreensCalculator struct
    fgc = FermionGreensCalculator{H,R}(forward, l, n_stab, N_stab, N, β, Δτ, Lτ, B_bar, F, G′, ldr_ws)

    # initialize FermionGreensCalculator struct
    for l in fgc
        update_factorizations!(fgc, B)
    end

    return fgc
end


@doc raw"""
    FermionGreensCalculator(fgc::FermionGreensCalculator{T,E}) where {T,E}

Return a new [`FermionGreensCalculator`](@ref) that is a copy of `fgc`.
"""
function FermionGreensCalculator(fgc::FermionGreensCalculator{T,E}) where {T,E}

    (; forward, l, N, β, Δτ, Lτ, n_stab, N_stab, B_bar, F, G′, ldr_ws) = fgc

    B_bar_new = [copy(B_bar[i]) for i in eachindex(B_bar)]
    F_new = [ldr(F[i]) for i in eachindex(F)]
    G′_new = copy(G′)
    copyto!(G′_new,I)
    ldr_ws_new = ldr_workspace(G′_new)

    return FermionGreensCalculator(forward, l, n_stab, N_stab, N, β, Δτ, Lτ, B_bar_new, F_new, G′_new, ldr_ws_new)
end


########################
## OVERLOADED METHODS ##
########################

@doc raw"""
    eltype(fgc::FermionGreensCalculator{T,E}) where {T,E}

Return matrix element type `T` associated with an instance of [`FermionGreensCalculator`](@ref).
"""
function eltype(fgc::FermionGreensCalculator{T,E}) where {T,E}

    return T
end


@doc raw"""
    resize!(
        fgc::FermionGreensCalculator{T,E},
        G::Matrix{T}, logdetG::E, sgndetG::T,
        B::Vector{P}, n_stab::Int
    ) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T}}

    resize!(
        fgc::FermionGreensCalculator{T,E}, n_stab::Int
    ) where {T,E}

Update `fgc` to reflect a new stabilizaiton frequency `n_stab`.
If `G`, `logdetG`, `sgndetG` and `B` are also passed then the equal-time Green's function `G` is re-calculated
and the corresponding updated values for `(logdetG, sgndetG)` are returned.
"""
function resize!(
    fgc::FermionGreensCalculator{T,E},
    G::Matrix{T}, logdetG::E, sgndetG::T,
    B::Vector{P}, n_stab::Int
) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T}}

    # check if stablization frequency is being updated
    if fgc.n_stab != n_stab

        # resize fgc
        resize!(fgc, n_stab)

        # calculate the equal-time Green's function
        logdetG, sgndetG = calculate_equaltime_greens!(G, fgc, B)
    end

    return (logdetG, sgndetG)
end

function resize!(
    fgc::FermionGreensCalculator{T,E}, n_stab::Int
) where {T,E}

    (; N, Lτ) = fgc
    B_bar = fgc.B_bar::Vector{Matrix{T}}
    F = fgc.F::Vector{LDR{T,E}}

    # check if stablization frequency is being updated
    if fgc.n_stab != n_stab

        # calculate the new number of stabilization intervals
        N_stab = ceil(Int, Lτ/n_stab)

        # calculate the change in stablization intervals
        ΔN_stab = N_stab - fgc.N_stab

        # update stabilization interval and frequency
        fgc.n_stab = n_stab
        fgc.N_stab = N_stab

        # if number of stabilization intervals increased
        if ΔN_stab > 0
            # grow B_bar and F vectors
            for n in 1:ΔN_stab
                B_bar_new = Matrix{T}(I, N, N)
                F_new = ldr(B_bar_new)
                push!(B_bar, B_bar_new)
                push!(F, F_new)
            end
        # if number of stablization intervals decreased
        else
            # shrink B_bar and F vectors
            for n in 1:abs(ΔN_stab)
                B_bar_deleted = pop!(B_bar)
                F_deleted = pop!(F)
            end
        end
    end

    return nothing
end


@doc raw"""
    copyto!(
        fgc_out::FermionGreensCalculator{T,E},
        fgc_in::FermionGreensCalculator{T,E}
    ) where {T,E}

Copy the contents of `fgc_in` to `fgc_out`. If `fgc_out.n_stab != fgc_in.n_stab` is true, then
`fgc_out` will be resized using [`resize!`](@ref) to match `fgc_in`.
"""
function copyto!(
    fgc_out::FermionGreensCalculator{T,E},
    fgc_in::FermionGreensCalculator{T,E}
) where {T,E}

    # resize fgc_out to match fgc_in if necessary
    if fgc_out.n_stab != fgc_in.n_stab
        resize!(fgc_out, fgc_in.n_stab)
    end

    # copy contents of fgc_in to fgc_out
    fgc_out.forward = fgc_in.forward
    fgc_out.l = fgc_in.l
    for i in eachindex(fgc_in.B_bar)
        copyto!(fgc_out.B_bar[i]::Matrix{T}, fgc_in.B_bar[i]::Matrix{T})
        copyto!(fgc_out.F[i]::LDR{T,E}, fgc_in.F[i]::LDR{T,E})
    end
    copyto!(fgc_out.ldr_ws, fgc_in.ldr_ws)

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
    update_factorizations!(
        fgc::FermionGreensCalculator,
        B::AbstractVector{P}
    ) where {P<:AbstractPropagator}

If current imaginary time slice `fgc.l` corresponds to the boundary of a stabilization interval,
calculate a LDR factorization to represent ``B(0, \tau)`` or ``B(\tau-\Delta\tau, \beta)``
if iterating over imaginary time in the forward (`fgc.forward = true`)
or reverse (`fgc.forward = false`) directions respectively.
This method should be called *after* all changes to the current time slice propagator matrix
``B_l`` have been made
This method will also recompute ``\bar{B}_n`` as needed.
"""
function update_factorizations!(
    fgc::FermionGreensCalculator,
    B::AbstractVector{P}
) where {P<:AbstractPropagator}

    # update B_bar matrix when necessary
    update_B̄!(fgc, B)

    # update the factorization
    update_factorizations!(fgc)

    return nothing
end

@doc raw"""
    update_factorizations!(
        fgc::FermionGreensCalculator{T,E}
    ) where {T, E}

If current imaginary time slice `fgc.l` corresponds to the boundary of a stabilization interval,
calculate a LDR factorization to represent ``B(\tau, 0)`` or ``B(\beta, \tau-\Delta\tau)``
if iterating over imaginary time in the forward (`fgc.forward = true`)
or reverse (`fgc.forward = false`) directions respectively.
This method should be called *after* all changes to the current time slice propagator matrix
``B_l`` have been made, and any required updates to a ``\bar{B}_n`` matrix have
been performed using the [`JDQMCFramework.update_B̄!`](@ref) routine.
"""
function update_factorizations!(
    fgc::FermionGreensCalculator{T,E}
) where {T, E}

    (; l, Lτ, n_stab, N_stab, forward) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}
    B_bar = fgc.B_bar::Vector{Matrix{T}}

    # get stabalizaiton interval
    n, l′ = stabilization_interval(fgc)

    # if iterating from l=1 => l=Lτ
    if forward
        # if at boundary of first stabilization interval (l=n_stab)
        if l′==n_stab && n==1
            # calculate LDR factorization of B(τ=nₛ⋅Δτ,0) = B̄[1]
            B̄₁ = B_bar[1]
            B_τ0 = F[1]::LDR{T,E}
            ldr!(B_τ0, B̄₁, ldr_ws)
        # if at the end of a stabilization interval
        elseif l′==n_stab || l==Lτ && N_stab > 1
            # calculate LDR factorization of B(τ=n⋅nₛ⋅Δτ,0) = B̄[n]⋅B(τ=(n-1)⋅nₛ⋅Δτ,0)
            B̄ₙ = B_bar[n]
            B_τ0_new = F[n]::LDR{T,E}
            B_τ0_prev = F[n-1]::LDR{T,E}
            mul!(B_τ0_new, B̄ₙ, B_τ0_prev, ldr_ws)
        end
    # if iterating from l=Lτ => l=1
    else
        # if at boundary of last stabilization interval (l=Lτ-n_stab+1)
        if l′==1 && n==N_stab
            # calculate LDR factorization of B(β,τ=β-(n_stab+1)⋅Δτ) = B̄[N_stab]
            B̄_Nₛ = B_bar[n]
            B_βτ = F[n]::LDR{T,E}
            ldr!(B_βτ, B̄_Nₛ, ldr_ws)
        # if at boundary of stabilization interval
        elseif l′==1 && N_stab > 1
            # calculate LDR factorization of B(β,τ=β-n⋅nₛ⋅Δτ-Δτ) = B(β,τ=β-n⋅nₛ⋅Δτ)⋅B̄[n]
            B̄ₙ = B_bar[n]
            B_βτ_new = F[n]::LDR{T,E}
            B_βτ_prev = F[n+1]::LDR{T,E}
            mul!(B_βτ_new, B_βτ_prev, B̄ₙ, ldr_ws)
        end
    end

    return nothing
end


@doc raw"""
    update_B̄!(
        fgc::FermionGreensCalculator,
        B::AbstractVector{P}
    ) where {P<:AbstractPropagator}

Recalculate ``\bar{B}_n`` if the current timeslice `fgc.l` corresponds to the boundary of a stabilization interval,
accounting for whether imaginary time is being iterated over in the forward (`fgc.forward = true`) or
reverse (`fgc.forward = false`) direction.
"""
function update_B̄!(
    fgc::FermionGreensCalculator,
    B::AbstractVector{P}
) where {P<:AbstractPropagator}

    (; forward, n_stab, l, Lτ) = fgc

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # if iterating over imaginary time in the forward direction l=1 ==> l=Lτ
    if forward
        # if at boundary of stabilization interval
        if l′ == n_stab || l==Lτ
            # update B_bar[n]
            calculate_B̄!(fgc, B, n)
        end
    # if iterating over imaginary time in the reviews direction l=Lτ ==> l=1
    else
        # if at boundary of stabilization interval
        if l′ == 1
            # update B_bar[n]
            calculate_B̄!(fgc, B, n)
        end
    end

    return nothing
end


@doc raw"""
    calculate_B̄!(
        fgc::FermionGreensCalculator,
        B::AbstractVector{P},
        n::Int
    ) where {P<:AbstractPropagator{T}}

Given `B`, a vector of all the propagator matrices ``B_l``, calculate the matrix product
```math
\bar{B}_{\sigma,n}=\prod_{l=(n-1)\cdot n_{s}+1}^{\min(n\cdot n_{s},L_{\tau})}B_{\sigma,l},
```
with the result getting written to `fgc.B_bar[n]`.
"""
function calculate_B̄!(
    fgc::FermionGreensCalculator,
    B::AbstractVector{P},
    n::Int
) where {P<:AbstractPropagator}

    (; B_bar, n_stab, Lτ, N_stab, ldr_ws) = fgc
    @assert 1 <= n <= N_stab
    B̄ₙ = B_bar[n]
    copyto!(B̄ₙ, I) # B̄ₙ := I
    # iterate over imaginary time slices associated with stabilization interval
    for l in min(n*n_stab,Lτ):-1:(n-1)*n_stab+1
        B_l = B[l]::P
        rmul!(B̄ₙ, B_l, M = ldr_ws.M) # B̄ₙ := B̄ₙ⋅B[l]
    end

    return nothing
end


@doc raw"""
    stabilization_interval(
        fgc::FermionGreensCalculator
    )::Tuple{Int,Int}

Given the current imaginary time slice `fgc.l`, return the corresponding
stabilization interval `n = ceil(Int, fgc.l/fgc.n_stab)`, and the relative location
within that stabilization interval `l′ = mod1(fgc.l, fgc.n_stab)`, such that `l′∈[1,n_stab]`. 
"""
function stabilization_interval(
    fgc::FermionGreensCalculator
)::Tuple{Int,Int}

    # calculate stabilization interval
    n = ceil(Int, fgc.l/fgc.n_stab)
    # location in stabilization interval l′∈[1,n_stab]
    l′ = mod1(fgc.l, fgc.n_stab)

    return n, l′
end