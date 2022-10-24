@doc raw"""
    calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T})::Tuple{T,E} where {T}

Calculate the equal-time Greens function ``G(0,0) = G(\beta,\beta) = [I + B(0,β)]^{-1}`` using a numerically stable procedure.
This method also returns ``\log(\vert \det G \vert)`` and ``\textrm{sign}(\det G).``
Note that this routine requires `fgc.l == 1` or `fgc.l == fgc.Lτ`.
"""
function calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E})::Tuple{E,T} where {T,E}

    (; forward, l, Lτ, Nₛ, ldr_ws) = fgc
    @assert l == 1 || l == Lτ

    # get B(0,β)
    if forward
        B_0β = fgc.F[1]::LDR{T,E}
    else
        B_0β = fgc.F[Nₛ]::LDR{T,E}
    end

    # calculate G(0,0) = [I + B(0,β)]⁻¹
    logdetG, sgndetG = inv_IpA!(G, B_0β, ldr_ws)

    return logdetG, sgndetG
end

@doc raw"""
    calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E},
                                B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T}}

Calculate the equal-time Greens function ``G(0,0) = G(\beta,\beta) = [I + B(0,β)]^{-1}`` using a numerically stable procedure.
Also re-calculate the ``\bar{B}_n`` matrices and the LDR matrix factorizations representing
either ``B(0,\tau)`` or ``B(\tau,\beta)`` stored in `fgc.F`. This routine is useful for implementing global updates where every
propagator matrix ``B_l`` has been modified, and the equal-time Green's function needs to be re-calculated from scratch.
This method also returns ``\log(\vert \det G \vert)`` and ``\textrm{sign}(\det G).``
Note that this routine requires `fgc.l == 1` or `fgc.l == fgc.Lτ`.
"""
function calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E},
                                     B::AbstractVector{P})::Tuple{E,T} where {T, E, P<:AbstractPropagator{T}}

    # iterate over imaginary time
    for l in fgc
        # re-calculate B̄ matrices and matrix factorizations B(0,τ) or B(τ,β) as needed
        update_factorizations!(fgc, B)
    end
    
    # calculate equal-time Greens funciton matrix G(0,0) = G(β,β)
    logdetG, sgndetG = calculate_equaltime_greens!(G, fgc)
    
    return logdetG, sgndetG
end


@doc raw"""
    calculate_unequaltime_greens!(G0τ::AbstractArray{T,3}, Gττ::AbstractArray{T,3},
                                  fgc::FermionGreensCalculator{T,E},
                                  B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T}}

Calculate the unequal-time Green's functions ``G(0,\tau)`` and the equal-time Green's function ``G(\tau,\tau)``
for all ``\tau \in \{ 0, \Delta\tau, 2\Delta\tau, \dots, \beta-\Delta\tau, \beta \}.``
The ``G(0,\tau)`` unequal-time Green's function is written to `G0τ[:,:,l+1]`,
and the ``G(\tau,\tau)`` equal-time Green's function is written to `Gττ[:,:,l+1]`,
where ``\tau = \Delta\tau \cdot l`` for ``l \in [0, L_τ].``
Therefore,
```julia
size(G0τ,1) == size(G0τ,2) == size(Gττ,1) == size(Gττ,2) == fgc.N
```
and
```julia
size(G0τ, 3) == size(Gττ, 3) == fgc.Lτ+1
```
return true. Note that ``G(0,0) = G(\beta,\beta)``, which means that
```julia
Gττ[1, 1] ≈ Gττ[fgc.Lτ+1, fgc.Lτ+1]
```
is true.
"""
function calculate_unequaltime_greens!(G0τ::AbstractArray{T,3}, Gττ::AbstractArray{T,3},
                                       fgc::FermionGreensCalculator{T,E},
                                       B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T,E}}

    @assert fgc.l == 1 || fgc.l == fgc.Lτ
    @assert size(G0τ,1) == size(G0τ,2) == fgc.N
    @assert size(G0τ,3) == fgc.Lτ+1
    @assert size(Gττ,1) == size(Gττ,2) == fgc.N
    @assert size(Gττ,3) == fgc.Lτ+1

    (; Lτ, Nₛ, nₛ, forward) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # calculate equal-time Green's function G(0,0) = G(β,β)
    G0τ_0 = @view G0τ[:,:,1]
    Gττ_0 = @view Gττ[:,:,1]
    Gττ_β = @view Gττ[:,:,Lτ+1]
    calculate_equaltime_greens!(G0τ_0, fgc)
    copyto!(Gττ_0, G0τ_0)
    copyto!(Gττ_β, G0τ_0) # G(β,β) = G(0,0)

    # apply anti-periodic boundary conditions in imaginary time: G(0,β) = I - G(0,0)
    G0τ_β = @view G0τ[:,:,Lτ+1]
    @. G0τ_β = -G0τ_0
    @fastmath @inbounds for i in axes(G0τ_β, 1)
        G0τ_β[i,i] += 1
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # iterate over stabilization intervals
        for n in 1:Nₛ
            # iterate over imaginary time slices in stabilization interval
            for l in (n-1)*nₛ+1:min(n*nₛ,Lτ)-1
                # get B[l]
                B_l = B[l]::P
                # propagate to later imaginary time G(0,τ) = G(0,τ-Δτ)⋅B[l]
                G0τ_τmΔτ = @view G0τ[:,:,l] # G(0,τ-Δτ)
                G0τ_τ = @view G0τ[:,:,l+1] # G(0,τ)
                mul!(G0τ_τ, G0τ_τmΔτ, B_l, M = ldr_ws.M)
                # propagate to later imaginary time G(τ,τ) = B⁻¹[l]⋅G(τ-Δτ,τ-Δτ)⋅B[l]
                Gττ_τmΔτ = @view Gττ[:,:,l]
                Gττ_τ = @view Gττ[:,:,l+1]
                mul!(Gττ_τ, Gττ_τmΔτ, B_l, M = ldr_ws.M) # G(τ-Δτ,τ-Δτ)⋅B[l]
                ldiv!(B_l, Gττ_τ, M = ldr_ws.M) # B⁻¹[l]⋅G(τ-Δτ,τ-Δτ)⋅B[l]
            end
            # update LDR factorization for B(0,τ=Δτ⋅n⋅nₛ)
            fgc.l = min(n*nₛ,Lτ)
            update_factorizations!(fgc)
            # calculate G(0,τ=Δτ⋅n⋅nₛ) and G(τ=Δτ⋅n⋅nₛ,τ=Δτ⋅n⋅nₛ)
            if n < Nₛ
                B_0τ = F[n]::LDR{T,E} # B(0,τ)
                B_τβ = F[n+1]::LDR{T,E} # B(τ,β)
                # calculate G(0,τ=Δτ⋅n⋅nₛ) using numerically stable procedure
                G0τ_τ = @view G0τ[:,:,fgc.l+1] # G(0,τ)
                inv_invUpV!(G0τ_τ, B_0τ, B_τβ, ldr_ws) # G(0,τ) = [B⁻¹(0,τ) + B(τ,β)]⁻¹
                # calculate G(τ=Δτ⋅n⋅nₛ,τ=Δτ⋅n⋅nₛ)
                Gττ_τ = @view Gττ[:,:,fgc.l+1]
                inv_IpUV!(Gττ_τ, B_τβ, B_0τ, ldr_ws) # G(τ,τ) = [I + B(τ,β)⋅B(0,τ)]⁻¹
            end
        end
        # reset forward to true (next iteration over imaginary time will be in the reverse direction)
        fgc.forward = false
    # if iterating from l=Lτ => l=1
    else
        # iterate over stabilization intervals
        for n in Nₛ:-1:1
            # calculate LDR factorization for B(τ=Δτ⋅n⋅nₛ-Δτ, β)
            fgc.l = (n-1)*nₛ+1
            update_factorizations!(fgc)
            # calculate G(0,τ=Δτ⋅(n-1)⋅nₛ) and G(τ=Δτ⋅(n-1)⋅nₛ,τ=Δτ⋅(n-1)⋅nₛ) using numerically stable procedure
            if n > 1
                B_τβ = F[n]::LDR{T,E} # B(τ=Δτ⋅(n-1)⋅nₛ, β)
                B_0τ = F[n-1]::LDR{T,E} # B(0, τ=Δτ⋅(n-1)⋅nₛ)
                # calculate G(0,τ=Δτ⋅(n-1)⋅nₛ) using numerically stable procedure
                G0τ_τ = @view G0τ[:,:,fgc.l] # G(0, τ=Δτ⋅(n-1)⋅nₛ)
                inv_invUpV!(G0τ_τ, B_0τ, B_τβ, ldr_ws) # G(0,τ) = [B⁻¹(0,τ) + B(τ,β)]⁻¹
                # calculate G(τ=Δτ⋅(n-1)⋅nₛ,τ=Δτ⋅(n-1)⋅nₛ) using numerically stable procedure
                Gττ_τ = @view Gττ[:,:,fgc.l]
                inv_IpUV!(Gττ_τ, B_τβ, B_0τ, ldr_ws) # G(τ,τ) = [I + B(τ,β)⋅B(0,τ)]⁻¹
            end
            # iterate over imaginary time slice in stabilization interval
            for l in (n-1)*nₛ+1:min(n*nₛ,Lτ)-1
                # get B[l]
                B_l = B[l]::P
                # propagate to later imaginary time G(0,τ) = G(0,τ-Δτ)⋅B[l]
                G0τ_τmΔτ = @view G0τ[:,:,l] # G(0,τ-Δτ)
                G0τ_τ = @view G0τ[:,:,l+1] # G(0,τ)
                mul!(G0τ_τ, G0τ_τmΔτ, B_l, M = ldr_ws.M)
                # propagate to later imaginary time G(τ,τ) = B⁻¹[l]⋅G(τ-Δτ,τ-Δτ)⋅B[l]
                Gττ_τmΔτ = @view Gττ[:,:,l]
                Gττ_τ = @view Gττ[:,:,l+1]
                mul!(Gττ_τ, Gττ_τmΔτ, B_l, M = ldr_ws.M) # G(τ-Δτ,τ-Δτ)⋅B[l]
                ldiv!(B_l, Gττ_τ, M = ldr_ws.M) # B⁻¹[l]⋅G(τ-Δτ,τ-Δτ)⋅B[l]
            end
        end
        # reset forward to true (next iteration over imaginary time will be in the forward direction)
        fgc.forward = true
    end

    return nothing
end


@doc raw"""
    propagate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E},
                                B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T}}

Propagate the equal-time Green's function matrix `G` from the previous imaginary time slice to the current
imaginary time slice `fgc.l`. If iterating over imaginary time in the forward direction (`fgc.forward = true`)
the relationship
```math
G(\tau+\Delta\tau,\tau+\Delta\tau)=B_{l+1}^{-1} \cdot G(\tau,\tau) \cdot B_{l+1}
```
is used, and if iterating over imaginary time in the reverse direction (`fgc.forward = false`)
the relationship
```math
G(\tau-\Delta\tau,\tau-\Delta\tau)=B_{l} \cdot G(\tau,\tau) \cdot B_{l}^{-1}
```
is used instead, where the ``B_l`` propagator is given by `B[l]`.
"""
function propagate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E},
                                     B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T,E}}

    (; nₛ, l, Lτ, forward) = fgc
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}
    M = ldr_ws.M::Matrix{T}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # if iterating from l=1 => l=Lτ
    if forward
        # G(l,l) = B⁻¹[l]⋅G(l-1,l-1)⋅B[l]
        B_l = B[l]::P
        ldiv!(B_l, G, M = M)
        rmul!(G, B_l, M = M)
    # if iterating from l=Lτ => l=1.
    # Note: if `fgc.l` corresponds to the "end" of a stabilization interval (l′=nₛ or l=Lτ), then
    # when the equal-time Green's function was previously re-computed at `fgc.l+1` in the stabilize_equatime_greens!()
    # routine, it already calculated G(l,l), so we don't need to propagate to G(l,l) here.
    elseif !(l′ == nₛ || l==Lτ)
        # G(l,l) = B[l+1]⋅G(l+1,l+1)⋅B⁻¹[l+1]
        B_lp1 = B[l+1]::P
        lmul!(B_lp1, G, M = M)
        rdiv!(G, B_lp1, M = M)
    end

    return nothing
end


@doc raw"""
    stabilize_equaltime_greens!(G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
                                fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P};
                                update_B̄::Bool=true)::Tuple{E,T} where {T, E, P<:AbstractPropagator{T,E}}

Stabilize the equal-time Green's function as iterating through imaginary time ``\tau = \Delta\tau \cdot l.``
For a given imaginary time slice `fgc.l`, this routine should be called *after* all changes to the ``B_l``
propagator have been made.
When iterating through imaginary time in the forwards direction (`fgc.forward = true`), this function
re-computes
```math
G(\tau,\tau) = [I + B(\tau,\beta)B(0,\tau)]^{-1}
```
when at imaginary time slice `fgc.l` every `fgc.nₛ` imaginary time slice.
When iterating through imaginary time in the reverse direction (`fgc.forward = false`), this function
instead re-computes
```math
G(\tau,\tau) = [I + B(\tau-\Delta\tau,\beta)B(0,\tau-\Delta\tau)]^{-1}
```
for `fgc.l`.
The method computes the required ``\bar{B}_n`` matrices, and the LDR matrix factorizations
representing ``B(0,\tau)`` or ``B(\tau-\Delta\tau, \beta)`` when iterating through imaginary
time ``\tau = \Delta\tau \cdot l`` in the forward and reverse directions respectively.
If `update_B̄ = true`, then the ``\bar{B}_n`` matrices are re-calculated as needed, but if
`update_B̄ = false,` then they are left unchanged.
"""
function stabilize_equaltime_greens!(G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
                                     fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P};
                                     update_B̄::Bool=true)::Tuple{E,T} where {T, E, P<:AbstractPropagator{T,E}}

    (; l, Lτ, forward, nₛ, Nₛ) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # update B(0,τ) if iterating forward or B(τ-Δτ,β) if iterating backwards
    if update_B̄
        # also re-calculate B̄ matrices
        update_factorizations!(fgc, B)
    else
        # do not update B̄ matrices
        update_factorizations!(fgc)
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # if at last time slice calculate G(β,β)
        if l == Lτ
            # G(β,β) = [I + B(0,β)]⁻¹
            B_0_β = F[Nₛ]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(G, B_0_β, ldr_ws)

        # if at boundary of stablization interval calculate G(τ,τ)
        elseif l′ == nₛ
            # calculate G(τ,τ) = [I + B(τ,β)⋅B(0,τ)]⁻¹
            B_τ_β = F[n+1]::LDR{T,E}
            B_0_τ = F[n]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(G, B_τ_β, B_0_τ, ldr_ws)
        end
    # if iterating from l=Lτ => l=1
    else
        # if at first time slice calculate G(0,0) = G(β,β)
        if l == 1
            # G(β,β) = [I + B(0,β)]⁻¹
            B_0_β = F[1]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(G, B_0_β, ldr_ws)

        # if at boundary of stabilization interval calculate G(τ-Δτ,τ-Δτ)
        elseif l′ == 1
            # calucate G(τ-Δτ,τ-Δτ) = [I + B(τ-Δτ,β)⋅B(0,τ-Δτ)]⁻¹
            B_τmΔτ_β = F[n]::LDR{T,E}
            B_0_τmΔτ = F[n-1]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(G, B_τmΔτ_β, B_0_τmΔτ, ldr_ws)
        end
    end
    
    return (logdetG, sgndetG)
end


@doc raw"""
    local_update_det_ratio(G::AbstractMatrix{T}, B::AbstractPropagator{T,E},
                           V′::T, i::Int, Δτ::E)::Tuple{T,T} where {T,E}

Calculate the determinant ratio ``R_{l,i}`` associated with a local update to the equal-time
Green's function ``G(\tau,\tau).`` Also returns ``\Delta_{l,i},`` which is defined below.

# Arguments

- `G::AbstractMatrix{T}`: Equal-time Green's function matrix ``G(\tau,\tau).``
- `B::AbstractPropagator{T,E}`: Represents the propagator matrix ``B_l,`` where ``\tau = \Delta\tau \cdot l.``
- `V′::T`: The new value for the ``V^{\prime}_{l,(i,i)}`` matrix element in the diagonal on-site energy matrix ``V_l.``
- `i::Int`: Diagonal matrix element index in ``V_l`` being updated.
- `Δτ::E`: Discretization in imaginary time ``\Delta\tau.``

# Algorithm

The propagator matrix ``B_l`` above is given by
```math
B_l = \Gamma_l(\Delta\tau) \cdot \Lambda_l,
```
where, assuming the we are working in the orbital basis, ``\Gamma_l(\Delta\tau) = e^{-\Delta\tau K_l}``
represents the exponentiated hopping matrix ``K_l``, and ``\Lambda_l = e^{-\Delta\tau V_l}`` represents
the exponentiated diagonal on-site energy matrix ``V_l.``

Given an update to the ``(i,i)`` matrix element of the diagonal on-site energy matrix ``V_l``,
``V_{l,(i,i)} \rightarrow V^\prime_{l,(i,i)},`` the corresponding determinant ratio associated
with this change is given by
```math
R_{l,i} = \frac{\det G(\tau,\tau)}{\det G^\prime({\tau,\tau})} = 1 + (1-G_{(i,i)}(\tau,\tau)) \Delta_{l,i},
```
where
```math
\Delta_{l,i} = e^{-\Delta\tau (V^\prime_{l,(i,i)} - V_{l,(i,i)})} - 1.
```
This routine returns the scalar quantities ``R_{l,i}`` and ``\Delta_{l,i}.``

Note that if the propagator matrix is instead represented using the symmetric form
```math
B_l = \Gamma_l(\Delta\tau/2) \cdot \Lambda_l \cdot \Gamma^\dagger_l(\Delta\tau/2),
```
then the matrix `G` needs to instead represent the transformed equal-time Green's function matrix
```math
\tilde{G}(\tau,\tau) = \Gamma_l^\dagger(\Delta\tau/2) \cdot G(\tau,\tau) \cdot \Gamma_l^{-\dagger}(\Delta\tau/2).
```
"""
function local_update_det_ratio(G::AbstractMatrix{T}, B::AbstractPropagator{T,E},
                                V′::T, i::Int, Δτ::E)::Tuple{T,T} where {T,E}

    Λ = B.expmΔτV
    Δ = exp(-Δτ*V′)/Λ[i] - 1
    R = 1 + (1 - G[i,i])*Δ

    return (R, Δ)
end


@doc raw"""
    local_update_greens!(G′::AbstractMatrix{T}, G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
                         R::T, Δ::T, i::Int)::Tuple{E,T} where {T, E<:AbstractFloat}

Update the equal-time Green's function matrix resulting from a local update.

# Arguments

- `G′::AbstractMatrix{T}`: Updated equal-time Green's function matrix ``G^\prime(\tau,\tau)``, which is modified in-place.
- `G::AbstractMatrix{T}`: Initial equal-time Green's function matrix ``G(\tau,\tau).``
- `logdetG::E`: The log of the absolute value of the initial Green's function matrix, ``\log( \vert \det G(\tau,\tau) \vert ).``
- `sgndetG::T`: The sign/phase of the determinant of the initial Green's function matrix, ``\textrm{sign}( \det G(\tau,\tau) ).``
- `R::T`: The determinant ratio ``\frac{\det G(\tau,\tau)}{\det G^\prime(\tau,\tau)}.``
- `Δ::T`: Change in the exponentiated on-site energy matrix, ``\Delta_{l,i} = e^{-\Delta\tau (V^\prime_{l,(i,i)} - V_{l,(i,i)})} - 1.``
- `i::Int`: Matrix element of diagonal on-site energy matrix ``V_l`` that is being updated.

# Algorithm

The equal-time Green's function matrix is updated using the relationship
```math
G^\prime_{k,j}(\tau,\tau) = G_{k,j}(\tau,\tau) - (\delta_{k,i} - G_{k,i}(\tau,\tau)) \Delta_{l,i} G_{i,j}(\tau,\tau) R_{l,i}^{-1}.
```
This method also returns ``\log( \vert \det G^\prime(\tau,\tau) \vert )`` and ``\textrm{sign}( \det G^\prime(\tau,\tau) ).``

An important note is that if the propagator matrices are represented in a symmetric form, then `G′` and `G` need to correspond
to the transformed eqaul-time Green's function matrices ``\tilde{G}^\prime(\tau,\tau)`` and ``\tilde{G}(\tau,\tau).``
Refer to the [`local_update_det_ratio`](@ref) docstring for more information.
"""
function local_update_greens!(G′::AbstractMatrix{T}, G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
                              R::T, Δ::T, i::Int)::Tuple{E,T} where {T, E<:AbstractFloat}

    @fastmath @inbounds for j in axes(G,2)
        for k in axes(G,1)
            G′[k,j] = G[k,j] - (I[k,i]-G[k,i])*Δ*G[i,j]*inv(R) # Note: I[k,j] = δₖⱼ
        end
    end

    # R = det(M′)/det(M) = det(G)/det(G′)
    # ==> log(|R|) = log(|det(M′)|) - log(|det(M)|) = log(|det(G)|) - log(|det(G′)|)
    # ==> log(|det(G′)|) = log(|det(G)|) - log(|R|)
    logdetG′ = logdetG - log(abs(R))
    sgndetG′ = sign(R) * sgndetG

    return (logdetG′, sgndetG′)
end