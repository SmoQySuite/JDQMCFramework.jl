@doc raw"""
    calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T})::Tuple{T,E} where {T}

Calculate the equal-time Greens function ``G(0,0) = G(\beta,\beta) = [I + B(\beta,0)]^{-1}`` using a numerically stable procedure.
This method also returns ``\log(\vert \det G \vert)`` and ``\textrm{sign}(\det G).``
Note that this routine requires `fgc.l == 1` or `fgc.l == fgc.Lτ`.
"""
function calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E})::Tuple{E,T} where {T,E}

    (; forward, l, Lτ, Nₛ, ldr_ws) = fgc
    @assert l == 1 || l == Lτ

    # get B(0,β)
    if forward
        B_β0 = fgc.F[1]::LDR{T,E}
    else
        B_β0 = fgc.F[Nₛ]::LDR{T,E}
    end

    # calculate G(0,0) = [I + B(β,0)]⁻¹
    logdetG, sgndetG = inv_IpA!(G, B_β0, ldr_ws)

    return logdetG, sgndetG
end

@doc raw"""
    calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E},
                                B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T}}

Calculate the equal-time Greens function ``G(0,0) = G(\beta,\beta) = [I + B(\beta,0)]^{-1}`` using a numerically stable procedure.
Also re-calculate the ``\bar{B}_n`` matrices and the LDR matrix factorizations representing
either ``B(\tau,0)`` or ``B(\beta,\tau)`` stored in `fgc.F`. This routine is useful for implementing global updates where every
propagator matrix ``B_l`` has been modified, and the equal-time Green's function needs to be re-calculated from scratch.
This method also returns ``\log(\vert \det G \vert)`` and ``\textrm{sign}(\det G).``
Note that this routine requires `fgc.l == 1` or `fgc.l == fgc.Lτ`.
"""
function calculate_equaltime_greens!(G::AbstractMatrix{T}, fgc::FermionGreensCalculator{T,E},
                                     B::AbstractVector{P})::Tuple{E,T} where {T, E, P<:AbstractPropagator{T}}

    # iterate over imaginary time
    for l in fgc
        # re-calculate B̄ matrices and matrix factorizations B(τ,0) or B(β,τ) as needed
        update_factorizations!(fgc, B)
    end
    
    # calculate equal-time Greens funciton matrix G(0,0) = G(β,β)
    logdetG, sgndetG = calculate_equaltime_greens!(G, fgc)
    
    return logdetG, sgndetG
end


@doc raw"""
    calculate_unequaltime_greens!(Gτ0::AbstractArray{T,3},
                                  fgc::FermionGreensCalculator{T,E},
                                  B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T}}

Calculate the unequal-time Green's functions ``G(\tau,0)``
for all ``\tau \in \{ 0, \Delta\tau, 2\Delta\tau, \dots, \beta-\Delta\tau, \beta \}.``
The ``G(\tau,0)`` unequal-time Green's function is written to `Gτ0[:,:,l+1]`,
where ``\tau = \Delta\tau \cdot l`` for ``l \in [0, L_τ].``
Therefore,
```julia
size(Gτ0,1) == size(Gτ0,2) == fgc.N
```
and
```julia
size(Gτ0, 3) == == fgc.Lτ+1
```
return true.
"""
function calculate_unequaltime_greens!(Gτ0::AbstractArray{T,3},
                                       fgc::FermionGreensCalculator{T,E},
                                       B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T,E}}

    @assert fgc.l == 1 || fgc.l == fgc.Lτ
    @assert size(Gτ0,1) == size(Gτ0,2) == fgc.N
    @assert size(Gτ0,3) == fgc.Lτ+1

    (; Lτ, Nₛ, nₛ, forward) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # calculate equal-time Green's function G(0,0) = G(β,β)
    Gτ0_0 = @view Gτ0[:,:,1]
    calculate_equaltime_greens!(Gτ0_0, fgc)

    # apply anti-periodic boundary conditions in imaginary time: G(β,0) = I - G(0,0)
    Gτ0_β = @view Gτ0[:,:,Lτ+1]
    @. Gτ0_β = -Gτ0_0
    @fastmath @inbounds for i in axes(Gτ0_β, 1)
        Gτ0_β[i,i] += 1
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # iterate over stabilization intervals
        for n in 1:Nₛ
            # iterate over imaginary time slices in stabilization interval
            for l in (n-1)*nₛ+1:min(n*nₛ,Lτ)-1
                # get B[l]
                B_l = B[l]::P
                # propagate to later imaginary time G(τ,0) = B[l]⋅G(τ-Δτ0)
                Gτ0_τmΔτ = @view Gτ0[:,:,l] # G(τ-Δτ,0)
                Gτ0_τ = @view Gτ0[:,:,l+1] # G(τ,0)
                mul!(Gτ0_τ, B_l, Gτ0_τmΔτ, M = ldr_ws.M)
            end
            # update LDR factorization for B(τ=Δτ⋅n⋅nₛ,0)
            fgc.l = min(n*nₛ,Lτ)
            update_factorizations!(fgc)
            # calculate G(τ=Δτ⋅n⋅nₛ,0) and G(τ=Δτ⋅n⋅nₛ,τ=Δτ⋅n⋅nₛ)
            if n < Nₛ
                B_τ0 = F[n]::LDR{T,E} # B(τ,0)
                B_βτ = F[n+1]::LDR{T,E} # B(β,τ)
                # calculate G(τ=Δτ⋅n⋅nₛ,0) using numerically stable procedure
                Gτ0_τ = @view Gτ0[:,:,fgc.l+1] # G(τ,0)
                inv_invUpV!(Gτ0_τ, B_τ0, B_βτ, ldr_ws) # G(0,τ) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
            end
        end
        # reset forward to true (next iteration over imaginary time will be in the reverse direction)
        fgc.forward = false
    # if iterating from l=Lτ => l=1
    else
        # iterate over stabilization intervals
        for n in Nₛ:-1:1
            # calculate LDR factorization for B(β, τ=Δτ⋅n⋅nₛ-Δτ,)
            fgc.l = (n-1)*nₛ+1
            update_factorizations!(fgc)
            # calculate G(τ=Δτ⋅(n-1)⋅nₛ, 0) and G(τ=Δτ⋅(n-1)⋅nₛ,τ=Δτ⋅(n-1)⋅nₛ) using numerically stable procedure
            if n > 1
                B_βτ = F[n]::LDR{T,E} # B(β, τ=Δτ⋅(n-1)⋅nₛ)
                B_τ0 = F[n-1]::LDR{T,E} # B(τ=Δτ⋅(n-1)⋅nₛ, 0,)
                # calculate G(τ=Δτ⋅(n-1)⋅nₛ, 0) using numerically stable procedure
                Gτ0_τ = @view Gτ0[:,:,fgc.l] # G(τ=Δτ⋅(n-1)⋅nₛ, 0,)
                inv_invUpV!(Gτ0_τ, B_τ0, B_βτ, ldr_ws) # G(0,τ) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
            end
            # iterate over imaginary time slice in stabilization interval
            for l in (n-1)*nₛ+1:min(n*nₛ,Lτ)-1
                # get B[l]
                B_l = B[l]::P
                # propagate to later imaginary time G(τ,0) = B[l]⋅G(τ-Δτ,0)
                Gτ0_τmΔτ = @view Gτ0[:,:,l] # G(τ-Δτ,0)
                Gτ0_τ = @view Gτ0[:,:,l+1] # G(τ,0)
                mul!(Gτ0_τ, B_l, Gτ0_τmΔτ, M = ldr_ws.M)
            end
        end
        # reset forward to true (next iteration over imaginary time will be in the forward direction)
        fgc.forward = true
    end

    return nothing
end


@doc raw"""
    calculate_unequaltime_greens!(Gτ0::AbstractArray{T,3}, Gττ::AbstractArray{T,3},
                                  fgc::FermionGreensCalculator{T,E},
                                  B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T}}

Calculate the unequal-time Green's functions ``G(\tau,0)`` and the equal-time Green's function ``G(\tau,\tau)``
for all ``\tau \in \{ 0, \Delta\tau, 2\Delta\tau, \dots, \beta-\Delta\tau, \beta \}.``
The ``G(\tau,0)`` unequal-time Green's function is written to `Gτ0[:,:,l+1]`,
and the ``G(\tau,\tau)`` equal-time Green's function is written to `Gττ[:,:,l+1]`,
where ``\tau = \Delta\tau \cdot l`` for ``l \in [0, L_τ].``
Therefore,
```julia
size(Gτ0,1) == size(Gτ0,2) == size(Gττ,1) == size(Gττ,2) == fgc.N
```
and
```julia
size(Gτ0, 3) == size(Gττ, 3) == fgc.Lτ+1
```
return true. Note that ``G(0,0) = G(\beta,\beta)``, which means that
```julia
Gττ[1, 1] ≈ Gττ[fgc.Lτ+1, fgc.Lτ+1]
```
is true.
"""
function calculate_unequaltime_greens!(Gτ0::AbstractArray{T,3}, Gττ::AbstractArray{T,3},
                                       fgc::FermionGreensCalculator{T,E},
                                       B::AbstractVector{P}) where {T, E, P<:AbstractPropagator{T,E}}

    @assert fgc.l == 1 || fgc.l == fgc.Lτ
    @assert size(Gτ0,1) == size(Gτ0,2) == fgc.N
    @assert size(Gτ0,3) == fgc.Lτ+1
    @assert size(Gττ,1) == size(Gττ,2) == fgc.N
    @assert size(Gττ,3) == fgc.Lτ+1

    (; Lτ, Nₛ, nₛ, forward) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # calculate equal-time Green's function G(0,0) = G(β,β)
    Gτ0_0 = @view Gτ0[:,:,1]
    Gττ_0 = @view Gττ[:,:,1]
    Gττ_β = @view Gττ[:,:,Lτ+1]
    calculate_equaltime_greens!(Gτ0_0, fgc)
    copyto!(Gττ_0, Gτ0_0)
    copyto!(Gττ_β, Gτ0_0) # G(β,β) = G(0,0)

    # apply anti-periodic boundary conditions in imaginary time: G(β,0) = I - G(0,0)
    Gτ0_β = @view Gτ0[:,:,Lτ+1]
    @. Gτ0_β = -Gτ0_0
    @fastmath @inbounds for i in axes(Gτ0_β, 1)
        Gτ0_β[i,i] += 1
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # iterate over stabilization intervals
        for n in 1:Nₛ
            # iterate over imaginary time slices in stabilization interval
            for l in (n-1)*nₛ+1:min(n*nₛ,Lτ)-1
                # get B[l]
                B_l = B[l]::P
                # propagate to later imaginary time G(τ,0) = B[l]⋅G(τ-Δτ0)
                Gτ0_τmΔτ = @view Gτ0[:,:,l] # G(τ-Δτ,0)
                Gτ0_τ = @view Gτ0[:,:,l+1] # G(τ,0)
                mul!(Gτ0_τ, B_l, Gτ0_τmΔτ, M = ldr_ws.M)
                # propagate to later imaginary time G(τ,τ) = B[l]⋅G(τ-Δτ,τ-Δτ)⋅B⁻¹[l]
                Gττ_τmΔτ = @view Gττ[:,:,l]
                Gττ_τ = @view Gττ[:,:,l+1]
                mul!(Gττ_τ, B_l, Gττ_τmΔτ, M = ldr_ws.M) # B[l]⋅G(τ-Δτ,τ-Δτ)
                rdiv!(Gττ_τ, B_l, M = ldr_ws.M) # B[l]⋅G(τ-Δτ,τ-Δτ)⋅B⁻¹[l]
            end
            # update LDR factorization for B(τ=Δτ⋅n⋅nₛ,0)
            fgc.l = min(n*nₛ,Lτ)
            update_factorizations!(fgc)
            # calculate G(τ=Δτ⋅n⋅nₛ,0) and G(τ=Δτ⋅n⋅nₛ,τ=Δτ⋅n⋅nₛ)
            if n < Nₛ
                B_τ0 = F[n]::LDR{T,E} # B(τ,0)
                B_βτ = F[n+1]::LDR{T,E} # B(β,τ)
                # calculate G(τ=Δτ⋅n⋅nₛ,0) using numerically stable procedure
                Gτ0_τ = @view Gτ0[:,:,fgc.l+1] # G(τ,0)
                inv_invUpV!(Gτ0_τ, B_τ0, B_βτ, ldr_ws) # G(0,τ) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
                # calculate G(τ=Δτ⋅n⋅nₛ,τ=Δτ⋅n⋅nₛ)
                Gττ_τ = @view Gττ[:,:,fgc.l+1]
                inv_IpUV!(Gττ_τ, B_τ0, B_βτ, ldr_ws) # G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
            end
        end
        # reset forward to true (next iteration over imaginary time will be in the reverse direction)
        fgc.forward = false
    # if iterating from l=Lτ => l=1
    else
        # iterate over stabilization intervals
        for n in Nₛ:-1:1
            # calculate LDR factorization for B(β, τ=Δτ⋅n⋅nₛ-Δτ,)
            fgc.l = (n-1)*nₛ+1
            update_factorizations!(fgc)
            # calculate G(τ=Δτ⋅(n-1)⋅nₛ, 0) and G(τ=Δτ⋅(n-1)⋅nₛ,τ=Δτ⋅(n-1)⋅nₛ) using numerically stable procedure
            if n > 1
                B_βτ = F[n]::LDR{T,E} # B(β, τ=Δτ⋅(n-1)⋅nₛ)
                B_τ0 = F[n-1]::LDR{T,E} # B(τ=Δτ⋅(n-1)⋅nₛ, 0,)
                # calculate G(τ=Δτ⋅(n-1)⋅nₛ, 0) using numerically stable procedure
                Gτ0_τ = @view Gτ0[:,:,fgc.l] # G(τ=Δτ⋅(n-1)⋅nₛ, 0,)
                inv_invUpV!(Gτ0_τ, B_τ0, B_βτ, ldr_ws) # G(0,τ) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
                # calculate G(τ=Δτ⋅(n-1)⋅nₛ,τ=Δτ⋅(n-1)⋅nₛ) using numerically stable procedure
                Gττ_τ = @view Gττ[:,:,fgc.l]
                inv_IpUV!(Gττ_τ, B_τ0, B_βτ, ldr_ws) # G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
            end
            # iterate over imaginary time slice in stabilization interval
            for l in (n-1)*nₛ+1:min(n*nₛ,Lτ)-1
                # get B[l]
                B_l = B[l]::P
                # propagate to later imaginary time G(τ,0) = B[l]⋅G(τ-Δτ,0)
                Gτ0_τmΔτ = @view Gτ0[:,:,l] # G(τ-Δτ,0)
                Gτ0_τ = @view Gτ0[:,:,l+1] # G(τ,0)
                mul!(Gτ0_τ, B_l, Gτ0_τmΔτ, M = ldr_ws.M)
                # propagate to later imaginary time G(τ,τ) = B[l]⋅G(τ-Δτ,τ-Δτ)⋅B⁻¹[l]
                Gττ_τmΔτ = @view Gττ[:,:,l]
                Gττ_τ = @view Gττ[:,:,l+1]
                mul!(Gττ_τ, B_l, Gττ_τmΔτ, M = ldr_ws.M) # B[l]⋅G(τ-Δτ,τ-Δτ)
                rdiv!(Gττ_τ, B_l, M = ldr_ws.M) # B[l]⋅G(τ-Δτ,τ-Δτ)⋅B⁻¹[l]
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
G(\tau+\Delta\tau,\tau+\Delta\tau) = B_{l+1} \cdot G(\tau,\tau) \cdot B_{l+1}^{-1}
```
is used, and if iterating over imaginary time in the reverse direction (`fgc.forward = false`)
the relationship
```math
G(\tau-\Delta\tau,\tau-\Delta\tau)= B_{l}^{-1} \cdot G(\tau,\tau) \cdot B_{l}
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
        # G(l,l) = B[l]⋅G(l-1,l-1)⋅B⁻¹[l]
        B_l = B[l]::P
        lmul!(B_l, G, M = M)
        rdiv!(G, B_l, M = M)
    # if iterating from l=Lτ => l=1.
    # Note: if `fgc.l` corresponds to the "end" of a stabilization interval (l′=nₛ or l=Lτ), then
    # when the equal-time Green's function was previously re-computed at `fgc.l+1` in the stabilize_equatime_greens!()
    # routine, it already calculated G(l,l), so we don't need to propagate to G(l,l) here.
    elseif !(l′ == nₛ || l==Lτ)
        # G(l,l) = B⁻¹[l+1]⋅G(l+1,l+1)⋅B[l+1]
        B_lp1 = B[l+1]::P
        ldiv!(B_lp1, G, M = M)
        rmul!(G, B_lp1, M = M)
    end

    return nothing
end


@doc raw"""
    stabilize_equaltime_greens!(G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
                                fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P};
                                update_B̄::Bool=true)::Tuple{E,T,E,E} where {T, E, P<:AbstractPropagator{T,E}}

Stabilize the equal-time Green's function as iterating through imaginary time ``\tau = \Delta\tau \cdot l.``
For a given imaginary time slice `fgc.l`, this routine should be called *after* all changes to the ``B_l``
propagator have been made.
When iterating through imaginary time in the forwards direction (`fgc.forward = true`), this function
re-computes
```math
G(\tau,\tau) = [I + B(\tau,0)B(\beta,\tau)]^{-1}
```
when at imaginary time slice `fgc.l` every `fgc.nₛ` imaginary time slice.
When iterating through imaginary time in the reverse direction (`fgc.forward = false`), this function
instead re-computes
```math
G(\tau-\Delta\tau,\tau-\Delta\tau) = [I + B(\tau-\Delta\tau,0)B(\beta,\tau-\Delta\tau)]^{-1}
```
for `fgc.l`.

This method returns four values.
The first two values returned are ``\log(\vert \det G(\tau,\tau) \vert)`` and ``\textrm{sign}(\det G(\tau,\tau))``.
The latter two are the maximum error in a Green's function corrected by numerical stabilization ``\vert \delta G \vert``,
and the error in the phase of the determinant corrected by numerical stabilization ``\delta\theta,``
relative to naive propagation of the Green's function matrix in imaginary time occuring instead.
If no stabilization was performed, than ``\vert \delta G \vert = 0`` and ``\delta \theta = 0.``

This method also computes the required ``\bar{B}_n`` matrices, and the LDR matrix factorizations
representing ``B(\tau, 0)`` or ``B(\beta, \tau-\Delta\tau)`` when iterating through imaginary
time ``\tau = \Delta\tau \cdot l`` in the forward and reverse directions respectively.
If `update_B̄ = true`, then the ``\bar{B}_n`` matrices are re-calculated as needed, but if
`update_B̄ = false,` then they are left unchanged.
"""
function stabilize_equaltime_greens!(G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
                                     fgc::FermionGreensCalculator{T,E}, B::AbstractVector{P};
                                     update_B̄::Bool=true)::Tuple{E,T,E,E} where {T, E, P<:AbstractPropagator{T,E}}

    (; l, Lτ, forward, nₛ, Nₛ, G′) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # initialize error corrected by stabilization in green's function
    # and error in phase of determinant
    δG = zero(E)
    δθ = zero(E)

    # record initial sign of determinant
    sgndetG′ = sgndetG

    # update B(τ,0) if iterating forward or B(β,τ-Δτ) if iterating backwards
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
            # G(β,β) = [I + B(β,0)]⁻¹
            B_β0 = F[Nₛ]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(G, B_β0, ldr_ws)

        # if at boundary of stablization interval calculate G(τ,τ)
        elseif l′ == nₛ && Nₛ > 1
            # calculate G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
            B_βτ = F[n+1]::LDR{T,E}
            B_τ0 = F[n]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(G, B_τ0, B_βτ, ldr_ws)
        end
        # perform naive propagation G′(τ,τ) = B[l]⋅G(τ-Δτ,τ-Δτ)⋅B⁻¹[l]
        B_l = B[l]::P
        mul!(G′, B_l, G, M = ldr_ws.M) # B[l]⋅G(τ-Δτ,τ-Δτ)
        rdiv!(G′, B_l, M = ldr_ws.M) # B[l]⋅G(τ-Δτ,τ-Δτ)⋅B⁻¹[l]
        # calculate the error corrected by stabilization
        ΔG = G′
        @. ΔG = abs(G′-G)
        δG = maximum(real, ΔG)
        δθ = angle(sgndetG′/sgndetG)
    # if iterating from l=Lτ => l=1
    else
        # if at first time slice calculate G(0,0) = G(β,β)
        if l == 1
            # G(β,β) = [I + B(β,0)]⁻¹
            B_β0 = F[1]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(G, B_β0, ldr_ws)

        # if at boundary of stabilization interval calculate G(τ-Δτ,τ-Δτ)
        elseif l′ == 1 && Nₛ > 1
            # calucate G(τ-Δτ,τ-Δτ) = [I + B(τ-Δτ,0)⋅B(β,τ-Δτ)]⁻¹
            B_β_τmΔτ = F[n]::LDR{T,E}
            B_τmΔτ_0 = F[n-1]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(G, B_τmΔτ_0, B_β_τmΔτ, ldr_ws)
        end
        # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
        B_l = B[l]::P
        mul!(G′, G, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
        ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
        # calculate the error corrected by stabilization
        ΔG = G′
        @. ΔG = abs(G′-G)
        δG = maximum(real, ΔG)
        δθ = angle(sgndetG′/sgndetG)
    end
    
    return (logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    local_update_det_ratio(G::AbstractMatrix{T}, B::AbstractPropagator{T,E},
                           V′::T, i::Int, Δτ::E)::Tuple{T,T} where {T,E}

Calculate the determinant ratio ``R_{l,i}`` associated with a local update to the equal-time
Green's function ``G(\tau,\tau).`` Also returns ``\Delta_{l,i},`` which is defined below.

# Arguments

- `G::AbstractMatrix{T}`: Equal-time Green's function matrix ``G(\tau,\tau).``
- `B::AbstractPropagator{T,E}`: Represents the propagator matrix ``B_l,`` where ``\tau = \Delta\tau \cdot l.``
- `V′::T`: The new value for the ``V^{\prime}_{l,i,i}`` matrix element in the diagonal on-site energy matrix ``V_l.``
- `i::Int`: Diagonal matrix element index in ``V_l`` being updated.
- `Δτ::E`: Discretization in imaginary time ``\Delta\tau.``

# Algorithm

The propagator matrix ``B_l`` above is given by
```math
B_l = \Lambda_l \cdot \Gamma_l(\Delta\tau),
```
where, assuming the we are working in the orbital basis, ``\Gamma_l(\Delta\tau) = e^{-\Delta\tau K_l}``
represents the exponentiated hopping matrix ``K_l``, and ``\Lambda_l = e^{-\Delta\tau V_l}`` represents
the exponentiated diagonal on-site energy matrix ``V_l.``

Given a proposed update to the ``(i,i)`` matrix element of the diagonal on-site energy matrix ``V_l``,
(``V_{l,i,i} \rightarrow V^\prime_{l,i,i}),`` the corresponding determinant ratio associated
with this proposed udpate is given by
```math
R_{l,i} = \frac{\det G(\tau,\tau)}{\det G^\prime(\tau,\tau)} = 1+\Delta_{i,i}(\tau,i)\left(1-G_{i,i}(\tau,\tau)\right),
```
where
```math
\Delta_{l,i} = \frac{\Lambda^\prime_{l,i,i}}{\Lambda_{l,i,i}} - 1 = e^{-\Delta\tau (V^\prime_{l,i,i} - V_{l,i,i})} - 1.
```
This routine returns the scalar quantities ``R_{l,i}`` and ``\Delta_{l,i}.``

Note that if the propagator matrix is instead represented using the symmetric form
```math
B_l = \Gamma_l(\Delta\tau/2) \cdot \Lambda_l \cdot \Gamma^\dagger_l(\Delta\tau/2),
```
then the matrix `G` needs to instead represent the transformed equal-time Green's function matrix
```math
\tilde{G}(\tau,\tau) = \Gamma_l^{-1}(\Delta\tau/2) \cdot G(\tau,\tau) \cdot \Gamma_l(\Delta\tau/2).
```
"""
function local_update_det_ratio(G::AbstractMatrix{T}, B::AbstractPropagator{T,E},
                                V′::T, i::Int, Δτ::E)::Tuple{T,T} where {T,E}

    Λ = B.expmΔτV
    Δ = exp(-Δτ*V′)/Λ[i] - 1
    R = 1 + Δ*(1 - G[i,i])

    return (R, Δ)
end


@doc raw"""
    local_update_greens!(G::AbstractMatrix{T}, logdetG::E, sgndetG::T, R::T, Δ::T, i::Int,
                         u::AbstractVector{T}, v::AbstractVector{T})::Tuple{E,T} where {T, E<:AbstractFloat}

Update the equal-time Green's function matrix `G` resulting from a local update in-place.

# Arguments

- `G::AbstractMatrix{T}`: Equal-time Green's function matrix ``G(\tau,\tau)`` that will be updated in-place.
- `logdetG::E`: The log of the absolute value of the initial Green's function matrix, ``\log( \vert \det G(\tau,\tau) \vert ).``
- `sgndetG::T`: The sign/phase of the determinant of the initial Green's function matrix, ``\textrm{sign}( \det G(\tau,\tau) ).``
- `R::T`: The determinant ratio ``R_{l,i} = \frac{\det G(\tau,\tau)}{\det G^\prime(\tau,\tau)}.``
- `Δ::T`: Change in the exponentiated on-site energy matrix, ``\Delta_{l,i} = e^{-\Delta\tau (V^\prime_{l,(i,i)} - V_{l,(i,i)})} - 1.``
- `i::Int`: Matrix element of diagonal on-site energy matrix ``V_l`` that is being updated.
- `u::AbstractVector{T}`: Vector of length `size(G,1)` that is used to avoid dynamic memory allocations.
- `v::AbstractVector{T}`: Vector of length `size(G,2)` that is used to avoid dynamic memory allocations.

# Algorithm

The equal-time Green's function matrix is updated using the relationship
```math
G_{j,k}^{\prime}\left(\tau,\tau\right)=G_{j,k}\left(\tau,\tau\right)-\frac{1}{R_{l,i}}G_{j,i}\left(\tau,\tau\right)\Delta_{l,i}\left(\delta_{i,k}-G_{i,k}\left(\tau,\tau\right)\right).
```
This method also returns ``\log( \vert \det G^\prime(\tau,\tau) \vert )`` and ``\textrm{sign}( \det G^\prime(\tau,\tau) ).``

An important note is that if the propagator matrices are represented in a symmetric form, then `G′` and `G` need to correspond
to the transformed eqaul-time Green's function matrices ``\tilde{G}^\prime(\tau,\tau)`` and ``\tilde{G}(\tau,\tau).``
Refer to the [`local_update_det_ratio`](@ref) docstring for more information.
"""
function local_update_greens!(G::AbstractMatrix{T}, logdetG::E, sgndetG::T, R::T, Δ::T, i::Int,
                              u::AbstractVector{T}, v::AbstractVector{T})::Tuple{E,T} where {T, E<:AbstractFloat}

    # u = G[:,i] <== column vector
    G0i = @view G[:,i]
    copyto!(u, G0i)

    # v = G[i,:] - I[i,:] <== row vector
    Gi0 = @view G[i,:]
    copyto!(v, Gi0) # v = G[i,:]
    v[i] = v[i] - 1 # v = G[i,:] - I[i,:]

    # G′ = G + (Δ/R)⋅[u⨂v] = G + (Δ/R)⋅G[:,i]⨂(G[i,:] - I[i,:])
    # where ⨂ denotes an outer product
    BLAS.ger!(Δ/R, u, v, G)

    # R = det(M′)/det(M) = det(G)/det(G′)
    # ==> log(|R|) = log(|det(M′)|) - log(|det(M)|) = log(|det(G)|) - log(|det(G′)|)
    # ==> log(|det(G′)|) = log(|det(G)|) - log(|R|)
    logdetG′ = logdetG - log(abs(R))
    sgndetG′ = sign(R) * sgndetG

    return (logdetG′, sgndetG′)
end