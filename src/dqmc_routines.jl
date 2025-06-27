@doc raw"""
    calculate_equaltime_greens!(
        G::AbstractMatrix{T},
        fgc::FermionGreensCalculator{T,E}
    )::Tuple{E,T} where {T<:Continuous, E<:AbstractFloat}

Calculate the equal-time Greens function ``G(0,0) = G(\beta,\beta) = [I + B(\beta,0)]^{-1}`` using a numerically stable procedure.
This method also returns ``\log(\vert \det G \vert)`` and ``\textrm{sign}(\det G).``
Note that this routine requires `fgc.l == 1` or `fgc.l == fgc.Lτ`.
"""
function calculate_equaltime_greens!(
    G::AbstractMatrix{T},
    fgc::FermionGreensCalculator{T,E}
)::Tuple{E,T} where {T<:Continuous, E<:AbstractFloat}

    (; forward, l, Lτ, N_stab, ldr_ws) = fgc
    @assert l == 1 || l == Lτ

    # get B(0,β)
    if forward
        B_β0 = fgc.F[1]::LDR{T,E}
    else
        B_β0 = fgc.F[N_stab]::LDR{T,E}
    end

    # calculate G(0,0) = [I + B(β,0)]⁻¹
    logdetG, sgndetG = inv_IpA!(G, B_β0, ldr_ws)

    return logdetG, sgndetG
end

@doc raw"""
    calculate_equaltime_greens!(
        G::AbstractMatrix{H},
        fgc::FermionGreensCalculator{H,R},
        B::AbstractVector{P}
    )::Tuple{R,H} where {H<:Continuous, R<:AbstractFloat, P<:AbstractPropagator}

Calculate the equal-time Greens function ``G(0,0) = G(\beta,\beta) = [I + B(\beta,0)]^{-1}`` using a numerically stable procedure.
Also re-calculate the ``\bar{B}_n`` matrices and the LDR matrix factorizations representing
either ``B(\tau,0)`` or ``B(\beta,\tau)`` stored in `fgc.F`. This routine is useful for implementing global updates where every
propagator matrix ``B_l`` has been modified, and the equal-time Green's function needs to be re-calculated from scratch.
This method also returns ``\log(\vert \det G \vert)`` and ``\textrm{sign}(\det G).``
Note that this routine requires `fgc.l == 1` or `fgc.l == fgc.Lτ`.
"""
function calculate_equaltime_greens!(
    G::AbstractMatrix{H},
    fgc::FermionGreensCalculator{H,R},
    B::AbstractVector{P}
)::Tuple{R,H} where {H<:Continuous, R<:AbstractFloat, P<:AbstractPropagator}

    # iterate over imaginary time
    for l in fgc
        # re-calculate B_bar matrices and matrix factorizations B(τ,0) or B(β,τ) as needed
        update_factorizations!(fgc, B)
    end
    
    # calculate equal-time Greens funciton matrix G(0,0) = G(β,β)
    logdetG, sgndetG = calculate_equaltime_greens!(G, fgc)
    
    return logdetG, sgndetG
end


@doc raw"""
    propagate_equaltime_greens!(
        G::AbstractMatrix{T},
        fgc::FermionGreensCalculator{T,E},
        B::AbstractVector{P}
    ) where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

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
function propagate_equaltime_greens!(
    G::AbstractMatrix{T},
    fgc::FermionGreensCalculator{T,E},
    B::AbstractVector{P}
) where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

    (; n_stab, l, Lτ, forward) = fgc
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
    # Note: if `fgc.l` corresponds to the "end" of a stabilization interval (l′=n_stab or l=Lτ), then
    # when the equal-time Green's function was previously re-computed at `fgc.l+1` in the stabilize_equatime_greens!()
    # routine, it already calculated G(l,l), so we don't need to propagate to G(l,l) here.
    elseif !(l′ == n_stab || l==Lτ)
        # G(l,l) = B⁻¹[l+1]⋅G(l+1,l+1)⋅B[l+1]
        B_lp1 = B[l+1]::P
        ldiv!(B_lp1, G, M = M)
        rmul!(G, B_lp1, M = M)
    end

    return nothing
end


@doc raw"""
    stabilize_equaltime_greens!(
        G::AbstractMatrix{T},
        logdetG::E, sgndetG::T,
        fgc::FermionGreensCalculator{T,E},
        B::AbstractVector{P};
        # KEYWORD ARGUMENTS
        update_B̄::Bool=true
    )::Tuple{E,T,E,E} where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

Stabilize the equal-time Green's function as iterating through imaginary time ``\tau = \Delta\tau \cdot l.``
For a given imaginary time slice `fgc.l`, this routine should be called *after* all changes to the ``B_l``
propagator have been made.
When iterating through imaginary time in the forwards direction (`fgc.forward = true`), this function
re-computes
```math
G(\tau,\tau) = [I + B(\tau,0)B(\beta,\tau)]^{-1}
```
when at imaginary time slice `fgc.l` every `fgc.n_stab` imaginary time slice.
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

This method also computes the LDR matrix factorizations
representing ``B(\tau, 0)`` or ``B(\beta, \tau-\Delta\tau)`` when iterating through imaginary
time ``\tau = \Delta\tau \cdot l`` in the forward and reverse directions respectively.
If `update_B̄ = true`, then the ``\bar{B}_n`` matrices are re-calculated as needed, but if
`update_B̄ = false,` then they are left unchanged.
"""
function stabilize_equaltime_greens!(
    G::AbstractMatrix{T},
    logdetG::E, sgndetG::T,
    fgc::FermionGreensCalculator{T,E},
    B::AbstractVector{P};
    # KEYWORD ARGUMENTS
    update_B̄::Bool=true
)::Tuple{E,T,E,E} where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

    (; l, Lτ, forward, n_stab, N_stab, G′) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # initialize error corrected by stabilization in green's function
    # and error in phase of determinant
    δG = zero(E)
    δθ = zero(E)

    # boolean specifying whether or not stabilization was performed
    stabilized = false

    # record initial sign of determinant
    sgndetG′ = sgndetG

    # update B(τ,0) if iterating forward or B(β,τ-Δτ) if iterating backwards
    if update_B̄
        # also re-calculate B_bar matrices
        update_factorizations!(fgc, B)
    else
        # do not update B_bar matrices
        update_factorizations!(fgc)
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # if at last time slice calculate G(β,β)
        if l == Lτ
            # record initial G′(β,β) = G(β,β) before stabilization
            copyto!(G′, G)
            # G(β,β) = [I + B(β,0)]⁻¹
            B_β0 = F[N_stab]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(G, B_β0, ldr_ws)
            # record that stabilization was performed
            stabilized = true
        # if at boundary of stablization interval calculate G(τ,τ)
        elseif l′ == n_stab && N_stab > 1
            # record initial G′(τ,τ) = G(τ,τ) before stabilization
            copyto!(G′, G)
            # calculate G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
            B_βτ = F[n+1]::LDR{T,E}
            B_τ0 = F[n]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(G, B_τ0, B_βτ, ldr_ws)
            # record that stabilization was performed
            stabilized = true
        end
    # if iterating from l=Lτ => l=1
    else
        # if at first time slice calculate G(0,0) = G(β,β)
        if l == 1
            # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
            B_l = B[l]::P
            mul!(G′, G, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
            ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
            # G(0,0) = [I + B(β,0)]⁻¹
            B_β0 = F[1]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(G, B_β0, ldr_ws)
            # record that stabilization was performed
            stabilized = true
        # if at boundary of stabilization interval calculate G(τ-Δτ,τ-Δτ)
        elseif l′ == 1 && N_stab > 1
            # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
            B_l = B[l]::P
            mul!(G′, G, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
            ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
            # calucate G(τ-Δτ,τ-Δτ) = [I + B(τ-Δτ,0)⋅B(β,τ-Δτ)]⁻¹
            B_β_τmΔτ = F[n]::LDR{T,E}
            B_τmΔτ_0 = F[n-1]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(G, B_τmΔτ_0, B_β_τmΔτ, ldr_ws)
            # record that stabilization was performed
            stabilized = true
        end
    end

    # calculate error corrected by stabilization
    if stabilized
        ΔG = G′
        @. ΔG = abs(G′-G)
        δG = maximum(real, ΔG)
        δθ = angle(sgndetG′/sgndetG)
    end
    
    return (logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    initialize_unequaltime_greens!(
        Gτ0::AbstractMatrix{T},
        G0τ::AbstractMatrix{T},
        Gττ::AbstractMatrix{T},
        G00::AbstractMatrix{T}
    ) where {T<:Number}

Initialize the Green's function matrices ``G(\tau,0),`` ``G(0,\tau),`` and ``G(\tau,\tau)`` for ``\tau = 0``
based on the matrix ``G(0,0).``
"""
function initialize_unequaltime_greens!(
    Gτ0::AbstractMatrix{T},
    G0τ::AbstractMatrix{T},
    Gττ::AbstractMatrix{T},
    G00::AbstractMatrix{T}
) where {T<:Number}

    # G(τ=0,τ=0) = G(0,0)
    copyto!(Gττ, G00)

    # G(τ=0,0) = G(0,0)
    copyto!(Gτ0, G00)

    # G(0,τ=0) = -(I-G(0,0))
    copyto!(G0τ, I)
    @. G0τ = -(G0τ - G00)

    return nothing
end


@doc raw"""
    propagate_unequaltime_greens!(
        Gτ0::AbstractMatrix{T},
        G0τ::AbstractMatrix{T},
        Gττ::AbstractMatrix{T},
        fgc::FermionGreensCalculator{T,E},
        B::AbstractVector{P}
    ) where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

Propagate the Green's function matrices ``G(\tau,0)``, ``G(0,\tau)`` and ``G(\tau,\tau)``
from the previous imaginary time slice to the current
imaginary time slice `fgc.l`. If iterating over imaginary time in the forward direction (`fgc.forward = true`)
the relationships
```math
\begin{align}
G(\tau,0)    = & B_{l} \cdot G(\tau-\Delta\tau, 0) \\
G(0,\tau)    = & G(0, \tau-\Delta\tau) \cdot B^{-1}_{l} \\
G(\tau,\tau) = & B_{l} \cdot G(\tau-\Delta\tau, \tau-\Delta\tau) \cdot B_{l}^{-1}
\end{align}
```
are used, and if iterating over imaginary time in the reverse direction (`fgc.forward = false`)
the relationships
```math
\begin{align}
G(\tau,0)    = & B_{l+1}^{-1} \cdot G(\tau+\Delta\tau, 0) \\
G(0,\tau)    = & G(0, \tau + \Delta\tau) \cdot B_{l+1} \\
G(\tau,\tau) = & B_{l+1}^{-1} \cdot G(\tau+\Delta\tau, \tau+\Delta\tau) \cdot B_{l+1}
\end{align}
```
are used instead, where the ``B_l`` propagator is given by `B[l]`.
"""
function propagate_unequaltime_greens!(
    Gτ0::AbstractMatrix{T},
    G0τ::AbstractMatrix{T},
    Gττ::AbstractMatrix{T},
    fgc::FermionGreensCalculator{T,E},
    B::AbstractVector{P}
) where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

    (; n_stab, l, Lτ, forward) = fgc
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}
    M = ldr_ws.M::Matrix{T}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # if iterating from l=1 => l=Lτ
    if forward

        # G(l,l) = B[l]⋅G(l-1,l-1)⋅B⁻¹[l]
        B_l = B[l]::P
        lmul!(B_l, Gττ, M = M)
        rdiv!(Gττ, B_l, M = M)

        # G(l,0) = B[l]⋅G(l-1,0)
        lmul!(B_l, Gτ0, M = M)

        # G(0,l) = G(0,l-1)⋅B⁻¹[l]
        rdiv!(G0τ, B_l, M = M)

    # if beginning iteration from l=Lτ => l=1, initialize uneqaul-time green's function matrices
    elseif l == Lτ

        # G(τ=β,0) = (I - G(0,0))
        copyto!(Gτ0, I)
        @. Gτ0 = (Gτ0 - Gττ)

        # G(0,τ=β) = -G(0,0)
        @. G0τ = -Gττ

    # if iterating from l=Lτ => l=1.
    # Note: if `fgc.l` corresponds to the "end" of a stabilization interval (l′=n_stab), then
    # when the equal-time Green's function was previously re-computed at `fgc.l+1` in the stabilize_equatime_greens!()
    # routine, it already calculated G(l,l), so we don't need to propagate to G(l,l) here.
    elseif !(l′ == n_stab)

        # G(l,l) = B⁻¹[l+1]⋅G(l+1,l+1)⋅B[l+1]
        B_lp1 = B[l+1]::P
        ldiv!(B_lp1, Gττ, M = M)
        rmul!(Gττ, B_lp1, M = M)

        # G(l, 0) = B⁻¹[l+1]⋅G(l+1,0)
        ldiv!(B_lp1, Gτ0, M = M)

        # G(0, l) = G(0,l+1)⋅B[l+1]
        rmul!(G0τ, B_lp1, M = M)
    end

    return nothing
end


@doc raw"""
    stabilize_unequaltime_greens!(
        Gτ0::AbstractMatrix{T},
        G0τ::AbstractMatrix{T},
        Gττ::AbstractMatrix{T},
        logdetG::E, sgndetG::T,
        fgc::FermionGreensCalculator{T,E},
        B::AbstractVector{P};
        # KEYWORD ARGUMENTS
        update_B̄::Bool=true
    )::Tuple{E,T,E,E} where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

Stabilize the Green's function matrice ``G(\tau,0)``, ``G(0,\tau)`` and ``G(\tau,\tau)``
as iterating through imaginary time ``\tau = \Delta\tau \cdot l.``
For a given imaginary time slice `fgc.l`, this routine should be called *after* all changes to the ``B_l``
propagator have been made.
When iterating through imaginary time in the forwards direction (`fgc.forward = true`), this function
re-computes
```math
\begin{align}
G(\tau,0)    = & [B^{-1}(\tau,0) + B(\beta,\tau)]^{-1} \\
G(0, \tau)   = & [B^{-1}(\beta,\tau) + B(\tau,0)]^{-1} \\
G(\tau,\tau) = & [I + B(\tau,0)B(\beta,\tau)]^{-1}
\end{align}
```
when at imaginary time slice `fgc.l` every `fgc.n_stab` imaginary time slice.
When iterating through imaginary time in the reverse direction (`fgc.forward = false`), this function
instead re-computes
```math
\begin{align*}
G(\tau-\Delta\tau,0)               = & [B^{-1}(\tau-\Delta\tau,0) + B(\beta,\tau-\Delta\tau)]^{-1} \\
G(0,\tau-\Delta\tau)               = & [B^{-1}(\beta,\tau-\Delta\tau) + B(\tau-\Delta\tau,0)]^{-1} \\
G(\tau-\Delta\tau,\tau-\Delta\tau) = & [I + B(\tau-\Delta\tau,0)B(\beta,\tau-\Delta\tau)]^{-1}
\end{align*}
```
for `fgc.l`.

This method returns four values.
The first two values returned are ``\log(\vert \det G(\tau,\tau) \vert)`` and ``\textrm{sign}(\det G(\tau,\tau))``.
The latter two are the maximum error in a Green's function corrected by numerical stabilization ``\vert \delta G \vert``,
and the error in the phase of the determinant corrected by numerical stabilization ``\delta\theta,``
relative to naive propagation of the Green's function matrix in imaginary time occuring instead.
If no stabilization was performed, than ``\vert \delta G \vert = 0`` and ``\delta \theta = 0.``

This method also computes the LDR matrix factorizations
representing ``B(\tau, 0)`` or ``B(\beta, \tau-\Delta\tau)`` when iterating through imaginary
time ``\tau = \Delta\tau \cdot l`` in the forward and reverse directions respectively.
If `update_B̄ = true`, then the ``\bar{B}_n`` matrices are re-calculated as needed, but if
`update_B̄ = false,` then they are left unchanged.
"""
function stabilize_unequaltime_greens!(
    Gτ0::AbstractMatrix{T},
    G0τ::AbstractMatrix{T},
    Gττ::AbstractMatrix{T},
    logdetG::E, sgndetG::T,
    fgc::FermionGreensCalculator{T,E},
    B::AbstractVector{P};
    # KEYWORD ARGUMENTS
    update_B̄::Bool=true
)::Tuple{E,T,E,E} where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

    (; l, Lτ, forward, n_stab, N_stab, G′) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # initialize error corrected by stabilization in green's function
    # and error in phase of determinant
    δG = zero(E)
    δθ = zero(E)

    # boolean specifying whether or not stabilization was performed
    stabilized = false

    # record initial sign of determinant
    sgndetG′ = sgndetG

    # update B(τ,0) if iterating forward or B(β,τ-Δτ) if iterating backwards
    if update_B̄
        # also re-calculate B_bar matrices
        update_factorizations!(fgc, B)
    else
        # do not update B_bar matrices
        update_factorizations!(fgc)
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # if at last time slice calculate G(β,β)
        if l == Lτ
            # record initial G′(β,β) = G(β,β) before stabilization
            copyto!(G′, Gττ)
            # G(0,0) = G(β,β) = [I + B(β,0)]⁻¹
            B_β0 = F[N_stab]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(Gττ, B_β0, ldr_ws)
            # G(β, 0) = I - G(0,0)
            copyto!(Gτ0, I)
            @. Gτ0 = Gτ0 - Gττ
            # G(0, β) = -G(0,0)
            @. G0τ = -Gττ
            # record that stabilization was performed
            stabilized = true
        # if at boundary of stablization interval calculate G(τ,τ)
        elseif l′ == n_stab && N_stab > 1
            # record initial G′(τ,τ) = G(τ,τ) before stabilization
            copyto!(G′, Gττ)
            # calculate G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
            B_βτ = F[n+1]::LDR{T,E}
            B_τ0 = F[n]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(Gττ, B_τ0, B_βτ, ldr_ws)
            # calculate G(τ,0) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
            inv_invUpV!(Gτ0, B_τ0, B_βτ, ldr_ws)
            # calculate G(0,τ) = -[B⁻¹(β,τ) + B(τ,0)]⁻¹
            inv_invUpV!(G0τ, B_βτ, B_τ0, ldr_ws)
            @. G0τ = -G0τ
            # record that stabilization was performed
            stabilized = true
        end
    # if iterating from l=Lτ => l=1
    else
        # if at first time slice calculate G(0,0) = G(β,β)
        if l == 1
            # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
            B_l = B[l]::P
            mul!(G′, Gττ, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
            ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
            # G(0,0) = [I + B(β,0)]⁻¹
            B_β0 = F[1]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(Gττ, B_β0, ldr_ws)
            # G(β, 0) = I - G(0,0)
            copyto!(Gτ0, I)
            @. Gτ0 = Gτ0 - Gττ
            # G(0, β) = -G(0,0)
            @. G0τ = -Gττ
            # record that stabilization was performed
            stabilized = true
        # if at boundary of stabilization interval calculate G(τ-Δτ,τ-Δτ)
        elseif l′ == 1 && N_stab > 1
            # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
            B_l = B[l]::P
            mul!(G′, Gττ, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
            ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
            # calucate G(τ-Δτ,τ-Δτ) = [I + B(τ-Δτ,0)⋅B(β,τ-Δτ)]⁻¹
            B_β_τmΔτ = F[n]::LDR{T,E}
            B_τmΔτ_0 = F[n-1]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(Gττ, B_τmΔτ_0, B_β_τmΔτ, ldr_ws)
            # calculate G(τ-Δτ,0) = [B⁻¹(τ-Δτ,0) + B(β,τ-Δτ)]⁻¹
            inv_invUpV!(Gτ0, B_τmΔτ_0, B_β_τmΔτ, ldr_ws)
            # calculate G(0,τ-Δτ) = -[B⁻¹(β,τ-Δτ) + B(τ-Δτ,0)]⁻¹
            inv_invUpV!(G0τ, B_β_τmΔτ, B_τmΔτ_0, ldr_ws)
            @. G0τ = -G0τ
            # record that stabilization was performed
            stabilized = true
        end
    end

    # calculate error corrected by stabilization
    if stabilized
        ΔG = G′
        @. ΔG = abs(G′-Gττ)
        δG = maximum(real, ΔG)
        δθ = angle(sgndetG′/sgndetG)
    end
    
    return (logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    stabilize_unequaltime_greens!(
        Gτ0::AbstractMatrix{T},
        G0τ::AbstractMatrix{T},
        Gττ::AbstractMatrix{T},
        G00::AbstractMatrix{T},
        logdetG::E, sgndetG::T,
        fgc::FermionGreensCalculator{T,E},
        B::AbstractVector{P};
        # KEYWORD ARGUMENTS
        update_B̄::Bool=true
    )::Tuple{E,T,E,E} where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

Stabilize the Green's function matrice ``G(\tau,0)``, ``G(0,\tau),`` ``G(\tau,\tau)`` and ``G(0,0)``
while iterating through imaginary time ``\tau = \Delta\tau \cdot l.``
For a given imaginary time slice `fgc.l`, this routine should be called *after* all changes to the ``B_l``
propagator have been made.
When iterating through imaginary time in the forwards direction (`fgc.forward = true`), this function
re-computes
```math
\begin{align}
G(\tau,0)    = & [B^{-1}(\tau,0) + B(\beta,\tau)]^{-1} \\
G(0, \tau)   = & [B^{-1}(\beta,\tau) + B(\tau,0)]^{-1} \\
G(\tau,\tau) = & [I + B(\tau,0)B(\beta,\tau)]^{-1} \\
G(0, 0)      = & [I + B(\beta,\tau)B(\tau,0)]^{-1} \\
\end{align}
```
when at imaginary time slice `fgc.l` every `fgc.n_stab` imaginary time slice.
When iterating through imaginary time in the reverse direction (`fgc.forward = false`), this function
instead re-computes
```math
\begin{align*}
G(\tau-\Delta\tau,0)               = & [B^{-1}(\tau-\Delta\tau,0) + B(\beta,\tau-\Delta\tau)]^{-1} \\
G(0,\tau-\Delta\tau)               = & [B^{-1}(\beta,\tau-\Delta\tau) + B(\tau-\Delta\tau,0)]^{-1} \\
G(\tau-\Delta\tau,\tau-\Delta\tau) = & [I + B(\tau-\Delta\tau,0)B(\beta,\tau-\Delta\tau)]^{-1} \\
G(0,0)                             = & [I + B(\beta,\tau-\Delta\tau)B(\tau-\Delta\tau,0)]^{-1}
\end{align*}
```
for `fgc.l`.

This method returns four values.
The first two values returned are ``\log(\vert \det G(\tau,\tau) \vert)`` and ``\textrm{sign}(\det G(\tau,\tau))``.
The latter two are the maximum error in a Green's function corrected by numerical stabilization ``\vert \delta G \vert``,
and the error in the phase of the determinant corrected by numerical stabilization ``\delta\theta,``
relative to naive propagation of the Green's function matrix in imaginary time occuring instead.
If no stabilization was performed, than ``\vert \delta G \vert = 0`` and ``\delta \theta = 0.``

This method also computes the LDR matrix factorizations
representing ``B(\tau, 0)`` or ``B(\beta, \tau-\Delta\tau)`` when iterating through imaginary
time ``\tau = \Delta\tau \cdot l`` in the forward and reverse directions respectively.
If `update_B̄ = true`, then the ``\bar{B}_n`` matrices are re-calculated as needed, but if
`update_B̄ = false,` then they are left unchanged.
"""
function stabilize_unequaltime_greens!(
    Gτ0::AbstractMatrix{T},
    G0τ::AbstractMatrix{T},
    Gττ::AbstractMatrix{T},
    G00::AbstractMatrix{T},
    logdetG::E, sgndetG::T,
    fgc::FermionGreensCalculator{T,E},
    B::AbstractVector{P};
    # KEYWORD ARGUMENTS
    update_B̄::Bool=true
)::Tuple{E,T,E,E} where {T<:Continuous, E<:AbstractFloat, P<:AbstractPropagator}

    (; l, Lτ, forward, n_stab, N_stab, G′) = fgc
    F = fgc.F::Vector{LDR{T,E}}
    ldr_ws = fgc.ldr_ws::LDRWorkspace{T,E}

    # get stabilization interval info
    n, l′ = stabilization_interval(fgc)

    # initialize error corrected by stabilization in green's function
    # and error in phase of determinant
    δG = zero(E)
    δθ = zero(E)

    # boolean specifying whether or not stabilization was performed
    stabilized = false

    # record initial sign of determinant
    sgndetG′ = sgndetG

    # update B(τ,0) if iterating forward or B(β,τ-Δτ) if iterating backwards
    if update_B̄
        # also re-calculate B_bar matrices
        update_factorizations!(fgc, B)
    else
        # do not update B_bar matrices
        update_factorizations!(fgc)
    end

    # if iterating from l=1 => l=Lτ
    if forward
        # if at last time slice calculate G(β,β)
        if l == Lτ
            # record initial G′(β,β) = G(β,β) before stabilization
            copyto!(G′, Gττ)
            # G(0,0) = G(β,β) = [I + B(β,0)]⁻¹
            B_β0 = F[N_stab]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(Gττ, B_β0, ldr_ws)
            copyto!(G00, Gττ)
            # G(β, 0) = I - G(0,0)
            copyto!(Gτ0, I)
            @. Gτ0 = Gτ0 - Gττ
            # G(0, β) = -G(0,0)
            @. G0τ = -Gττ
            # record that stabilization was performed
            stabilized = true
        # if at boundary of stablization interval calculate G(τ,τ)
        elseif l′ == n_stab && N_stab > 1
            # record initial G′(τ,τ) = G(τ,τ) before stabilization
            copyto!(G′, Gττ)
            # calculate G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
            B_βτ = F[n+1]::LDR{T,E}
            B_τ0 = F[n]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(Gττ, B_τ0, B_βτ, ldr_ws)
            # calculate G(0,0) = [I + B(β,τ)⋅B(τ,0)]⁻¹
            logdetG00, sgndetG00 = inv_IpUV!(G00, B_βτ, B_τ0, ldr_ws)
            # calculate G(τ,0) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
            inv_invUpV!(Gτ0, B_τ0, B_βτ, ldr_ws)
            # calculate G(0,τ) = -[B⁻¹(β,τ) + B(τ,0)]⁻¹
            inv_invUpV!(G0τ, B_βτ, B_τ0, ldr_ws)
            @. G0τ = -G0τ
            # record that stabilization was performed
            stabilized = true
        end
    # if iterating from l=Lτ => l=1
    else
        # if at first time slice calculate G(0,0) = G(β,β)
        if l == 1
            # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
            B_l = B[l]::P
            mul!(G′, Gττ, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
            ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
            # G(0,0) = [I + B(β,0)]⁻¹
            B_β0 = F[1]::LDR{T,E}
            logdetG, sgndetG = inv_IpA!(Gττ, B_β0, ldr_ws)
            copyto!(G00, Gττ)
            # G(β, 0) = I - G(0,0)
            copyto!(Gτ0, I)
            @. Gτ0 = Gτ0 - Gττ
            # G(0, β) = -G(0,0)
            @. G0τ = -Gττ
            # record that stabilization was performed
            stabilized = true
        # if at boundary of stabilization interval calculate G(τ-Δτ,τ-Δτ)
        elseif l′ == 1 && N_stab > 1
            # perform naive propagation G′(τ-Δτ,τ-Δτ) = B⁻¹[l]⋅G(τ,τ)⋅B[l]
            B_l = B[l]::P
            mul!(G′, Gττ, B_l, M = ldr_ws.M) # G(τ,τ)⋅B[l]
            ldiv!(B_l, G′, M = ldr_ws.M) # B⁻¹[l]⋅G(τ,τ)⋅B[l]
            # calculate G(τ-Δτ,τ-Δτ) = [I + B(τ-Δτ,0)⋅B(β,τ-Δτ)]⁻¹
            B_β_τmΔτ = F[n]::LDR{T,E}
            B_τmΔτ_0 = F[n-1]::LDR{T,E}
            logdetG, sgndetG = inv_IpUV!(Gττ, B_τmΔτ_0, B_β_τmΔτ, ldr_ws)
            # calculate G(0,0) = [I + B(β,τ-Δτ)⋅B(τ-Δτ,0)]⁻¹
            logdetG00, sgndetG00 = inv_IpUV!(G00, B_β_τmΔτ, B_τmΔτ_0, ldr_ws)
            # calculate G(τ-Δτ,0) = [B⁻¹(τ-Δτ,0) + B(β,τ-Δτ)]⁻¹
            inv_invUpV!(Gτ0, B_τmΔτ_0, B_β_τmΔτ, ldr_ws)
            # calculate G(0,τ-Δτ) = -[B⁻¹(β,τ-Δτ) + B(τ-Δτ,0)]⁻¹
            inv_invUpV!(G0τ, B_β_τmΔτ, B_τmΔτ_0, ldr_ws)
            @. G0τ = -G0τ
            # record that stabilization was performed
            stabilized = true
        end
    end

    # calculate error corrected by stabilization
    if stabilized
        ΔG = G′
        @. ΔG = abs(G′-Gττ)
        δG = maximum(real, ΔG)
        δθ = angle(sgndetG′/sgndetG)
    end
    
    return (logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    local_update_det_ratio(
        G::AbstractMatrix,
        B::AbstractPropagator,
        V′::T, i::Int, Δτ::E
    ) where {T<:Number, E<:AbstractFloat}

Calculate the determinant ratio ``R_{l,i}`` associated with a local update to the equal-time
Green's function ``G(\tau,\tau).`` Also returns ``\Delta_{l,i},`` which is defined below.
Specifically, this function returns the tuple ``(R_{l,i}, \Delta_{l,i})``.

# Arguments

- `G::AbstractMatrix`: Equal-time Green's function matrix ``G(\tau,\tau).``
- `B::AbstractPropagator`: Represents the propagator matrix ``B_l,`` where ``\tau = \Delta\tau \cdot l.``
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
function local_update_det_ratio(
    G::AbstractMatrix,
    B::AbstractPropagator,
    V′::T, i::Int, Δτ::E
) where {T<:Number, E<:AbstractFloat}

    Λ = B.expmΔτV
    Δ = exp(-Δτ*V′)/Λ[i] - 1.0
    R = 1.0 + Δ * (1.0 - G[i,i])

    return (R, Δ)
end


@doc raw"""
    local_update_greens!(
        G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
        B::AbstractPropagator,
        R::T, Δ::F, i::Int,
        u::AbstractVector{T}, v::AbstractVector{T}
    )::Tuple{E,T} where {T<:Continuous, F<:Continuous, E<:AbstractFloat}

Update the equal-time Green's function matrix `G` resulting from a local update in-place.

# Arguments

- `G::AbstractMatrix{T}`: Equal-time Green's function matrix ``G(\tau,\tau)`` that will be updated in-place.
- `logdetG::E`: The log of the absolute value of the initial Green's function matrix, ``\log( \vert \det G(\tau,\tau) \vert ).``
- `sgndetG::T`: The sign/phase of the determinant of the initial Green's function matrix, ``\textrm{sign}( \det G(\tau,\tau) ).``
- `B::AbstractPropagator{T,E}`: Propagator that needs to be updated to reflect accepted local update.
- `R::T`: The determinant ratio ``R_{l,i} = \frac{\det G(\tau,\tau)}{\det G^\prime(\tau,\tau)}.``
- `Δ::F`: Change in the exponentiated on-site energy matrix, ``\Delta_{l,i} = e^{-\Delta\tau (V^\prime_{l,(i,i)} - V_{l,(i,i)})} - 1.``
- `i::Int`: Matrix element of diagonal on-site energy matrix ``V_l`` that is being updated.
- `u::AbstractVector{T}`: Vector of length `size(G,1)` that is used to avoid dynamic memory allocations.
- `v::AbstractVector{T}`: Vector of length `size(G,2)` that is used to avoid dynamic memory allocations.

# Algorithm

The equal-time Green's function matrix is updated using the relationship
```math
G_{j,k}^{\prime}\left(\tau,\tau\right)=G_{j,k}\left(\tau,\tau\right)-\frac{1}{R_{l,i}}G_{j,i}\left(\tau,\tau\right)\Delta_{l,i}\left(\delta_{i,k}-G_{i,k}\left(\tau,\tau\right)\right).
```
The  ``B_l`` progpagator `B` is also udpated.
Additionally, this method returns ``\log( \vert \det G^\prime(\tau,\tau) \vert )`` and ``\textrm{sign}( \det G^\prime(\tau,\tau) ).``

An important note is that if the propagator matrices are represented in a symmetric form, then `G′` and `G` need to correspond
to the transformed eqaul-time Green's function matrices ``\tilde{G}^\prime(\tau,\tau)`` and ``\tilde{G}(\tau,\tau).``
Refer to the [`local_update_det_ratio`](@ref) docstring for more information.
"""
function local_update_greens!(
    G::AbstractMatrix{T}, logdetG::E, sgndetG::T,
    B::AbstractPropagator,
    R::T, Δ::F, i::Int,
    u::AbstractVector{T}, v::AbstractVector{T}
)::Tuple{E,T} where {T<:Continuous, F<:Continuous, E<:AbstractFloat}

    # get diagonal exponentiated potential energy matrix exp(-Δτ⋅V)
    expmΔτV = B.expmΔτV

    # u = G[:,i] <== column vector
    G0i = @view G[:,i]
    copyto!(u, G0i)

    # v = G[i,:] - I[i,:] <== row vector
    @views @. v = conj(G[i,:]) # v = G[i,:]
    v[i] = v[i] - 1.0 # v = G[i,:] - I[i,:]

    # G′ = G + (Δ/R)⋅[u⨂v] = G + (Δ/R)⋅G[:,i]⨂(G[i,:] - I[i,:])
    # where ⨂ denotes an outer product
    BLAS.ger!(Δ/R, u, v, G)

    # R = det(M′)/det(M) = det(G)/det(G′)
    # ==> log(|R|) = log(|det(M′)|) - log(|det(M)|) = log(|det(G)|) - log(|det(G′)|)
    # ==> log(|det(G′)|) = log(|det(G)|) - log(|R|)
    logdetG′ = logdetG - log(abs(R))
    sgndetG′ = sign(R) * sgndetG

    # update the diagonal exponentiated on-site energy matrix appearing in propagator
    # (1 + Δ[i,l]) × exp(-Δτ⋅V[i,l]) = (1 + [exp(-Δτ⋅V′[i,l])/exp(-Δτ⋅V[i,l])-1]) × exp(-Δτ⋅V[i,l])
    #                                = exp(-Δτ⋅V′[i,l])/exp(-Δτ⋅V[i,l]) × exp(-Δτ⋅V[i,l])
    #                                = exp(-Δτ⋅V′[i,l])
    expmΔτV[i] = (1+Δ) * expmΔτV[i]

    return (logdetG′, sgndetG′)
end