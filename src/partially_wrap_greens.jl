@doc raw"""
    partially_wrap_greens_reverse!(
        G::Matrix{H},
        B::AbstractPropagator,
        M::Matrix{H} = similar(G)
    ) where {H<:Continuous}

If the propagator `B` is represented in the symmetric form
```math
B_l = \Gamma_l(\Delta\tau/2) \cdot \Lambda_l(\Delta\tau) \cdot \Gamma_l^\dagger(\Delta\tau/2)
```
with ``\tau = l \cdot \Delta\tau,`` where ``\Gamma(\Delta\tau/2) = e^{-\Delta\tau K_l/2}`` and ``\Lambda(\Delta\tau) = e^{-\Delta\tau V_l}``,
then apply the transformation
```math
\tilde{G}(\tau,\tau) = \Gamma^{-1}_l(\Delta\tau/2) \cdot G(\tau,\tau) \cdot \Gamma_l(\Delta\tau/2)
```
to the equal-time Green's function matrix `G` in-place.
"""
function partially_wrap_greens_reverse!(
    G::Matrix{H},
    B::AbstractPropagator,
    M::Matrix{H} = similar(G)
) where {H<:Continuous}

    # only apply transformation if symmetric/hermitian definition for propagator is being used
    _partially_wrap_greens_reverse!(G, B, M)

    return nothing
end

# perform the G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l] transformation in the case the Γ[l]=exp(-Δτ⋅K[l]/2) is the exactly exponentiated hopping matrix
function _partially_wrap_greens_reverse!(G::Matrix{H}, B::SymExactPropagator, M::Matrix{H}) where {H<:Continuous}

    (; expmΔτKo2, exppΔτKo2) = B
    mul!(M, G, expmΔτKo2) # G(τ,τ)⋅Γ[l]
    mul!(G, exppΔτKo2, M) # G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l]

    return nothing
end

# perform the G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l] transformation in the case the Γ[l] is the checkerboard approximation of exp(-Δτ⋅K[l]/2)
function _partially_wrap_greens_reverse!(G::Matrix, B::SymChkbrdPropagator, ignore...)

    (; expmΔτKo2) = B
    rmul!(G, expmΔτKo2) # G(τ,τ)⋅Γ[l]
    ldiv!(expmΔτKo2, G) # G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l]

    return nothing
end

# do nothing for asymmetric propagator
function _partially_wrap_greens_reverse!(G::Matrix, B::AsymExactPropagator, ignore...)

    return nothing
end

# do nothing for asymmetric propagator
function _partially_wrap_greens_reverse!(G::Matrix, B::AsymChkbrdPropagator, ignore...)

    return nothing
end

@doc raw"""
    partially_wrap_greens_forward!(
        G::Matrix{H},
        B::AbstractPropagator,
        M::Matrix{H} = similar(G)
    ) where {H<:Continuous}

If the propagator `B` is represented in the symmetric form
```math
B_l = \Gamma_l(\Delta\tau/2) \cdot \Lambda_l(\Delta\tau) \cdot \Gamma_l^\dagger(\Delta\tau/2)
```
with ``\tau = l \cdot \Delta\tau,`` where ``\Gamma(\Delta\tau/2) = e^{-\Delta\tau K_l/2}`` and ``\Lambda(\Delta\tau) = e^{-\Delta\tau V_l}``,
then apply the transformation
```math
G(\tau,\tau) = \Gamma_l(\Delta\tau/2) \cdot \tilde{G}(\tau,\tau) \cdot \Gamma_l^{-1}(\Delta\tau/2)
```
to the equal-time Green's function matrix `G` in-place.
"""
function partially_wrap_greens_forward!(
    G::Matrix{H},
    B::AbstractPropagator,
    M::Matrix{H} = similar(G)
) where {H<:Continuous}

    # only apply transformation if symmetric/hermitian definition for propagator is being used
    _partially_wrap_greens_forward!(G, B, M)

    return nothing
end

# perform the G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l] transformation in the case the Γ[l]=exp(-Δτ⋅K[l]/2) is the exactly exponentiated hopping matrix
function _partially_wrap_greens_forward!(G::Matrix{H}, B::SymExactPropagator, M::Matrix{H}) where {H<:Continuous}

    (; expmΔτKo2, exppΔτKo2) = B
    mul!(M, G, exppΔτKo2) # G̃(τ,τ)⋅Γ⁻¹[l]
    mul!(G, expmΔτKo2, M) # G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l]

    return nothing
end

# perform the G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l] transformation in the case the Γ[l] is the checkerboard approximation of exp(-Δτ⋅K[l]/2)
function _partially_wrap_greens_forward!(G::Matrix, B::SymChkbrdPropagator, ignore...)

    (; expmΔτKo2) = B
    rdiv!(G, expmΔτKo2) # G̃(τ,τ)⋅Γ⁻¹[l]
    lmul!(expmΔτKo2, G) # G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l]

    return nothing
end

# do nothing for asymmetric propagator
function _partially_wrap_greens_forward!(G::Matrix, B::AsymExactPropagator, ignore...)

    return nothing
end

# do nothing for asymmetric propagator
function _partially_wrap_greens_forward!(G::Matrix, B::AsymChkbrdPropagator, ignore...)

    return nothing
end