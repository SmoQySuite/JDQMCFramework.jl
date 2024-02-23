@doc raw"""
    eval_length_imaginary_axis(
        β::T,
        Δτ::T
    )::Int where {T<:AbstractFloat}

Given an inverse temperature `β` and discretization in imaginary time `Δτ`,
return the length of the imaginary time axis `Lτ`.
"""
function eval_length_imaginary_axis(
    β::T,
    Δτ::T
)::Int where {T<:AbstractFloat}

    Lτ = round(Int, β/Δτ)
    @assert (Lτ*Δτ)≈β
    return Lτ
end

@doc raw"""
    exp!(
        expαH::AbstractMatrix{T},
        H::AbstractMatrix{T},
        α::E;
        # KEYWORD ARGUMENTS
        workspace::HermitianEigenWs{T,Matrix{T},R} = HermitianEigenWs(H),
        tol::R = 1e-6
    ) where {T<:Number, E<:Number, R<:AbstractFloat}

Given a Hermitian matrix `H`, calculate the matrix exponentials ``e^{\alpha H}.``
Note that `H` is left modified by this method. The `workspace` field is of
type [`HermitianEigenWs`](https://dynarejulia.github.io/FastLapackInterface.jl/dev/workspaces/#FastLapackInterface.HermitianEigenWs),
which is exported from the [`FastLapackInterface.jl`](https://github.com/DynareJulia/FastLapackInterface.jl) package,
is used to avoid dynamic memory allocations.
"""
function exp!(
    expαH::AbstractMatrix{T},
    H::AbstractMatrix{T},
    α::E;
    # KEYWORD ARGUMENTS
    workspace::HermitianEigenWs{T,Matrix{T},R} = HermitianEigenWs(H),
    tol::R = 1e-6
) where {T<:Number, E<:Number, R<:AbstractFloat}
    
    # diagonalize the matrix such that H = U⋅ϵ⋅Uᵀ
    LAPACK.syevr!(workspace, 'V', 'A', 'U', H, 0.0, 0.0, 0, 0, tol)
    U  = workspace.Z
    ϵ  = workspace.w
    Uᵀ = adjoint(U)
    
    # calculate the matrix product exp(α⋅H) = U⋅exp(α⋅ϵ)⋅Uᵀ
    @. ϵ = exp(α*ϵ)
    mul_D!(H, ϵ, Uᵀ)
    mul!(expαH, U, H)
    
    return nothing
end

@doc raw"""
    exp!(
        exppαH::AbstractMatrix{T},
        expmαH::AbstractMatrix{T},
        H::AbstractMatrix{T}, α::E;
        # KEYWORD ARGUMENTS
        workspace::HermitianEigenWs{T,Matrix{T},R} = HermitianEigenWs(H),
        tol::R = 1e-6
    ) where {T<:Number, E<:Number, R<:AbstractFloat}

Given a Hermitian matrix `H`, calculate the matrix exponentials ``e^{+\alpha H}``
and ``e^{-\alpha H}``, which are written to `exppαH` and `expmαH` respectively.
Note that `H` is left modified by this method. The `workspace` field is of
type [`HermitianEigenWs`](https://dynarejulia.github.io/FastLapackInterface.jl/dev/workspaces/#FastLapackInterface.HermitianEigenWs),
which is exported from the [`FastLapackInterface.jl`](https://github.com/DynareJulia/FastLapackInterface.jl) package,
is used to avoid dynamic memory allocations.
"""
function exp!(
    exppαH::AbstractMatrix{T},
    expmαH::AbstractMatrix{T},
    H::AbstractMatrix{T}, α::E;
    # KEYWORD ARGUMENTS
    workspace::HermitianEigenWs{T,Matrix{T},R} = HermitianEigenWs(H),
    tol::R = 1e-6
) where {T<:Number, E<:Number, R<:AbstractFloat}
    
    # diagonalize the matrix such that H = U⋅ϵ⋅Uᵀ
    LAPACK.syevr!(workspace, 'V', 'A', 'U', H, 0.0, 0.0, 0, 0, tol)
    U  = workspace.Z
    ϵ  = workspace.w
    Uᵀ = adjoint(U)
    
    # calculate the matrix product exp(α⋅H) = U⋅exp(α⋅ϵ)⋅Uᵀ
    @. ϵ = exp(α*ϵ)
    mul_D!(H, ϵ, Uᵀ)
    mul!(exppαH, U, H)

    # calculate the matrix product exp(-α⋅H) = U⋅exp(-α⋅ϵ)⋅Uᵀ
    div_D!(H, ϵ, Uᵀ)
    mul!(expmαH, U, H)
    
    return nothing
end

@doc raw"""
    build_hopping_matrix!(
        K::AbstractMatrix{T},
        neighbor_table::Matrix{Int},
        t::AbstractVector{T}
    ) where {T<:Continuous}

Construct a hopping matrix `K` using `neighbor_table` along with the corresponding hopping amplitudes `t`.
Each column of `neighbor_table` stores a pair of neighboring orbitals in the lattice, such that `size(neighbor_table,1) = 2`.
"""
function build_hopping_matrix!(
    K::AbstractMatrix{T},
    neighbor_table::Matrix{Int},
    t::AbstractVector{T}
) where {T<:Continuous}

    fill!(K, 0)
    @fastmath @inbounds for n in axes(neighbor_table,2)
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        K[j,i] = -t[n]
        K[i,j] = conj(-t[n])
    end

    return nothing
end