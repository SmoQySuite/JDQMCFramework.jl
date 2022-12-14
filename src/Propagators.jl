@doc raw"""
    abstract type AbstractPropagator{T<:Continuous, E<:AbstractFloat} end

Abstract type to represent imaginary time propagator matrices ``B``. All specific propagators
types inherit from this abstract type. In the above `T` is data type of the matrix elements
in a propagator matrix ``B_l`` and `E` is the type of ``\Delta\tau``.
"""
abstract type AbstractPropagator{T<:Continuous, E<:AbstractFloat} end

@doc raw"""
    abstract type AbstractExactPropagator{T,E} <: AbstractPropagator{T,E} end

Abstract type to represent imaginary time propagator matrices ``B`` defined with an exactly
exponentiated hopping matrix ``K``.
"""
abstract type AbstractExactPropagator{T,E} <: AbstractPropagator{T,E} end

@doc raw"""
    abstract type AbstractChkbrdPropagator{T,E} <: AbstractPropagator{T,E} end

Abstract type to represent imaginary time propagator matrices ``B`` defined with the exponentiated
hopping matrix ``K`` represented by the checkerboard approximation.
"""
abstract type AbstractChkbrdPropagator{T,E} <: AbstractPropagator{T,E} end


@doc raw"""
    SymExactPropagator{T, E} <: AbstractExactPropagator{T,E}

Represents imaginary time propagator matrix as using the symmetric form
```math
B_l = e^{-\Delta\tau K_l/2} e^{-\Delta\tau V_l} e^{-\Delta\tau K_l/2},
```
where ``K_l`` is the strictly off-diagonal hopping matrix and ``V_l``
is the diagonal total on-site energy matrix.

# Fields

- `expmΔτV::Vector{E}`: A vector representing the diagonal exponeniated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτKo2::Matrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l/2}.``
- `exppΔτKo2::Matrix{T}`: Inverse of the exponentiated hopping matrix ``e^{+\Delta\tau K_l/2}.``
"""
struct SymExactPropagator{T,E} <: AbstractExactPropagator{T,E}
    
    "A vector representing the diagonal exponeniated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
    expmΔτV::Vector{E}

    "The exponentiated hopping matrix `exp(-Δτ⋅Kₗ/2)`."
    expmΔτKo2::Matrix{T}

    "Inverse of the exponentiated hopping matrix `exp(+Δτ⋅Kₗ/2)`."
    exppΔτKo2::Matrix{T}
end


@doc raw"""
    AsymExactPropagator{T, E} <: AbstractExactPropagator{T,E}

Represents imaginary time propagator matrix as using the symmetric form
```math
B_l = e^{-\Delta\tau V_l} e^{-\Delta\tau K_l},
```
where ``K_l`` is the strictly off-diagonal hopping matrix and ``V_l``
is the diagonal total on-site energy matrix.

# Fields

- `expmΔτV::Vector{E}`: A vector representing the diagonal exponeniated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτK::Matrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l}.``
- `exppΔτK::Matrix{T}`: Inverse of the exponentiated hopping matrix ``e^{+\Delta\tau K_l}.``
"""
struct AsymExactPropagator{T,E} <: AbstractExactPropagator{T,E}
    
    "A vector representing the diagonal exponeniated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
    expmΔτV::Vector{E}

    "The exponentiated hopping matrix `exp(-Δτ⋅Kₗ)`"
    expmΔτK::Matrix{T}

    "Inverse of the exponentiated hopping matrix `exp(-Δτ⋅Kₗ)`"
    exppΔτK::Matrix{T}
end


@doc raw"""
    SymChkbrdPropagator{T, E} <: AbstractChkbrdPropagator{T,E}

Represents imaginary time propagator matrix as using the symmetric form
```math
B_l = e^{-\Delta\tau K_l/2} e^{-\Delta\tau V_l} [e^{-\Delta\tau K_l/2}]^\dagger,
```
where ``K_l`` is the strictly off-diagonal hopping matrix and ``V_l``
is the diagonal total on-site energy matrix. The exponentiated hopping
matrix ``e^{-\Delta\tau K/2}`` is represented by the checkerboard approximation.

# Fields

- `expmΔτV::Vector{E}`: A vector representing the diagonal exponeniated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτKo2::CheckerboardMatrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l/2}`` represented by the checkerboard approximation.
"""
struct SymChkbrdPropagator{T, E} <: AbstractChkbrdPropagator{T,E}
    
    "A vector representing the diagonal exponeniated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
    expmΔτV::Vector{E}

    "The exponentiated hopping matrix `exp(-Δτ⋅Kₗ/2)` represented by the checkerboard approximation."
    expmΔτKo2::CheckerboardMatrix{T}
end


@doc raw"""
    AsymChkbrdPropagator{T, E} <: AbstractChkbrdPropagator{T,E}

Represents imaginary time propagator matrix as using the symmetric form
```math
B_l = e^{-\Delta\tau V_l} e^{-\Delta\tau K_l},
```
where ``K_l`` is the strictly off-diagonal hopping matrix and ``V_l``
is the diagonal total on-site energy matrix. The exponentiated hopping
matrix ``e^{-\Delta\tau K}`` is represented by the checkerboard approximation.

# Fields

- `expmΔτV::Vector{E}`: The vector representing the diagonal exponeniated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτK::CheckerboardMatrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l}`` represented by the checkerboard approximation.
"""
struct AsymChkbrdPropagator{T,E} <: AbstractChkbrdPropagator{T,E}
    
    "The vector representing the diagonal exponeniated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
    expmΔτV::Vector{E}

    "The exponentiated hopping matrix `exp(-Δτ⋅Kₗ/2)` represented by the checkerboard approximation."
    expmΔτK::CheckerboardMatrix{T}
end


@doc raw"""
    SymPropagators

A union of the all the symmetric propagators types to help test whether a propagator type is symmetric.
Assuming `typeof{B} <: AbstractPropagator` returns `true`, if `typeof(B) <: SymPropagators` returns
`true`, then `B` represents a symmetric propagator, otherwise it represents an asymmetric propagator.
"""
SymPropagators = Union{SymExactPropagator, SymChkbrdPropagator}


@doc raw"""
    size(B::AbstractPropagator)

    size(B::AbstractPropagator, dim)

Return the size of a propagator.
"""
size(B::AbstractPropagator) = (size(B.expmΔτV), size(B.expmΔτV))
size(B::AbstractPropagator, dim) = size(B.expmΔτV)


@doc raw"""
    ishermitian(B::AbstractPropagator)

Return whether a propagator is hermitian or not.
"""
ishermitian(B::AbstractPropagator)    = return _ishermitian(B)
_ishermitian(B::SymExactPropagator)   = return true
_ishermitian(B::SymChkbrdPropagator)  = return true
_ishermitian(B::AsymExactPropagator)  = return false
_ishermitian(B::AsymChkbrdPropagator) = return false


@doc raw"""
    copyto!(B′::SymExactPropagator{T,E}, B::SymExactPropagator{T,E}) where {T,E}

    copyto!(B′::AsymExactPropagator{T,E}, B::AsymExactPropagator{T,E}) where {T,E}

    copyto!(B′::SymChkbrdPropagator{T,E}, B::SymChkbrdPropagator{T,E}) where {T,E}

    copyto!(B′::AsymChkbrdPropagator{T,E}, B::AsymChkbrdPropagator{T,E}) where {T,E}

Copy the propagator `B` to `B′`.
"""
function copyto!(B′::SymExactPropagator{T,E}, B::SymExactPropagator{T,E}) where {T,E}

    copyto!(B′.expmΔτV, B.expmΔτV)
    copyto!(B′.expmΔτKo2, B.expmΔτKo2)
    copyto!(B′.exppΔτKo2, B.exppΔτKo2)

    return nothing
end

function copyto!(B′::AsymExactPropagator{T,E}, B::AsymExactPropagator{T,E}) where {T,E}

    copyto!(B′.expmΔτV, B.expmΔτV)
    copyto!(B′.expmΔτK, B.expmΔτK)
    copyto!(B′.exppΔτK, B.exppΔτK)

    return nothing
end

function copyto!(B′::SymChkbrdPropagator{T,E}, B::SymChkbrdPropagator{T,E}) where {T,E}

    copyto!(B′.expmΔτV, B.expmΔτV)
    copyto!(B′.expmΔτKo2, B.expmΔτKo2)

    return nothing
end

function copyto!(B′::AsymChkbrdPropagator{T,E}, B::AsymChkbrdPropagator{T,E}) where {T,E}

    copyto!(B′.expmΔτV, B.expmΔτV)
    copyto!(B′.expmΔτK, B.expmΔτK)

    return nothing
end


@doc raw"""
    eltype(B::AbstractPropagator{T,E}) where {T,E}

Return the matrix element type of the propagator `T`.
"""
function eltype(B::AbstractPropagator{T,E}) where {T,E}

    return T
end


@doc raw"""
    mul!(A::AbstractMatrix{T}, B::SymExactPropagator{T}, C::AbstractMatrix{T};
         M::AbstractMatrix{T}=similar(A)) where {T}

    mul!(A::AbstractMatrix{T}, B::AsymExactPropagator{T}, C::AbstractMatrix{T};
         M::AbstractMatrix{T}=similar(A)) where {T}

    mul!(A::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T}, C::AbstractMatrix{T};
         M=nothing) where {T}

Calculate the product ``A := B \cdot C``, where ``B`` is a propagator matrix represented
by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A = B^\dagger \cdot C`` is evaluated instead.
"""
function mul!(A::AbstractMatrix{T}, B::SymExactPropagator{T}, C::AbstractMatrix{T};
              M::AbstractMatrix{T}=similar(A)) where {T}

    mul!(M, B.expmΔτKo2, C) # exp(-Δτ⋅K/2)⋅C
    lmul_D!(B.expmΔτV, M) # exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅C
    mul!(A, B.expmΔτKo2, M) # A = B⋅C = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅C

    return nothing
end

function mul!(A::AbstractMatrix{T}, B::AsymExactPropagator{T}, C::AbstractMatrix{T};
              M::AbstractMatrix{T}=similar(A)) where {T}

    mul!(A, B.expmΔτK, C) # exp(-Δτ⋅K)⋅C
    lmul_D!(B.expmΔτV, A) # A = B⋅C = exp(-Δτ⋅V)⋅[exp(-Δτ⋅K)⋅C]

    return nothing
end

function mul!(A::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T}, C::AbstractMatrix{T};
              M=nothing) where {T}

    copyto!(A, C)
    lmul!(B, A)

    return nothing
end


@doc raw"""
    mul!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::SymExactPropagator{T};
         M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AsymExactPropagator{T};
         M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T};
         M=nothing) where {T}

Calculate the matrix product ``A := C \cdot B``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A = C \cdot B^\dagger`` is evaluated instead.
"""
function mul!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::SymExactPropagator{T};
              M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(M, C, B.expmΔτKo2) # C⋅exp(-Δτ⋅K/2)
    rmul_D!(M, B.expmΔτV) # C⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτKo2) # A := B⋅C = C⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)

    return nothing
end

function mul!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AsymExactPropagator{T};
              M::AbstractMatrix{T} = similar(A)) where {T}

    mul_D!(M, C, B.expmΔτV) # C⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτK) # A := C⋅B = [C⋅exp(-Δτ⋅V)]⋅exp(-Δτ⋅K)

    return nothing
end

function mul!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T};
              M = nothing) where {T}

    copyto!(A, C)
    rmul!(A, B)

    return nothing
end


@doc raw"""
    lmul!(B::SymExactPropagator{T}, A::AbstractMatrix{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    lmul!(B::AsymExactPropagator{T}, A::AbstractMatrix{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    lmul!(B::AsymExactPropagator{T}, A::AbstractMatrix{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    lmul!(B::AsymChkbrdPropagator{T}, A::AbstractMatrix{T};
          M = nothing) where {T}

Calculate the matrix product ``A := B \cdot A``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, when ``A := B^\dagger \cdot A`` is evaluated instead.
"""
function lmul!(B::SymExactPropagator{T}, A::AbstractMatrix{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(M, B.expmΔτKo2, A) # exp(-Δτ⋅K/2)⋅A
    lmul_D!(B.expmΔτV, M) # exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅A
    mul!(A, B.expmΔτKo2, M) # A := B⋅A = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅A
    
    return nothing
end

function lmul!(B::AsymExactPropagator{T}, A::AbstractMatrix{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(M, B.expmΔτK, A) # exp(-Δτ⋅K)⋅A
    mul_D!(A, B.expmΔτV, M) # A := B⋅A = exp(-Δτ⋅V)⋅[exp(-Δτ⋅K)⋅A]

    return nothing
end

function lmul!(B::SymChkbrdPropagator{T}, A::AbstractMatrix{T};
               M = nothing) where {T}

    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    lmul!(expmΔτKo2ᵀ, A) # exp(-Δτ⋅K/2)ᵀ⋅A
    lmul_D!(B.expmΔτV, A) # exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ⋅A
    lmul!(B.expmΔτKo2, A) # A := B⋅A = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ⋅A

    return nothing
end

function lmul!(B::AsymChkbrdPropagator{T}, A::AbstractMatrix{T};
               M = nothing) where {T}

    lmul!(B.expmΔτK, A) # exp(-Δτ⋅K)⋅A
    lmul_D!(B.expmΔτV, A) # A := B⋅A = exp(-Δτ⋅V)⋅[exp(-Δτ⋅K)⋅A]

    return nothing
end


@doc raw"""
    rmul!(A::AbstractMatrix{T}, B::SymExactPropagator{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    rmul!(A::AbstractMatrix{T}, B::AsymExactPropagator{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    rmul!(A::AbstractMatrix{T}, B::SymChkbrdPropagator{T};
          M = nothing) where {T}

    rmul!(A::AbstractMatrix{T}, B::AsymChkbrdPropagator{T};
          M = nothing) where {T}

Calculate the matrix product ``A := A \cdot B``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A := A \cdot B^\dagger`` is evaluated instead. 
"""
function rmul!(A::AbstractMatrix{T}, B::SymExactPropagator{T};
               M::AbstractMatrix{T} = similar(A)) where {T}
    
    mul!(M, A, B.expmΔτKo2) # A⋅exp(-Δτ⋅K/2)
    rmul_D!(M, B.expmΔτV) # A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτKo2) # A := A⋅B = A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    
    return nothing
end

function rmul!(A::AbstractMatrix{T}, B::AsymExactPropagator{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    mul_D!(M, A, B.expmΔτV) # A⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτK) # A := A⋅B = [A⋅exp(-Δτ⋅V)]⋅exp(-Δτ⋅K)

    return nothing
end

function rmul!(A::AbstractMatrix{T}, B::SymChkbrdPropagator{T};
               M = nothing) where {T}

    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    rmul!(A, B.expmΔτKo2) # A⋅exp(-Δτ⋅K/2)
    rmul_D!(A, B.expmΔτV) # A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)
    rmul!(A, expmΔτKo2ᵀ) # A := A⋅B = A⋅exp(-Δτ⋅K/2)ᵀ⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ

    return nothing
end

function rmul!(A::AbstractMatrix{T}, B::AsymChkbrdPropagator{T};
               M = nothing) where {T}

    rmul_D!(A, B.expmΔτV) # A⋅exp(-Δτ⋅V)
    rmul!(A, B.expmΔτK) # A := A⋅B = [A⋅exp(-Δτ⋅V)]⋅exp(-Δτ⋅K)

    return nothing
end


@doc raw"""
    ldiv!(A::AbstractMatrix{T}, B::AbstractExactPropagator{T}, C::AbstractMatrix{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    ldiv!(A::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T}, C::AbstractMatrix{T};
          M = nothing) where {T}

Calculate the matrix product ``A := B^{-1} \cdot C``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A = [B^\dagger]^{-1} \cdot C`` is evaluated instead.
"""
function ldiv!(A::AbstractMatrix{T}, B::AbstractExactPropagator{T}, C::AbstractMatrix{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    copyto!(A, C)
    ldiv!(B, A, M=M)

    return nothing
end

function ldiv!(A::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T}, C::AbstractMatrix{T};
               M = nothing) where {T}

    copyto!(A, C)
    ldiv!(B, A)

    return nothing
end


@doc raw"""
    ldiv!(B::SymExactPropagator{T}, A::AbstractMatrix{T};
          M::AbstractMatrix{T} = similar(A)) where {T}
    
    ldiv!(B::AsymExactPropagator{T}, A::AbstractMatrix{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    ldiv!(B::SymChkbrdPropagator{T}, A::AbstractMatrix{T};
          M = nothing) where {T}

    ldiv!(B::AsymChkbrdPropagator{T}, A::AbstractMatrix{T};
          M = nothing) where {T}

Calculate the matrix product ``A := B^{-1} \cdot A``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A := [B^\dagger]^{-1} \cdot A`` is evaluated instead.
"""
function ldiv!(B::SymExactPropagator{T}, A::AbstractMatrix{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(M, B.exppΔτKo2, A) # exp(+Δτ⋅K/2)⋅A
    ldiv_D!(B.expmΔτV, M) # exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    mul!(A, B.exppΔτKo2, M) # A := B⁻¹⋅A = exp(+Δτ⋅K/2)⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    
    return nothing
end

function ldiv!(B::AsymExactPropagator{T}, A::AbstractMatrix{T};
               M::AbstractMatrix{T} = similar(A)) where {T}
    
    div_D!(M, B.expmΔτV, A) # exp(+Δτ⋅V)⋅A
    mul!(A, B.exppΔτK, M) # A := B⁻¹⋅A = exp(+Δτ⋅K)⋅[exp(+Δτ⋅V)⋅A]
    
    return nothing
end

function ldiv!(B::SymChkbrdPropagator{T}, A::AbstractMatrix{T};
               M = nothing) where {T}
    
    expmΔτKo2ᵀ = transpose(B.expmΔτKo2)
    ldiv!(B.expmΔτKo2, A) # exp(+Δτ⋅K/2)⋅A
    ldiv_D!(B.expmΔτV, A) # exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    ldiv!(expmΔτKo2ᵀ, A) # A := B⁻¹⋅A = [exp(+Δτ⋅K/2)]ᵀ⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    
    return nothing
end

function ldiv!(B::AsymChkbrdPropagator{T}, A::AbstractMatrix{T};
               M = nothing) where {T}

    ldiv_D!(B.expmΔτV, A) # exp(+Δτ⋅V)⋅A
    ldiv!(B.expmΔτK, A) # A := B⁻¹⋅A = exp(+Δτ⋅K)⋅[exp(+Δτ⋅V)⋅A]
    
    return nothing
end


@doc raw"""
    rldiv!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractExactPropagator{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    rdiv!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T};
          M = nothing) where {T}

Calculate the matrix product ``A := C \cdot B^{-1}``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B` is asymmetric and `B.adjointed = true`, then ``A = C \cdot [B^\dagger]^{-1}`` is evaluated instead.
"""
function rdiv!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractExactPropagator{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    copyto!(A, C)
    rdiv!(A, B, M=M)

    return nothing
end

function rdiv!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractChkbrdPropagator{T};
               M = nothing) where {T}

    copyto!(A, C)
    rdiv!(A, B)

    return nothing
end


@doc raw"""
    rdiv!(A::AbstractMatrix{T}, B::SymExactPropagator{T};
          M::AbstractMatrix{T} = similar(A)) where {T}
    
    rdiv!(A::AbstractMatrix{T}, B::AsymExactPropagator{T};
          M::AbstractMatrix{T} = similar(A)) where {T}

    rdiv!(A::AbstractMatrix{T}, B::SymChkbrdPropagator{T};
          M = nothing) where {T}

    rdiv!(A::AbstractMatrix{T}, B::AsymChkbrdPropagator{T};
          M = nothing) where {T}

Calculate the matrix product ``A := A \cdot B^{-1}``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A := A \cdot [B^\dagger]^{-1}`` is evaluated instead.
"""
function rdiv!(A::AbstractMatrix{T}, B::SymExactPropagator{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(M, A, B.exppΔτKo2) # A⋅exp(+Δτ⋅K/2)
    rdiv_D!(M, B.expmΔτV) # A⋅exp(+Δτ⋅K/2)⋅exp(+Δτ⋅V)
    mul!(A, M, B.exppΔτKo2) # A := A⋅B⁻¹ = A⋅exp(+Δτ⋅K/2)⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)
    
    return nothing
end

function rdiv!(A::AbstractMatrix{T}, B::AsymExactPropagator{T};
               M::AbstractMatrix{T} = similar(A)) where {T}

    mul!(M, A, B.exppΔτK) # A⋅exp(+Δτ⋅K)
    div_D!(A, M, B.expmΔτV) # A := A⋅B⁻¹ = [A⋅exp(+Δτ⋅K)]⋅exp(+Δτ⋅V)

    return nothing
end

function rdiv!(A::AbstractMatrix{T}, B::SymChkbrdPropagator{T};
               M = nothing) where {T}

    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    rdiv!(A, expmΔτKo2ᵀ) # A⋅[exp(+Δτ⋅K/2)]ᵀ
    rdiv_D!(A, B.expmΔτV) # A⋅[exp(+Δτ⋅K/2)]ᵀ⋅exp(+Δτ⋅V)
    rdiv!(A, B.expmΔτKo2) # A := A⋅B⁻¹ = A⋅[exp(+Δτ⋅K/2)]ᵀ⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)
    
    return nothing
end

function rdiv!(A::AbstractMatrix{T}, B::AsymChkbrdPropagator{T};
               M = nothing) where {T}

    rdiv!(A, B.expmΔτK) # A⋅exp(+ΔτK)
    rdiv_D!(A, B.expmΔτV) # A := A⋅B⁻¹ = [A⋅exp(+ΔτK)]⋅exp(+Δτ⋅V)

    return nothing
end