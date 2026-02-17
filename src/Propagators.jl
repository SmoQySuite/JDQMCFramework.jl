@doc raw"""
    abstract type AbstractPropagator{T<:Continuous, E<:Continuous} end

Abstract type to represent imaginary time propagator matrices ``B``. All specific propagators
types inherit from this abstract type. In the above `T` is data type of the matrix elements
of the exponentiated kinetic energy matrix ``e^{-\Delta\tau K_l}`` appearing in ``B_l``, and `E` is data type of the
matrix elements appearing in the diagonal exponentiated potential energy matrix ``e^{-\Delta\tau V_l}``.
"""
abstract type AbstractPropagator{T<:Continuous, E<:Continuous} end

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

- `expmΔτV::Vector{E}`: A vector representing the diagonal exponentiated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτKo2::Matrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l/2}.``
- `exppΔτKo2::Matrix{T}`: Inverse of the exponentiated hopping matrix ``e^{+\Delta\tau K_l/2}.``
"""
struct SymExactPropagator{T,E} <: AbstractExactPropagator{T,E}
    
    "A vector representing the diagonal exponentiated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
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

- `expmΔτV::Vector{E}`: A vector representing the diagonal exponentiated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτK::Matrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l}.``
- `exppΔτK::Matrix{T}`: Inverse of the exponentiated hopping matrix ``e^{+\Delta\tau K_l}.``
"""
struct AsymExactPropagator{T,E} <: AbstractExactPropagator{T,E}
    
    "A vector representing the diagonal exponentiated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
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

- `expmΔτV::Vector{E}`: A vector representing the diagonal exponentiated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτKo2::CheckerboardMatrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l/2}`` represented by the checkerboard approximation.
"""
struct SymChkbrdPropagator{T, E} <: AbstractChkbrdPropagator{T,E}
    
    "A vector representing the diagonal exponentiated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
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

- `expmΔτV::Vector{E}`: The vector representing the diagonal exponentiated on-site energy matrix ``e^{-\Delta\tau V_l}.``
- `expmΔτK::CheckerboardMatrix{T}`: The exponentiated hopping matrix ``e^{-\Delta\tau K_l}`` represented by the checkerboard approximation.
"""
struct AsymChkbrdPropagator{T,E} <: AbstractChkbrdPropagator{T,E}
    
    "The vector representing the diagonal exponentiated on-site energy matrix `exp(-Δτ⋅Vₗ)`."
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
size(B::AbstractPropagator) = (length(B.expmΔτV), length(B.expmΔτV))
size(B::AbstractPropagator, dim::Int) = length(B.expmΔτV)


@doc raw"""
    ishermitian(B::AbstractPropagator)

Return whether a propagator is hermitian or not.
"""
ishermitian(B::AbstractPropagator) = return _ishermitian(B)
_ishermitian(B::SymExactPropagator{T,E}) where {T,E} = return (E <: AbstractFloat)
_ishermitian(B::SymChkbrdPropagator{T,E}) where {T,E} = return (E <: AbstractFloat)
_ishermitian(B::AsymExactPropagator) = return false
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

    V = real(E)
    V = (E <: AbstractFloat) ? V : E
    V = (T <: AbstractFloat) ? V : T
    return V
end


@doc raw"""
    mul!(
        A::AbstractVecOrMat, B::SymExactPropagator, C::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )

    mul!(
        A::AbstractVecOrMat, B::AsymExactPropagator, C::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )

    mul!(
        A::AbstractVecOrMat, B::AbstractChkbrdPropagator, C::AbstractVecOrMat;
        M = nothing
    )

Calculate the product ``A := B \cdot C``, where ``B`` is a propagator matrix represented
by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A = B^\dagger \cdot C`` is evaluated instead.
"""
function mul!(
    A::AbstractVecOrMat, B::SymExactPropagator, C::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    mul!(M, B.expmΔτKo2, C) # exp(-Δτ⋅K/2)⋅C
    expmΔτV = Diagonal(B.expmΔτV)
    lmul!(expmΔτV, M) # exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅C
    mul!(A, B.expmΔτKo2, M) # A = B⋅C = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅C

    return nothing
end

function mul!(
    A::AbstractVecOrMat, B::AsymExactPropagator, C::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    mul!(A, B.expmΔτK, C) # exp(-Δτ⋅K)⋅C
    expmΔτV = Diagonal(B.expmΔτV)
    lmul!(expmΔτV, A) # A = B⋅C = exp(-Δτ⋅V)⋅[exp(-Δτ⋅K)⋅C]

    return nothing
end

function mul!(
    A::AbstractVecOrMat, B::AbstractChkbrdPropagator, C::AbstractVecOrMat;
    M = nothing
)

    copyto!(A, C)
    lmul!(B, A)

    return nothing
end


@doc raw"""
    mul!(
        A::AbstractVecOrMat, C::AbstractVecOrMat, B::SymExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )

    mul!(
        A::AbstractVecOrMat, C::AbstractVecOrMat, B::AsymExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )

    mul!(
        A::AbstractVecOrMat, C::AbstractVecOrMat, B::AbstractChkbrdPropagator;
        M = nothing
    )

Calculate the matrix product ``A := C \cdot B``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A = C \cdot B^\dagger`` is evaluated instead.
"""
function mul!(
    A::AbstractVecOrMat, C::AbstractVecOrMat, B::SymExactPropagator;
    M::AbstractVecOrMat = similar(A)
)

    expmΔτV = Diagonal(B.expmΔτV)
    mul!(M, C, B.expmΔτKo2) # C⋅exp(-Δτ⋅K/2)
    rmul!(M, expmΔτV) # C⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτKo2) # A := B⋅C = C⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)

    return nothing
end

function mul!(
    A::AbstractVecOrMat, C::AbstractVecOrMat, B::AsymExactPropagator;
    M::AbstractVecOrMat = similar(A)
)

    expmΔτV = Diagonal(B.expmΔτV)
    mul!(M, C, expmΔτV) # C⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτK) # A := C⋅B = [C⋅exp(-Δτ⋅V)]⋅exp(-Δτ⋅K)

    return nothing
end

function mul!(
    A::AbstractVecOrMat, C::AbstractVecOrMat, B::AbstractChkbrdPropagator;
    M = nothing
)

    copyto!(A, C)
    rmul!(A, B)

    return nothing
end


@doc raw"""
    lmul!(
        B::SymExactPropagator, A::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )

    lmul!(
        B::AsymExactPropagator, A::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )

    lmul!(
        B::SymChkbrdPropagator, A::AbstractVecOrMat;
        M = nothing
    )

    lmul!(
        B::AsymChkbrdPropagator, A::AbstractVecOrMat;
        M = nothing
    )

Calculate the matrix product ``A := B \cdot A``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, when ``A := B^\dagger \cdot A`` is evaluated instead.
"""
function lmul!(
    B::SymExactPropagator, A::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    mul!(M, B.expmΔτKo2, A) # exp(-Δτ⋅K/2)⋅A
    expmΔτV = Diagonal(B.expmΔτV)
    lmul!(expmΔτV, M) # exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅A
    mul!(A, B.expmΔτKo2, M) # A := B⋅A = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)⋅A
    
    return nothing
end

function lmul!(
    B::AsymExactPropagator, A::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    mul!(M, B.expmΔτK, A) # exp(-Δτ⋅K)⋅A
    expmΔτV = Diagonal(B.expmΔτV)
    mul!(A, expmΔτV, M) # A := B⋅A = exp(-Δτ⋅V)⋅[exp(-Δτ⋅K)⋅A]

    return nothing
end

function lmul!(
    B::SymChkbrdPropagator, A::AbstractVecOrMat;
    M = nothing
)

    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    lmul!(expmΔτKo2ᵀ, A) # exp(-Δτ⋅K/2)ᵀ⋅A
    expmΔτV = Diagonal(B.expmΔτV)
    lmul!(expmΔτV, A) # exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ⋅A
    lmul!(B.expmΔτKo2, A) # A := B⋅A = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ⋅A

    return nothing
end

function lmul!(
    B::AsymChkbrdPropagator, A::AbstractVecOrMat;
    M = nothing
)

    lmul!(B.expmΔτK, A) # exp(-Δτ⋅K)⋅A
    expmΔτV = Diagonal(B.expmΔτV)
    lmul!(expmΔτV, A) # A := B⋅A = exp(-Δτ⋅V)⋅[exp(-Δτ⋅K)⋅A]

    return nothing
end


@doc raw"""
    rmul!(
        A::AbstractVecOrMat, B::SymExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )

    rmul!(
        A::AbstractVecOrMat, B::AsymExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )

    rmul!(
        A::AbstractVecOrMat, B::SymChkbrdPropagator;
        M = nothing
    )

    rmul!(
        A::AbstractVecOrMat, B::AsymChkbrdPropagator;
        M = nothing
    )

Calculate the matrix product ``A := A \cdot B``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A := A \cdot B^\dagger`` is evaluated instead. 
"""
function rmul!(
    A::AbstractVecOrMat, B::SymExactPropagator;
    M::AbstractVecOrMat = similar(A)
)
    
    mul!(M, A, B.expmΔτKo2) # A⋅exp(-Δτ⋅K/2)
    expmΔτV = Diagonal(B.expmΔτV)
    rmul!(M, expmΔτV) # A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτKo2) # A := A⋅B = A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    
    return nothing
end

function rmul!(
    A::AbstractVecOrMat, B::AsymExactPropagator;
    M::AbstractVecOrMat = similar(A)
)

    expmΔτV = Diagonal(B.expmΔτV)
    mul!(M, A, expmΔτV) # A⋅exp(-Δτ⋅V)
    mul!(A, M, B.expmΔτK) # A := A⋅B = [A⋅exp(-Δτ⋅V)]⋅exp(-Δτ⋅K)

    return nothing
end

function rmul!(
    A::AbstractVecOrMat, B::SymChkbrdPropagator;
    M = nothing
)

    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    rmul!(A, B.expmΔτKo2) # A⋅exp(-Δτ⋅K/2)
    expmΔτV = Diagonal(B.expmΔτV)
    rmul!(A, expmΔτV) # A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)
    rmul!(A, expmΔτKo2ᵀ) # A := A⋅B = A⋅exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)ᵀ

    return nothing
end

function rmul!(
    A::AbstractVecOrMat, B::AsymChkbrdPropagator;
    M = nothing
)

    expmΔτV = Diagonal(B.expmΔτV)
    rmul!(A, expmΔτV) # A⋅exp(-Δτ⋅V)
    rmul!(A, B.expmΔτK) # A := A⋅B = [A⋅exp(-Δτ⋅V)]⋅exp(-Δτ⋅K)

    return nothing
end


@doc raw"""
    ldiv!(
        A::AbstractVecOrMat, B::AbstractExactPropagator, C::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )

    ldiv!(
        A::AbstractVecOrMat, B::AbstractChkbrdPropagator, C::AbstractVecOrMat;
        M = nothing
    )

Calculate the matrix product ``A := B^{-1} \cdot C``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A = [B^\dagger]^{-1} \cdot C`` is evaluated instead.
"""
function ldiv!(
    A::AbstractVecOrMat, B::AbstractExactPropagator, C::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    copyto!(A, C)
    ldiv!(B, A, M=M)

    return nothing
end

function ldiv!(
    A::AbstractVecOrMat, B::AbstractChkbrdPropagator, C::AbstractVecOrMat;
    M = nothing
)

    copyto!(A, C)
    ldiv!(B, A)

    return nothing
end


@doc raw"""
    ldiv!(
        B::SymExactPropagator, A::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )
    
    ldiv!(
        B::AsymExactPropagator, A::AbstractVecOrMat;
        M::AbstractVecOrMat = similar(A)
    )

    ldiv!(
        B::SymChkbrdPropagator, A::AbstractVecOrMat;
        M = nothing
    )

    ldiv!(
        B::AsymChkbrdPropagator, A::AbstractVecOrMat;
        M = nothing
    )

Calculate the matrix product ``A := B^{-1} \cdot A``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A := [B^\dagger]^{-1} \cdot A`` is evaluated instead.
"""
function ldiv!(
    B::SymExactPropagator, A::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    mul!(M, B.exppΔτKo2, A) # exp(+Δτ⋅K/2)⋅A
    expmΔτV = Diagonal(B.expmΔτV)
    ldiv!(expmΔτV, M) # exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    mul!(A, B.exppΔτKo2, M) # A := B⁻¹⋅A = exp(+Δτ⋅K/2)⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    
    return nothing
end

function ldiv!(
    B::AsymExactPropagator, A::AbstractVecOrMat;
    M::AbstractVecOrMat = similar(A)
)

    expmΔτV = Diagonal(B.expmΔτV)
    ldiv!(M, expmΔτV, A) # exp(+Δτ⋅V)⋅A
    mul!(A, B.exppΔτK, M) # A := B⁻¹⋅A = exp(+Δτ⋅K)⋅[exp(+Δτ⋅V)⋅A]
    
    return nothing
end

function ldiv!(
    B::SymChkbrdPropagator, A::AbstractVecOrMat;
    M = nothing
)
    
    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    ldiv!(B.expmΔτKo2, A) # exp(+Δτ⋅K/2)⋅A
    expmΔτV = Diagonal(B.expmΔτV)
    ldiv!(expmΔτV, A) # exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    ldiv!(expmΔτKo2ᵀ, A) # A := B⁻¹⋅A = [exp(+Δτ⋅K/2)]ᵀ⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)⋅A
    
    return nothing
end

function ldiv!(
    B::AsymChkbrdPropagator, A::AbstractVecOrMat;
    M = nothing
)

    expmΔτV = Diagonal(B.expmΔτV)
    ldiv!(expmΔτV, A) # exp(+Δτ⋅V)⋅A
    ldiv!(B.expmΔτK, A) # A := B⁻¹⋅A = exp(+Δτ⋅K)⋅[exp(+Δτ⋅V)⋅A]
    
    return nothing
end


@doc raw"""
    rdiv!(
        A::AbstractVecOrMat, C::AbstractVecOrMat, B::AbstractExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )

    rdiv!(
        A::AbstractMatrix, C::AbstractMatrix, B::AbstractChkbrdPropagator;
        M = nothing
    )

Calculate the matrix product ``A := C \cdot B^{-1}``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B` is asymmetric and `B.adjointed = true`, then ``A = C \cdot [B^\dagger]^{-1}`` is evaluated instead.
"""
function rdiv!(
    A::AbstractVecOrMat, C::AbstractVecOrMat, B::AbstractExactPropagator;
    M::AbstractVecOrMat = similar(A)
)

    copyto!(A, C)
    rdiv!(A, B, M=M)

    return nothing
end

function rdiv!(
    A::AbstractVecOrMat, C::AbstractVecOrMat, B::AbstractChkbrdPropagator;
    M = nothing
)

    copyto!(A, C)
    rdiv!(A, B)

    return nothing
end


@doc raw"""
    rdiv!(
        A::AbstractVecOrMat, B::SymExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )
    
    rdiv!(
        A::AbstractVecOrMat, B::AsymExactPropagator;
        M::AbstractVecOrMat = similar(A)
    )

    rdiv!(
        A::AbstractVecOrMat, B::SymChkbrdPropagator;
        M = nothing
    )

    rdiv!(
        A::AbstractVecOrMat, B::AsymChkbrdPropagator;
        M = nothing
    )

Calculate the matrix product ``A := A \cdot B^{-1}``, where ``B`` is a propagator matrix
represented by an instance of a type inheriting from [`AbstractPropagator`](@ref).
If `B` is asymmetric and `B.adjointed = true`, then ``A := A \cdot [B^\dagger]^{-1}`` is evaluated instead.
"""
function rdiv!(
    A::AbstractVecOrMat, B::SymExactPropagator;
    M::AbstractVecOrMat = similar(A)
)

    mul!(M, A, B.exppΔτKo2) # A⋅exp(+Δτ⋅K/2)
    expmΔτV = Diagonal(B.expmΔτV)
    rdiv!(M, expmΔτV) # A⋅exp(+Δτ⋅K/2)⋅exp(+Δτ⋅V)
    mul!(A, M, B.exppΔτKo2) # A := A⋅B⁻¹ = A⋅exp(+Δτ⋅K/2)⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)
    
    return nothing
end

function rdiv!(
    A::AbstractVecOrMat, B::AsymExactPropagator;
    M::AbstractVecOrMat = similar(A)
)

    mul!(M, A, B.exppΔτK) # A⋅exp(+Δτ⋅K)
    expmΔτV = Diagonal(B.expmΔτV)
    rdiv!(M, expmΔτV) # A⋅exp(+Δτ⋅K)⋅exp(+Δτ⋅V)
    copyto!(A, M)

    return nothing
end

function rdiv!(
    A::AbstractVecOrMat, B::SymChkbrdPropagator;
    M = nothing
)

    expmΔτKo2ᵀ = adjoint(B.expmΔτKo2)
    rdiv!(A, expmΔτKo2ᵀ) # A⋅[exp(+Δτ⋅K/2)]ᵀ
    expmΔτV = Diagonal(B.expmΔτV)
    rdiv!(A, expmΔτV) # A⋅[exp(+Δτ⋅K/2)]ᵀ⋅exp(+Δτ⋅V)
    rdiv!(A, B.expmΔτKo2) # A := A⋅B⁻¹ = A⋅[exp(+Δτ⋅K/2)]ᵀ⋅exp(+Δτ⋅V)⋅exp(+Δτ⋅K/2)
    
    return nothing
end

function rdiv!(
    A::AbstractVecOrMat, B::AsymChkbrdPropagator;
    M = nothing
)

    rdiv!(A, B.expmΔτK) # A⋅exp(+ΔτK)
    expmΔτV = Diagonal(B.expmΔτV)
    rdiv!(A, expmΔτV) # A := A⋅B⁻¹ = [A⋅exp(+ΔτK)]⋅exp(+Δτ⋅V)

    return nothing
end