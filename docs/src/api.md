# API

## Propagator Types

- [`JDQMCFramework.Continuous`](@ref)
- [`AbstractPropagator`](@ref)
- [`AbstractExactPropagator`](@ref)
- [`AbstractChkbrdPropagator`](@ref)
- [`SymExactPropagator`](@ref)
- [`AsymExactPropagator`](@ref)
- [`SymChkbrdPropagator`](@ref)
- [`AsymChkbrdPropagator`](@ref)
- [`SymPropagators`](@ref)

```@docs
JDQMCFramework.Continuous
AbstractPropagator
AbstractExactPropagator
AbstractChkbrdPropagator
SymExactPropagator
AsymExactPropagator
SymChkbrdPropagator
AsymChkbrdPropagator
SymPropagators
```

## FermionGreensCalculator Type

- [`FermionGreensCalculator`](@ref)

```@docs
FermionGreensCalculator
FermionGreensCalculator(::AbstractVector{P}, ::E, ::E, ::Int) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T}}
FermionGreensCalculator(::FermionGreensCalculator{T,E}) where {T,E}
```

## DQMC Building Block Routines

- [`calculate_equaltime_greens!`](@ref)
- [`propagate_equaltime_greens!`](@ref)
- [`stabilize_equaltime_greens!`](@ref)
- [`initialize_unequaltime_greens!`](@ref)
- [`propagate_unequaltime_greens!`](@ref)
- [`stabilize_unequaltime_greens!`](@ref)
- [`local_update_det_ratio`](@ref)
- [`local_update_greens!`](@ref)
- [`partially_wrap_greens_reverse!`](@ref)
- [`partially_wrap_greens_forward!`](@ref)

```@docs
calculate_equaltime_greens!
propagate_equaltime_greens!
stabilize_equaltime_greens!
initialize_unequaltime_greens!
propagate_unequaltime_greens!
stabilize_unequaltime_greens!
local_update_det_ratio
local_update_greens!
partially_wrap_greens_reverse!
partially_wrap_greens_forward!
```

## Overloaded Functions

- [`iterate`](@ref)
- [`eltype`](@ref)
- [`resize!`](@ref)
- [`size`](@ref)
- [`copyto!`](@ref)
- [`ishermitian`](@ref)
- [`mul!`](@ref)
- [`lmul!`](@ref)
- [`rmul!`](@ref)
- [`ldiv!`](@ref)
- [`rdiv!`](@ref)

```@docs
Base.iterate
Base.eltype
Base.resize!
Base.size
Base.copyto!
LinearAlgebra.ishermitian
LinearAlgebra.mul!
LinearAlgebra.lmul!
LinearAlgebra.rmul!
LinearAlgebra.ldiv!
LinearAlgebra.rdiv!
```

## Utility Functions

- [`eval_length_imaginary_axis`](@ref)
- [`exp!`](@ref)
- [`build_hopping_matrix!`](@ref)

```@docs
eval_length_imaginary_axis
exp!
build_hopping_matrix!
```

## Developer API

- [`JDQMCFramework.update_factorizations!`](@ref)
- [`JDQMCFramework.update_B̄!`](@ref)
- [`JDQMCFramework.calculate_B̄!`](@ref)
- [`JDQMCFramework.stabilization_interval`](@ref)

```@docs
JDQMCFramework.update_factorizations!
JDQMCFramework.update_B̄!
JDQMCFramework.calculate_B̄!
JDQMCFramework.stabilization_interval
```