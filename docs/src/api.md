# API

## Propagator Types

- [`AbstractPropagator`](@ref)
- [`AbstractExactPropagator`](@ref)
- [`AbstractChkbrdPropagator`](@ref)
- [`SymExactPropagator`](@ref)
- [`AsymExactPropagator`](@ref)
- [`SymChkbrdPropagator`](@ref)
- [`AsymChkbrdPropagator`](@ref)
- [`SymPropagators`](@ref)

```@docs
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
- [`fermion_greens_calculator`](@ref)

```@docs
FermionGreensCalculator
fermion_greens_calculator
```

## DQMC Building Block Routines

- [`calculate_equaltime_greens!`](@ref)
- [`propagate_equaltime_greens!`](@ref)
- [`stabilize_equaltime_greens!`](@ref)
- [`calculate_unequaltime_greens!`](@ref)
- [`local_update_det_ratio`](@ref)
- [`local_update_greens!`](@ref)

```@docs
calculate_equaltime_greens!
propagate_equaltime_greens!
stabilize_equaltime_greens!
calculate_unequaltime_greens!
local_update_det_ratio
local_update_greens!
```

## Overloaded Functions

- [`iterate`](@ref)
- [`eltype`](@ref)
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

- [`JDQMCFramework.Continuous`](@ref)
- [`JDQMCFramework.update_factorizations!`](@ref)
- [`JDQMCFramework.update_B̄!`](@ref)
- [`JDQMCFramework.calculate_B̄!`](@ref)
- [`JDQMCFramework.stabilization_interval`](@ref)

```@docs
JDQMCFramework.Continuous
JDQMCFramework.update_factorizations!
JDQMCFramework.update_B̄!
JDQMCFramework.calculate_B̄!
JDQMCFramework.stabilization_interval
```