using JDQMCFramework
using Test
using LinearAlgebra
using LatticeUtilities

# fermi function
function fermi(ϵ, β)

    return (1-tanh(β/2*ϵ))/2
end

# calculate exact analytic value for the retarded imaginary Green's function
function retarded_greens(τ,β,ϵ,U)
    
    gτ = similar(ϵ)
    @. gτ = inv(exp(τ*ϵ) + exp((τ-β)*ϵ))
    Gτ = U * Diagonal(gτ) * adjoint(U)
    logdetGτ, sgndetGτ = logabsdet(Diagonal(gτ))
    
    return Gτ, logdetGτ, sgndetGτ
end

# calculate exact analytic value for the advanced imaginary Green's function
function advanced_greens(τ,β,ϵ,U)
    
    gτ = similar(ϵ)
    @. gτ = -inv(exp(-τ*ϵ) + exp(-(τ-β)*ϵ))
    Gτ = U * Diagonal(gτ) * adjoint(U)
    logdetGτ, sgndetGτ = logabsdet(Diagonal(gτ))
    
    return Gτ, logdetGτ, sgndetGτ
end

@testset "JDQMCFramework.jl" begin
    
    # model parameters
    t = 1.0 # hopping amplitude
    μ = 0.0 # chemical potential
    L = 4 # lattice size
    β = 3.7 # inverse temperature
    Δτ = 0.1 # discretization in imaginary time
    n_stab = 10 # frequency of numerical stabilization

    # calculate length of imagninary time axis
    Lτ = eval_length_imaginary_axis(β,Δτ)

    # construct neighbor table for square lattice
    unit_cell = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])
    lattice = Lattice(L = [L,L], periodic = [true,true])
    bond_x = Bond(orbitals = (1,1), displacement = [1,0])
    bond_y = Bond(orbitals = (1,1), displacement = [0,1])
    neighbor_table = build_neighbor_table([bond_x, bond_y], unit_cell, lattice)

    # calculate number of sites in lattice
    N = nsites(unit_cell, lattice)

    # calculate number of bonds in lattice
    Nbonds = size(neighbor_table, 2)

    # build hopping matrix
    K = zeros(typeof(t), N, N)
    build_hopping_matrix!(K, neighbor_table, fill(t, Nbonds))

    # build diagonal on-site energy matrix
    V = fill(-μ, N)

    # construct Hamiltonian matrix
    H = K + Diagonal(V)
    ϵ, U = eigen(H)

    # calculate exact unequal time Green's function
    G_ret = zeros(typeof(t), N, N, Lτ+1)
    G_adv = zeros(typeof(t), N, N, Lτ+1)
    for l in 0:Lτ
        G_ret[:,:,l+1] = retarded_greens(Δτ*l, β, ϵ, U)[1]
        G_adv[:,:,l+1] = advanced_greens(Δτ*l, β, ϵ, U)[1]
    end

    # calculate equal-time Green's function matrix
    G, logdetG, sgndetG = retarded_greens(0, β, ϵ, U)

    # define vector of propagators
    expmΔτV = exp.(-Δτ*V)
    expmΔτK = exp(-Δτ*K)
    exppΔτK = exp(+Δτ*K)
    Bup = AsymExactPropagator{eltype(expmΔτK),eltype(expmΔτV)}[]; # spin up propagators
    Bdn = AsymExactPropagator{eltype(expmΔτK),eltype(expmΔτV)}[]; # spin down propagators
    for l in 1:Lτ
        B_l = AsymExactPropagator(expmΔτV, expmΔτK, exppΔτK)
        push!(Bup, B_l)
        push!(Bdn, B_l)
    end

    @test eltype(Bup[1]) == eltype(expmΔτK)

    # initiate FermionGreensCalculator struct
    fgc_up = FermionGreensCalculator(Bup, β, Δτ, n_stab+1)
    fgc_dn = FermionGreensCalculator(Bdn, β, Δτ, n_stab+1)

    # calculate equal-time Green's function
    Gup = zeros(typeof(t), N, N)
    Gdn = zeros(typeof(t), N, N)
    logdetGup, sgndetGup = calculate_equaltime_greens!(Gup, fgc_up)
    logdetGdn, sgndetGdn = calculate_equaltime_greens!(Gdn, fgc_dn)

    # test that spin up equal-time Green's function is correct
    @test G ≈ Gup
    @test logdetG ≈ logdetGup
    @test sgndetG ≈ sgndetGup

    # test that spin down equal-time Green's function is correct
    @test G ≈ Gdn
    @test logdetG ≈ logdetGdn
    @test sgndetG ≈ sgndetGdn

    # initialize copy FermionGreensCalculator
    fgc = FermionGreensCalculator(fgc_up)
    @test typeof(fgc) <: FermionGreensCalculator

    # copy FermionGreensCalculator
    copyto!(fgc, fgc_dn)
    @test typeof(fgc) <: FermionGreensCalculator

    # resize FermionGreensCalculator struct
    logdetGup, sgndetGup = resize!(fgc_up, Gup, logdetGup, sgndetGup, Bup, n_stab)
    logdetGdn, sgnderGdn = resize!(fgc_dn, Gdn, logdetGdn, sgndetGdn, Bdn, n_stab)
    @test fgc_up.n_stab == n_stab
    @test fgc_dn.n_stab == n_stab
    @test logdetG ≈ logdetGup
    @test sgndetG ≈ sgndetGup
    @test logdetG ≈ logdetGdn
    @test sgndetG ≈ sgndetGdn

    # copy FermionGreensCalculator with resizing
    copyto!(fgc, fgc_up)
    @test fgc.n_stab == n_stab

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_equaltime_greens!(Gup, fgc_up, Bup)
        propagate_equaltime_greens!(Gdn, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # test that spin up equal-time Green's function is correct
        @test G ≈ Gup
        @test logdetG ≈ logdetGup
        @test sgndetG ≈ sgndetGup

        # test that spin down equal-time Green's function is correct
        @test G ≈ Gdn
        @test logdetG ≈ logdetGdn
        @test sgndetG ≈ sgndetGdn

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B_barₙ matrices.
        logdetGup, sgndetGup, δGup, δθup = stabilize_equaltime_greens!(Gup, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=true)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_equaltime_greens!(Gdn, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=true)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_equaltime_greens!(Gup, fgc_up, Bup)
        propagate_equaltime_greens!(Gdn, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # test that spin up equal-time Green's function is correct
        @test G ≈ Gup
        @test logdetG ≈ logdetGup
        @test sgndetG ≈ sgndetGup

        # test that spin down equal-time Green's function is correct
        @test G ≈ Gdn
        @test logdetG ≈ logdetGdn
        @test sgndetG ≈ sgndetGdn

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B_barₙ matrices.
        logdetGup, sgndetGup, δGup, δθup = stabilize_equaltime_greens!(Gup, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=false)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_equaltime_greens!(Gdn, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=false)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end

    # initialize unequal-time Green's functions
    Gup_τ0 = similar(Gdn)
    Gup_0τ = similar(Gdn)
    Gup_ττ = similar(Gdn)
    Gdn_τ0 = similar(Gdn)
    Gdn_0τ = similar(Gdn)
    Gdn_ττ = similar(Gdn)
    initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup)
    initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn)

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fgc_up, Bup)
        propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # test that spin up unequal-time Green's function is correct
        G_τ0 = @view G_ret[:,:,l+1]
        G_0τ = @view G_adv[:,:,l+1]
        @test G ≈ Gup_ττ
        @test G_τ0 ≈ Gup_τ0
        @test G_0τ ≈ Gup_0τ
        @test G ≈ Gdn_ττ
        @test G_τ0 ≈ Gdn_τ0
        @test G_0τ ≈ Gdn_0τ

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B_barₙ matrices.
        logdetGup, sgndetGup, δGup, δθup = stabilize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=false)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=false)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end

    # initialize unequal-time Green's functions
    initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup)
    initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn)

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fgc_up, Bup)
        propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # test that spin up unequal-time Green's function is correct
        G_τ0 = @view G_ret[:,:,l+1]
        G_0τ = @view G_adv[:,:,l+1]
        @test G ≈ Gup_ττ
        @test G_τ0 ≈ Gup_τ0
        @test G_0τ ≈ Gup_0τ
        @test G ≈ Gdn_ττ
        @test G_τ0 ≈ Gdn_τ0
        @test G_0τ ≈ Gdn_0τ

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B_barₙ matrices.
        logdetGup, sgndetGup, δGup, δθup = stabilize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=true)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=true)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end


    # initialize unequal-time Green's functions
    Gup_00 = Gup
    Gdn_00 = Gdn
    initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup_00)
    initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00)

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fgc_up, Bup)
        propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # test that spin up unequal-time Green's function is correct
        G_τ0 = @view G_ret[:,:,l+1]
        G_0τ = @view G_adv[:,:,l+1]
        @test G ≈ Gup_ττ
        @test G ≈ Gup_00
        @test G_τ0 ≈ Gup_τ0
        @test G_0τ ≈ Gup_0τ
        @test G ≈ Gdn_ττ
        @test G ≈ Gdn_00
        @test G_τ0 ≈ Gdn_τ0
        @test G_0τ ≈ Gdn_0τ

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B_barₙ matrices.
        logdetGup, sgndetGup, δGup, δθup = stabilize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup_00, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=false)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=false)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end

    # initialize unequal-time Green's functions
    initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup_00)
    initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00)

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fgc_up, Bup)
        propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # test that spin up unequal-time Green's function is correct
        G_τ0 = @view G_ret[:,:,l+1]
        G_0τ = @view G_adv[:,:,l+1]
        @test G ≈ Gup_ττ
        @test G ≈ Gup_00
        @test G_τ0 ≈ Gup_τ0
        @test G_0τ ≈ Gup_0τ
        @test G ≈ Gdn_ττ
        @test G ≈ Gdn_00
        @test G_τ0 ≈ Gdn_τ0
        @test G_0τ ≈ Gdn_0τ

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B_barₙ matrices.
        logdetGup, sgndetGup, δGup, δθup = stabilize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup_00, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=false)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=false)

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end
end
