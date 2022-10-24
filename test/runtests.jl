using JDQMCFramework
using Test
using LinearAlgebra
using LatticeUtilities

# calculate exact analytic value for Green's function
function greens(τ,β,ϵ,U)
    
    gτ = similar(ϵ)
    @. gτ = exp(-τ*ϵ)/(1+exp(-β*ϵ))
    Gτ = U * Diagonal(gτ) * adjoint(U)
    logdetGτ, sgndetGτ = logabsdet(Diagonal(gτ))
    
    return Gτ, logdetGτ, sgndetGτ
end

@testset "JDQMCFramework.jl" begin
    
    # model parameters
    t = 1.0; # hopping amplitude
    μ = 0.0; # chemical potential
    L = 4; # lattice size
    β = 3.7; # inverse temperature
    Δτ = 0.1; # discretization in imaginary time
    nₛ = 10; # frequency of numerical stabilization

    # calculate length of imagninary time axis
    Lτ = eval_length_imaginary_axis(β,Δτ)

    # construct neighbor table for square lattice
    unit_cell = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])
    lattice = Lattice(L = [L,L], periodic = [true,true])
    bond_x = Bond(orbitals = (1,1), displacement = [1,0])
    bond_y = Bond(orbitals = (1,1), displacement = [0,1])
    neighbor_table = build_neighbor_table([bond_x, bond_y], unit_cell, lattice)

    # calculate number of sites in lattice
    N = get_num_sites(unit_cell, lattice)

    # calculate number of bonds in lattice
    Nbonds = size(neighbor_table, 2)

    # build hopping matrix
    K = zeros(N,N)
    build_hopping_matrix!(K, neighbor_table, fill(t, Nbonds))

    # build diagonal on-site energy matrix
    V = fill(-μ, N)

    # construct Hamiltonian matrix
    H = K + Diagonal(V)
    ϵ, U = eigen(H)

    # calculate equal-time Green's function matrix
    G, logdetG, sgndetG = greens(0, β, ϵ, U)

    # define vector of propagators
    expmΔτV = exp.(-Δτ*V)
    expmΔτK = exp(-Δτ*K)
    exppΔτK = exp(+Δτ*K)
    Bup = AsymExactPropagator{eltype(expmΔτK),eltype(expmΔτV)}[]; # spin up propagators
    Bdn = AsymExactPropagator{eltype(expmΔτK),eltype(expmΔτV)}[]; # spin down propagators
    for l in 1:Lτ
        B_l = AsymExactPropagator(expmΔτV, expmΔτK, exppΔτK, false)
        push!(Bup, B_l)
        push!(Bdn, B_l)
    end

    # initiate FermionGreensCalculator struct
    fgc_up = fermion_greens_calculator(Bup, N, β, Δτ, nₛ)
    fgc_dn = fermion_greens_calculator(Bdn, N, β, Δτ, nₛ)

    # calculate equal-time Green's function
    Gup = zeros(N,N)
    Gdn = zeros(N,N)
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


    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_equaltime_greens!(Gup, fgc_up, Bup)
        propagate_equaltime_greens!(Gdn, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B̄ₙ matrices.
        logdetGup, sgndetGup = stabilize_equaltime_greens!(Gup, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=true)
        logdetGdn, sgndetGdn = stabilize_equaltime_greens!(Gdn, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=true)

        # test that spin up equal-time Green's function is correct
        @test G ≈ Gup
        @test logdetG ≈ logdetGup
        @test sgndetG ≈ sgndetGup

        # test that spin down equal-time Green's function is correct
        @test G ≈ Gdn
        @test logdetG ≈ logdetGdn
        @test sgndetG ≈ sgndetGdn

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end

    # Iterate over imaginary time τ=Δτ⋅l.
    @testset for l in fgc_up

        # Propagate Green's function matrix to current imaginary time G(l,l).
        propagate_equaltime_greens!(Gup, fgc_up, Bup)
        propagate_equaltime_greens!(Gdn, fgc_dn, Bdn)

        # LOCAL UPDATES OR EVALUATION OF DERIVATIVE OF FERMIONIC ACTION WOULD GO HERE

        # Periodically re-calculate the Green's function matrix for numerical stability.
        # If not performing updates, but just evaluating the derivative of the action, then
        # set update_B̄=false to avoid wasting cpu time re-computing B̄ₙ matrices.
        logdetGup, sgndetGup = stabilize_equaltime_greens!(Gup, logdetGup, sgndetGup, fgc_up, Bup, update_B̄=true)
        logdetGdn, sgndetGdn = stabilize_equaltime_greens!(Gdn, logdetGdn, sgndetGdn, fgc_dn, Bdn, update_B̄=true)

        # test that spin up equal-time Green's function is correct
        @test G ≈ Gup
        @test logdetG ≈ logdetGup
        @test sgndetG ≈ sgndetGup

        # test that spin down equal-time Green's function is correct
        @test G ≈ Gdn
        @test logdetG ≈ logdetGdn
        @test sgndetG ≈ sgndetGdn

        # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
        iterate(fgc_dn, fgc_up.forward)
    end

    # calculate exact unequal time Green's function
    G_l = zeros(N, N, Lτ+1)
    for l in 1:Lτ+1
        G_l[:,:,l] = greens(Δτ*(l-1), β, ϵ, U)[1]
    end

    # calculate uneqaul time Green's function, and equal time Green's function
    # for all imaginary time slices
    G0τ_up = zeros(N, N, Lτ+1)
    G0τ_dn = zeros(N, N, Lτ+1)
    Gττ_up = zeros(N, N, Lτ+1)
    Gττ_dn = zeros(N, N, Lτ+1)
    calculate_unequaltime_greens!(G0τ_up, Gττ_up, fgc_up, Bup)
    calculate_unequaltime_greens!(G0τ_dn, Gττ_dn, fgc_dn, Bdn)

    # test that spin up unequal time Greens function is correct
    @test G_l ≈ G0τ_up

    # test that spin up equal time Greens functions is correct
    @testset for l in 0:Lτ
        Gup_ττ = @view Gττ_up[:,:,l+1]
        @test Gup_ττ ≈ G
    end

    # test that spin down unequal time Greens function is correct
    @test G_l ≈ G0τ_dn

    # test that spin down equal time Greens functions is correct
    @testset for l in 0:Lτ
        Gdn_ττ = @view Gττ_dn[:,:,l+1]
        @test Gdn_ττ ≈ G
    end
end
