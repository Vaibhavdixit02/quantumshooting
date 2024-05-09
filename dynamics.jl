using Optimization, OptimizationMOI, Ipopt, LinearAlgebra, ForwardDiff, ReverseDiff, ExponentialUtilities
using OptimizationOptimJL, FiniteDiff, Plots, Zygote, SparseDiffTools, Symbolics
using Test, Serialization

dt = 0.01
T = 5.0
N = Int(T/dt)

function psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax, N = N)
    # psi = Vector(undef, Int(T/dt))
    h0 = (delta*sigmaz)/2
    h1 = (omega*sigmax)/2
    # @show h0
    # @show h1
    # @show u
    psiii = psi0

    psi = map(1:N) do i
        psii = exponential!(-im * (h0 + (h1*u[i]))*dt) * psiii
        psiii = psii
        return psii
    end
    return psi
end

function fidelity(state1, state2)
    return norm(state1'*state2)^2
end

delta = 20
sigmaz = [1.0 0.0; 0.0 -1.0]
omega = 2
sigmax = [0.0 1.0; 1.0 0.0]

psi0 = [1.0, 1.0]./sqrt(2)
psitarget = [1.0, 0.0]

u0 = [rand([-1.0,1.0]) for i in 1:N]

function lambda_dynamics(psitarget, u, dt, delta, sigmaz, omega, sigmax)

    h0 = (delta*sigmaz)/2
    h1 = (omega*sigmax)/2

    lambdaii = psitarget

    lambda = map(N:-1:2) do i
        lambdai = exponential!(im * (h0 + h1*u[i])*dt) * lambdaii
        lambdaii = lambdai
        return lambdai
    end

    lambda = push!(reverse(lambda), psitarget)

    # test
    # lambda2 = psitarget
    # t = 5.0
    # for i = N:-1:1
    #     @test lambda2 ≈ lambda[i]
    #     @test lambda2'*lambda2 ≈ 1.0
    #     t -= dt
    #     lambda2 = exp(im * (h0 + h1*u[i])*dt) * lambda2
    # end
    # @test t ≈ 0.0 atol=1e-10 # hardcoded for test

    return lambda
end
