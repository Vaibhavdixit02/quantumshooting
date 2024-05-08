include("dynamics.jl")

using SparseArrays

function make_complex_state(s::Vector)
    return [s[1] + im*s[2], s[3] + im*s[4]]
end

function disc_multi_shoot_obj(u_psi, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
    u = u_psi[1:N]

    delta, sigmaz, omega, sigmax, psitarget = p
    h0 = (delta*sigmaz)/2
    h1 = (omega*sigmax)/2
    psii = exponential!(-im * (h0 + (h1*u[N]))*dt) * make_complex_state(u_psi[end-3:end])
    @show norm(psii)
    return 1 - fidelity(psii, psitarget)
end

function multi_shoot_cons(cons, u_psi, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
    u = u_psi[1:N]
    psi = [make_complex_state(u_psi[i:i+3]) for i in N+1:4:length(u_psi)]
    delta, sigmaz, omega, sigmax, psitarget, psi0 = p
    h0 = (delta*sigmaz)/2
    h1 = (omega*sigmax)/2
    psiii = psi0
    psisim = map(1:N-1) do i
        psii = exponential!(-im * (h0 + (h1*u[i]))*dt) * psiii
        psiii = psi[i]
        return psii
    end
    for i in 1:N-2
        cons[i] = norm(psi[i] - psisim[i])^2 + 100 * (norm(psisim[i]) - 1)^2
    end

    return cons
end

function make_norm_states()
    x = rand(2)
    return x./norm(x)
end

u0 = vcat(u0, [make_norm_states() for _ in 1:2*(N-1)]...)

jacpattern = deserialize("jacpattern.jld")
cons_hess_pattern = deserialize("hesscons_pattern.jld")
hessobj_pattern = sparse(ones(length(u0), length(u0)))
optf = OptimizationFunction(disc_multi_shoot_obj, AutoSparseFiniteDiff(), cons = multi_shoot_cons, cons_jac_prototype = jacpattern, cons_hess_prototype = cons_hess_pattern, hess_prototype = hessobj_pattern)
prob = OptimizationProblem(optf, u0, [delta, sigmaz, omega, sigmax, psitarget, psi0] , lb = [-1.0 for i in 1:length(u0)], ub = [1.0 for i in 1:length(u0)], lcons = [0.0 for i in 1:N-2], ucons = [0.0 for i in 1:N-2])

@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)

@show res.objective
@show res.minimizer
plot(res.u[1:N])
# plot!(prob.u0)
psi = psi = [make_complex_state(res.u[i:i+3]) for i in N+1:4:length(res.u)]
fidelityy = [1 - fidelity(x,psitarget) for x in psi]
plot(fidelityy)

optf = Optimization.instantiate_function(optf, u0, AutoForwardDiff(), [delta, sigmaz, omega, sigmax, psitarget, psi0])

G3 = Vector(undef, length(prob.u0))
optf.grad(G3, u0)

multi_shoot_cons(zeros(N-1), u0)

H = zeros(N, N)
optf.cons_h(H, u0)

