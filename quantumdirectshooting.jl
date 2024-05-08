include("dynamics.jl")


function disc_final_time_obj(u, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
    delta, sigmaz, omega, sigmax, psitarget, psi0 = p

    psi = psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax)
    # @show psi[1]'*psitarget
    # @show norm(psi[1]'*psitarget)
    return 1 - fidelity(psi[end], psitarget)
end

optf = OptimizationFunction(disc_final_time_obj, AutoFiniteDiff())
prob = OptimizationProblem(optf, u0, [delta, sigmaz, omega, sigmax, psitarget, psi0] , lb = [-1.0 for i in 1:N], ub = [1.0 for i in 1:N])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 5000)

@show res.objective
@show res.minimizer
psi = psi_dynamics(psi0, res.minimizer, dt, delta, sigmaz, omega, sigmax)
plot(res.u)
# plot!(prob.u0)
fidelityy = [fidelity(x, psitarget) for x in psi]
plot(fidelityy)

# directhess_pattern = deserialize("directhesscons_pattern.jld")
optf = OptimizationFunction(disc_final_time_obj, AutoForwardDiff())
prob = OptimizationProblem(optf, u0, [delta, sigmaz, omega, sigmax, psitarget, psi0] , lb = [-1.0 for i in 1:N], ub = [1.0 for i in 1:N])
@time res = solve(prob, Ipopt.Optimizer(), maxiters = 5000)

@time res = solve(prob, Optimization.LBFGS(), maxiters = 5000)

optf = Optimization.instantiate_function(optf, u0, AutoForwardDiff(), [delta, sigmaz, omega, sigmax, psitarget, psi0])

@show res.objective
@show res.minimizer
psi = psi_dynamics(psi0, res.minimizer, dt, delta, sigmaz, omega, sigmax)
plot(res.u)
# plot!(prob.u0)
fidelityy = [1 - fidelity(x,psitarget) for x in psi]
plot(fidelityy)

G1 = zeros(N)
optf.grad(G1, u0)

H1 = zeros(N,N)
optf.hess(H1, u0)

G2 = Vector(undef, length(prob.u0))
gradient_disc_final_time_obj_mayer(G2, u0)
println(G2)

function gradient_disc_final_time_obj_mayer(G, u, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
    delta, sigmaz, omega, sigmax, psitarget, psi0 = p

    psi = psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax)
    lambda = lambda_dynamics(psitarget, u0, dt, delta, sigmaz, omega, sigmax)

    h1 = (omega*sigmax)/2
    overlap = psi[end]'*psitarget
    for i in 1:N
        G[i] = -2.0*real(-im*dt*lambda[i]'*h1*psi[i]*overlap)
    end
end

optf = OptimizationFunction(disc_final_time_obj, grad = gradient_disc_final_time_obj_mayer)
prob = OptimizationProblem(optf, u0, [delta, sigmaz, omega, sigmax, psitarget, psi0] , lb = [-1.0 for i in 1:N], ub = [1.0 for i in 1:N])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 10000, reltol = 1e-10)

@show res.objective
@show res.minimizer
psi = psi_dynamics(psi0, res.u, dt, delta, sigmaz, omega, sigmax)

plot(res.u)

fidelityy = [1 - fidelity(x,psitarget) for x in psi]
plot(fidelityy)

savefig("traj.png")
plot(res.u)
# plot!(prob.u0)
fidelity = [1 - fidelityy(x,psitarget) for x in psi]
plot(fidelity)

# function disc_stabilization_obj(u, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
#     delta, sigmaz, omega, sigmax, psitarget, psi0 = p
#     psi = psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax)
#     return sum(fidelity.(psi, Ref(psitarget)))
# end


# optf = OptimizationFunction(disc_stabilization_obj, AutoForwardDiff())
# prob = OptimizationProblem(optf, u0, [delta, sigmaz, omega, sigmax, psitarget, psi0] , lb = [-1.0 for i in 1:N], ub = [1.0 for i in 1:N])
# @time res = solve(prob, Optimization.LBFGS(), maxiters = 5000)

# @show res.objective
# @show res.minimizer
# psi = psi_dynamics(psi0, res.minimizer, dt, delta, sigmaz, omega, sigmax)
# plot(res.u)
# # plot!(prob.u0)
# fidelitys = [(1- norm(x'*psitarget)^2) for x in psi]
# plot(fidelitys)


# optf = Optimization.instantiate_function(optf, u0, AutoForwardDiff(), [delta, sigmaz, omega, sigmax, psitarget, psi0])

# G3 = Vector(undef, length(prob.u0))
# optf.grad(G3, u0)

# function gradient_disc_stabilization_obj(G, u, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
#     delta, sigmaz, omega, sigmax, psitarget, psi0 = p
#     psi = psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax)
#     lambda = lambda_dynamics(psitarget, u0, dt, delta, sigmaz, omega, sigmax)

#     h1 = (omega*sigmax)/2
#     overlap = [psi[i]'*psitarget for i in 1:length(psi)]

#     for i in 1:N
#         G[i] = -2.0*sum(real(-im*dt*lambda[j]'*h1*psi[j]*overlap[j]) for j in i:N)
#     end
# end

# function gradient_disc_stabilization_obj(G, u, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
#     delta, sigmaz, omega, sigmax, psitarget, psi0 = p
#     psi = psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax)
#     h1 = (omega*sigmax)/2
#     for i in 1:length(u)
#         overlap = sum(2 .* norm.(adjoint.(psi) .* Ref(psitarget)).^2 .* (psitarget' .* psi))
#         G[i] = real(-im*dt*overlap*lambda_dynamics(psitarget, u, dt, delta, sigmaz, omega, sigmax)[i]'*h1*psi[i])
#     end
#     return G
# end

# G4 = Vector(undef, length(prob.u0))
# gradient_disc_stabilization_obj(G4, u0)

# psi_derivative(i, j, psi_prev, u, dt, delta, sigmaz, omega, sigmax) = begin
#     h0 = (delta*sigmaz)/2
#     h1 = (omega*sigmax)/2
#     if i == j
#         -im * dt * (h0 + h1*u[j]) * psi_prev
#     else
#         0
#     end
# end

# function gradient_disc_stabilization_obj(u, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
#     delta, sigmaz, omega, sigmax, psitarget, psi0 = p
#     psi = psi_dynamics(psi0, u, dt, delta, sigmaz, omega, sigmax)
#     G = zeros(length(u))
#     for j in 1:length(u)
#         psi_prev = psi0
#         for i in 1:length(u)
#             overlap_psi_psitarget = psi[i]' * psitarget
#             dpsi = psi_derivative(i, j, psi_prev, u, dt, delta, sigmaz, omega, sigmax)
#             G[j] += 2 * real(overlap_psi_psitarget * conj(dpsi' * psitarget))
#             psi_prev = psi[i]
#         end
#     end
#     return G
# end

# gradient_disc_stabilization_obj(u0)

