include("dynamics.jl")

using SparseArrays, ThreadsX

u0 = [rand([-1.0,1.0]) for i in 1:N]

function make_complex_state(s::Vector{T}) where {T}
    return [s[1] + im*s[2], s[3] + im*s[4]]
end

iter = 0
function disc_multi_shoot_obj(u_psi, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
    global iter
    u = u_psi[1:N]

    delta, sigmaz, omega, sigmax, psitarget = p
    h0 = (delta*sigmaz)/2
    h1 = (omega*sigmax)/2

    psiprev = make_complex_state(u_psi[end-3:end])
    psiprev = psiprev ./ norm(psiprev) # Normalize the initial state
    # @show psiprev
    psii = psi_dynamics(psiprev, u[end-4:end], dt, delta, sigmaz, omega, sigmax, 5)[end]
    return 1 - fidelity(psii, psitarget)
end

function multi_shoot_cons(cons, u_psi, p = [delta, sigmaz, omega, sigmax, psitarget, psi0])
    u = u_psi[1:N]
    psi1 = make_complex_state(u_psi[N+1:N+4])
    psi = Vector{typeof(psi1)}(undef, Int(N/5))
    j = 1
    for i in N+1:4:length(u_psi)
        psi[j] = make_complex_state(u_psi[i:i+3])
        j += 1
    end

    delta, sigmaz, omega, sigmax, psitarget, psi0 = p
    h0 = (delta*sigmaz)/2
    h1 = (omega*sigmax)/2

    psiii = psi0
    # expmethod = ExponentialUtilities.ExpMethodHigham2005()
    # expcache = ExponentialUtilities.alloc_mem(-im * (h0 + (h1*u[1]))*dt, expmethod)
    # psi1 = exponential!(-im * (h0 + (h1*u[1]))*dt, expmethod, expcache) * psiii
    psisim = Vector{typeof(psi1)}(undef, Int(N/5))
    
    # j = 1
    # for i in 1:N-2
    #     psisim[j] = exponential!(-im * (h0 + (h1*u[i]))*dt, expmethod, expcache) * psiii
    #     psiii = psi[i] ./ norm(psi[i]) # Normalize the state from the optimization variables vector
    # end

    for i in 1:Int(N/5)
        psisim[i] = psi_dynamics(psiii, u[i*5-4:i*5], dt, delta, sigmaz, omega, sigmax, 5)[end]
        psiii = psi[i] ./ norm(psi[i]) # Normalize the state from the optimization variables vector
    end


    j = 1
    for i in 1:4:4*Int(N/5)
        cons[i] = (real(psi[j][1]) - real(psisim[j][1]))^2
        cons[i+1] = (imag(psi[j][1]) - imag(psisim[j][1]))^2
        cons[i+2] = (real(psi[j][2]) - real(psisim[j][2]))^2
        cons[i+3] = (imag(psi[j][2]) - imag(psisim[j][2]))^2
        j += 1
    end
    return cons
end

function make_norm_states()
    x = rand(2)
    x = x ./norm(x)
    y = rand(2)
    y = y ./norm(y)
    return [x[1], x[2], y[1], y[2]]
end

# function decompose2real(cv::Vector)
#     return [real(cv[1]), imag(cv[1]), real(cv[2]), imag(cv[2])]
# end

# u0 = vcat(u0, [decompose2real(psi[i]) .+ (randn(4).*0.01) for i in 1:100:500]...)

u0 = vcat(u0, [make_norm_states() for i in 1:Int(N/5)]...)

jacpattern = deserialize("jacpattern.jld")
cons_hess_pattern = deserialize("hesscons_pattern.jld")
hessobj_pattern = deserialize("hessobj_pattern.jld")
optf = OptimizationFunction(disc_multi_shoot_obj, AutoSparseForwardDiff(), cons = multi_shoot_cons, hess_prototype = hessobj_pattern, cons_jac_prototype = jacpattern, cons_hess_prototype = cons_hess_pattern)
prob = OptimizationProblem(optf, u0, [delta, sigmaz, omega, sigmax, psitarget, psi0] , lb = [-1.0 for i in 1:length(u0)], ub = [1.0 for i in 1:length(u0)], lcons = [0.0 for i in 1:4*Int(N/5)], ucons = [0.0 for i in 1:4*Int(N/5)])

@time res = solve(prob, Ipopt.Optimizer(), maxiters = 500)

@time res1 = solve(prob, Optimization.LBFGS(), maxiters = 5000)

c = zeros(4*Int(N/5))
@time multi_shoot_cons(c, u0)

optf = Optimization.instantiate_function(optf, u0, AutoSparseForwardDiff(), [delta, sigmaz, omega, sigmax, psitarget, psi0], 4*(N-2))

ch = [Float64.(chh) for chh in cons_hess_pattern]
@time optf.cons_h(ch, u0)

@show res1.objective
@show res1.minimizer
plot(res1.u[1:N])
# plot!(prob.u0)
psi = [make_complex_state(res1.u[i:i+3]) for i in N+1:4:length(res1.u)]
fidelityy = [1- fidelity(x,psitarget) for x in psi]
plot(fidelityy)

psi = vcat(psi, psi_dynamics(psi[end], res1.u[496:500], dt, delta, sigmaz, omega, sigmax, 5))

optf = Optimization.instantiate_function(optf, u0, AutoForwardDiff(), [delta, sigmaz, omega, sigmax, psitarget, psi0])

psiplot = Array(hcat([norm.(p) for p in psi]...)')
plot(psiplot)

G3 = Vector(undef, length(prob.u0))
optf.grad(G3, u0)

multi_shoot_cons(zeros(N-1), u0)

H = zeros(N, N)
optf.cons_h(H, u0)

