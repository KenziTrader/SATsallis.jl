## tests for SATsallis.jl
using KernelDensity
using FactCheck

# global min at (1,1), f(1,1)=0
function rosenbrock(x::Vector)
  return (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
end

facts("Box-constrained refinement functions") do
  xnormtol = 1e-7
  context("fminbox local") do
    optres = Optim.fminbox(Optim.DifferentiableFunction(rosenbrock),
                           [0.0, 0.0], -10*ones(2), 10*ones(2))
    println(optres)
    @fact optres.f_converged --> true
    @fact norm(optres.minimum - [1.0, 1.0]) < xnormtol --> true
  end
end

# Tsallis-distributed RNG vs. Fig.1 of Schanze2006
vxrand_qg(q::Real, t::Real, N::Integer) = Float64[
  SATsallis.randtsallis(SATsallis.psrng(q, t)..., 1) for i=1:N ]
kdens_qg(q::Real, t::Real, N::Integer, bnd) =
  kde(vxrand_qg(q, t, N), boundary=bnd)
facts("Tsallis RNG") do
  nsamp = Int64(1e6)
  bnd = (-100.0, 100.0)
  context("q = 1.5") do
    kd = kdens_qg(1.5, 1.0, nsamp, bnd)
    @fact kd.x[indmax(kd.density)] < 0.1 --> true
    @fact 1e-4 < kd.density[indmin(abs(kd.x - 7.5))] < 1e-3 --> true
    @fact kd.density[indmin(abs(kd.x - 25.0))] < 1e-4 --> true
  end
  context("q = 2.0") do
    kd = kdens_qg(2.0, 1.0, nsamp, bnd)
    @fact kd.x[indmax(kd.density)] < 0.1 --> true
    @fact 1e-4 < kd.density[indmin(abs(kd.x - 25.0))] < 1e-3 --> true
  end
end

facts("GSA: Rosenbrock function") do
  xnormtol = 1e-4
  context("TSA defaults, no additional refinement") do
    optres = SATsallis.satopt(rosenbrock, zeros(2),
                              -10*ones(2), 10*ones(2),
                              locrefine = false)
    println(optres)
    @fact optres.f_converged --> false
    @fact norm(optres.minimum - [1.0, 1.0]) < xnormtol --> true

    @fact 0 --> 1
  end
#   context("TSA with CG refinement") do
#     optres = SATsallis.satopt(rosenbrock, zeros(2),
#                               -10*ones(2), 10*ones(2),
#                               locrefine = true,
#                               iterations = itermax)
#     println("With refinement: **********************************")
#     println(optres)
#     @fact Optim.converged(optres) --> true
#     @fact norm(optres.minimum - [1.0, 1.0]) < xnormtol --> true
#   end
end





# sec3 of Tsallis1996 D=4 polynomial from molecular geometry opt.,
#  with degenerate local minima and one global
# function etsallis(x::Vector)
#   return sumabs2(x.^2 - 8.0) + 5.0*sum(x) + 57.3276
# end

# facts("GSA: Tsallis1996 4-dim. polynomial") do
#   itermax = 20
#   context("No accelerated convergence, Tqa=Tqv") do
#     optres = SATsallis.satopt(etsallis, zeros(4),
#                               -10*ones(4), 10*ones(4),
#                               accel_converg = false,
#                               tem_initial = 100.0,
#                               stepspertemp = 1,
#                               locrefine = false,
#                               iterations = itermax,
#                               extended_trace = true,
#                               show_trace = true)
#     println(optres)
#   end
#   @fact 0 --> 1
# end




# Schanze2006 example verification
# Rastrigin function: global min at xi=0 for all i
# function rastrigin3d(x::Vector)
#   return 30.0 + sum(x.^2 - 10.0*cos(2pi*x))
# end
# facts("GSA: Rastrigin function") do
#   xnormtol = 1e-4
#   context("TSA without additional refinement") do
#     optres = SATsallis.satopt(rastrigin3d, -3.0*ones(3),
#                               -5.12*ones(3), 5.12*ones(3),
#                               qa = 1.05, qv = 2.2,
#                               tem_initial = 1.0,
#                               accel_converg = false,
#                               locrefine = false,
#                               iterations = 50,
#                               stepspertemp = 500,
#                               extended_trace = true,
#                               show_trace = true,
#                               store_trace = true)
#     println(optres)
#     @fact optres.f_converged --> false
#     @fact norm(optres.minimum) < xnormtol --> true

#     @fact 0 --> 1
#   end
# end

