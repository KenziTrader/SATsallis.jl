## Tsallis simulated annealing with lower and upper bounds (box) constraints
## @TODO explore better options for constraint handling
##
## Multi-disciplinary Insights LLC
## MIT license
## We encourage everyone to build upon this and let us know about their progress.

module SATsallis

using Optim
using Distributions

export satopt

# optimization trace
macro satrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["x_proposal"] = copy(x_proposal)
            end
            grnorm = NaN
            Optim.update!(tr, iteration, f_x, grnorm, dt,
                          store_trace, show_trace)
        end
    end
end

# adjusted version of Optim.assess_convergence
function fminreltol(f::Real, f0::Real, ftol::Real)
  return abs(f - f0)/(abs(f0) + ftol) < ftol || nextfloat(f) >= f0
end
function sa_convergence(f_x::Real, f_x_previous::Real,
                        f_x_best::Real, ftol::Real)
  return fminreltol(f_x, f_x_previous, ftol) &&
    fminreltol(f_x, f_x_best, ftol)
end

# sampling from Tsallis (q-Gaussian) D-dim. distribution
#  implementation from Schanze2006
function psrng(q::Real, temp::Real)
  p = (3.0 - q) / (2.0 * (q - 1.0))
  s = sqrt(2.0 * (q - 1.0)) / temp^(1.0 / (3.0 - q))
  return p, s
end
function randtsallis(p::Real, s::Real, d::Integer)
  d > 0 || error("Dimension should be 1 or more")
  if d > 1
    return randn(d)/( s * sqrt(Distributions.rand(Distributions.Gamma(p))) )
  else
    return randn()/( s * sqrt(Distributions.rand(Distributions.Gamma(p))) )
  end
end

# trial move from D-dim. Tsallis visiting distribution,
# folded into the bounding box if exceeding the box
function sajump!(x::Array, x_proposal::Array,
                 xmin::Array, xbox::Array, p::Real, s::Real)
  @assert size(x) == size(x_proposal) == size(xmin) == size(xbox)
  dxtry = randtsallis(p, s, length(x))
  for i in 1:length(x)
    @inbounds ( x_proposal[i] = xmin[i] +
                 rem(xbox[i] + rem(x[i] + dxtry[i] - xmin[i], xbox[i]),
                     xbox[i]) )
  end
  return
end

# Tsallis1996 eq(14): T(t) = T(1)*tqvscal
tqvscal(q::Real, niter::Integer) = (2^(q - 1.0) - 1.0) /
  ( (1 + niter)^(q - 1.0) - 1.0 )

# Tsallis1996 eq(5) acceptance probability
function pqa(q::Real, tem::Real, deltf::Real)
  paa = 1.0 + (q - 1.0) * deltf / tem
  if paa <= 0.0
    return 0.0
  else
    return exp( log(paa) / (1.0 - q) )
  end
end

# Tsallis GSA with added box constraints
function satopt{T}(fobj::Function, initial_x::Array{T},
                   bound_x_lower::Array{T},
                   bound_x_upper::Array{T};
                   iterations::Integer = 10_000,
                   iter_min::Integer = 100,
                   ftol::Real = 1e-8,
                   f_converge::Real = -Inf,
                   tem_initial::Real = 1e3,
                   tem_restart::Real = 1e-5,
                   fscal::Real = 1.0, # help ensure fobj-roughness < T0
                   qv::Real = 2.7, # Tsallis1996
                   qa::Real = -5.0, # Tsallis1996
                   accel_converg::Bool = false,
                   stepspertemp::Integer = 100,
                   locrefine::Bool = true, # constr. conj. grad. refinement
                   store_trace::Bool = false,
                   show_trace::Bool = false,
                   extended_trace::Bool = false,
                   trace_all_moves = false)
  # settings and checks
  #2.0 < qv < 3.0 || error("qv must be between 2 and 3 in SATsallis")
  #qa < 0.0 || error("qa must be negative in SATsallis")



  @assert size(initial_x) == size(bound_x_lower) == size(bound_x_upper)
  @assert all(isfinite(bound_x_lower)) && all(isfinite(bound_x_upper))
  @assert all(bound_x_lower .< initial_x .< bound_x_upper)
  xbox = bound_x_upper - bound_x_lower
  if extended_trace
    show_trace = true
  end
  show_trace && @printf "Iter Function value Gradient norm \n"

  # hard constraints on objective function for SA
  f(x) = all(bound_x_lower .<= x .<= bound_x_upper) ? fscal*fobj(x) : Inf

  # objective for local box minimization using finite-diff. gradients
  fdf = Optim.DifferentiableFunction(fobj)

  # Maintain current and proposed state
  x, x_proposal = copy(initial_x), copy(initial_x)

  # Record the number of iterations we perform
  iteration = 0
  # Track the number of function calls:
  #  may grow faster if more than one MC step per cooling iteration
  #    (i.e. if stepspertemp > 1)
  f_calls = 0

  # Store f(x) in f_x
  f_x = f(x); f_calls += 1
  f_proposal = f_x

  # Store the best state ever visited
  best_x = copy(x); best_f_x = f_x

  # Trace the history of states visited
  tr = Optim.OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @satrace

  # initialize control parameters
  temqv = temqa = tem_initial
  prng, srng = psrng(qv, temqv)

  # SA loop
  saiter = 1; converged = stepconv = false
  while !converged && iteration < iterations
    # Increment the total number of steps
    iteration += 1

    # run stepspertemp steps at the current temperature
    for j in 1:stepspertemp
      # try a move: new x_proposal
      sajump!(x, x_proposal, bound_x_lower, xbox, prng, srng)
      f_proposal = f(x_proposal); f_calls += 1



      # should be within constraints
#       inbox = false; nrngtries = 0
#       while !inbox && nrngtries < 10
#         nrngtries += 1
#         x_proposal = x + randtsallis(prng, srng, length(x))
#         f_proposal = f(x_proposal)
#         if isfinite(f_proposal)
#           inbox = true
#           # all attempts before (outside box)
#           #  returned Inf without calling fobj: only last one called it
#           f_calls += 1
#         end
#       end
#       if !inbox
#         # still could not generate jump inside the box

#         println("using rand")

#         x_proposal = rand(length(x)) .* (bound_x_upper - bound_x_lower)
#         f_proposal = f(x_proposal); f_calls += 1
#       end

      # part of convergence criterion
      #  (before f_x may be overwritten if a move is accepted below)
      stepconv = sa_convergence(f_proposal, f_x, best_f_x, ftol)

      # Metropolis part
      if f_proposal <= f_x
        # accept the move
        copy!(x, x_proposal); f_x = f_proposal
        # if the best state yet, keep a record of it
        if f_proposal < best_f_x
          best_f_x = f_proposal
          copy!(best_x, x_proposal)
        end
      else
        # use acceptance probability p
        if rand() <= pqa(qa, temqa, f_proposal - f_x)
          # accept
          copy!(x, x_proposal); f_x = f_proposal
        end
      end
      if trace_all_moves
        @satrace
      end
    end # stepspertemp-loop
    if !trace_all_moves
      @satrace
    end

    # new temperature: step (iv) in Tsallis1996
    saiter += 1 # next t in Tsallis1996: started with 1 at tem_initial
    temqv = tem_initial * tqvscal(qv, saiter)

    # accel_converg chooses accept. temp. between Tsallis1996 and Xiang2013
    temqa = accel_converg ? temqv/float(saiter) : temqv

    # process control
    if f_x < f_converge && saiter > iter_min
      converged = stepconv
    end
    if !converged && saiter > iter_min
      # restart annealing if temqv got too low
      if temqv < tem_restart
        # heat up
        temqv = temqa = tem_initial
        saiter = 1

        println("Reheat")

      end
    end

    # prepare randtsallis parameters for the current temqv
    prng, srng = psrng(qv, temqv)
  end # SA cooling loop

  # solution refinement: box-constrained CG
  if locrefine
    optres = Optim.fminbox(fdf, best_x, bound_x_lower, bound_x_upper, ftol=ftol)
    optres.method = "Tsallis SA with refinement"
    optres.initial_x = initial_x
    optres.iterations = iterations
    optres.iteration_converged = (iteration == iterations)
    optres.trace = tr
    optres.f_calls += f_calls
  else
    optres = Optim.MultivariateOptimizationResults("Tsallis SA",
                                                   initial_x, best_x,
                                                   Float64(best_f_x),
                                                   iterations,
                                                   iteration == iterations,
                                                   false, NaN,
                                                   converged, ftol,
                                                   false, NaN,
                                                   tr, f_calls, 0)
  end
  return optres
end

end # module
