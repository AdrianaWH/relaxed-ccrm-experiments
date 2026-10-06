using LinearAlgebra
using Random
using Printf


const BASE_FILE = joinpath(@__DIR__, "cutting_planes_with_finite.jl")
isfile(BASE_FILE) || error("Base methods file not found: $BASE_FILE")

# original methods
module BallMethods
    include("cutting_planes_with_finite.jl")

    function configure_sequences!(mode::Symbol)
        if mode == :one_over_k
            perturb = (iter::Int) -> 1.0 / iter
            relax = perturb
            back_epsilon = 1.0

        elseif mode == :one_over_sqrt_k
            perturb = (iter::Int) -> 1.0 / sqrt(iter)
            relax = perturb
            back_epsilon = 1.0

        else
            error("SEQUENCE_MODE desconhecido: $mode")
        end

        global approx_epsilon = relax
        global paca_epsilon = perturb
        global prsccrm_eta = (iter::Int) -> 1.0 / (iter) #perturb
        global prsccrm_lambda_plain = (iter::Int) -> 1.0
        global prsccrm_lambda_over = (iter::Int) -> 1.0 + relax(iter)
        global prsccrm_lambda_under = (iter::Int) -> 1.0 - relax(iter)


        return back_epsilon
    end

    #  Q_i = I in this experiment (this is the exact projection of the ball)
    function elipsoid_projection(
        x::AbstractVector,
        y::AbstractVector,
        Q::Matrix{Float64},
        radius::Real,
        env::Gurobi.Env,
    )
        diff = x .- y
        distance = norm(diff)

        return distance <= radius ?
            copy(x) :
            y .+ (radius / distance) .* diff
    end
end

# -------------------------------------------------------------------------
# Experiments configuration
# -------------------------------------------------------------------------

 const NS = UInt[20, 50, 100, 200]               
 const MS = UInt[2, 5, 10, 20, 50, 100]       
    


const MAX_ITER_BALLS = UInt(10000)
const TIME_LIMIT_BALLS = 100.0       
const STOP_RADIUS_EPS = 1e-8
const SEED_BALLS = 0


const V6_INNER_RADIUS = 0.01 #0.01 
const V6_RADIUS_RANGE = (1.0, 100.0) # Minimum must exceed the inner radius.
const V6_DIRECTION_CANDIDATES = 150 # More candidates improve angular spread.
const V6_MAX_ATTEMPTS = 50
const V6_START_GAP = 100.0          # Minimum distance from x0 to every ball.

const SEQUENCE_MODE = :one_over_k
# const SEQUENCE_MODE = :one_over_sqrt_k

const MAX_BACKTRACKING_BALLS = UInt(20)
const BACKTRACK_STOP_BALLS = 1e-6
const BACKTRACK_BETA_BALLS = 0.5
const PACA_STOP = :common


const METHODS_BALLS = [
    #"Alt. Proj.",
     "MV Alt. Proj.",
     "Cimmino",
    #"SCCRM",
    #"OSCCRM",
    #"BOSCCRM",
    #"USCCRM",
    #"BUSCCRM",
    "PACA",
     "PRSCCRM",
     "OPRSCCRM",
     "UPRSCCRM",
]

# -------------------------------------------------------------------------
# Balls construction
# -------------------------------------------------------------------------


function make_ball_instance_inner_spread(n::Int, m::Int)
    rho = V6_INNER_RADIUS
    rmin, rmax = V6_RADIUS_RANGE
    gap = V6_START_GAP
    n >= 2 && m >= 2 || error("Require n >= 2 and m >= 2")
    all(isfinite, (rho, rmin, rmax, gap)) || error("Nonfinite parameter")
    0 < rho < rmin <= rmax && gap > 0 || error("Invalid geometry")
    V6_DIRECTION_CANDIDATES >= 1 && V6_MAX_ATTEMPTS >= 1 ||
        error("Invalid candidate or attempt count")
    rng = MersenneTwister(SEED_BALLS)

    for attempt in 1:V6_MAX_ATTEMPTS
        # Greedily maximize separation from previously selected directions.
        # This is a spreading heuristic, not exact equidistribution.
        # directions = zeros(n, m)
        # for i in 1:m
        #     candidates = randn(rng, n, V6_DIRECTION_CANDIDATES)
        #     for v in eachcol(candidates)
        #         normalize!(v)
        #     end
        #     j = i == 1 ? 1 : argmin(vec(maximum(
        #         directions[:, 1:i-1]' * candidates; dims=1)))
        #     directions[:, i] .= candidates[:, j]
        # end
        # First two directions are exactly opposite.
directions = zeros(n, m)
directions[:, 1] .= normalize(randn(rng, n))
directions[:, 2] .= -directions[:, 1]

# Spread the remaining directions relative to all previous ones.
# For m == 2, this loop is empty.
for i in 3:m
    candidates = randn(rng, n, V6_DIRECTION_CANDIDATES)

    for v in eachcol(candidates)
        normalize!(v)
    end

    # Choose the candidate with the largest minimum angular separation.
    scores = vec(maximum(
        directions[:, 1:i-1]' * candidates;
        dims = 1,
    ))

    j = argmin(scores)
    directions[:, i] .= candidates[:, j]
end

        # Boundary points: p_i = rho*u_i.
        # Cross the origin and stop at y_i = -(k_i-rho)*u_i.
        # Then ||p_i-y_i|| = k_i and B(0,rho) is contained in every ball.
        k = exp.(log(rmin) .+ (log(rmax) - log(rmin)) .* rand(rng, m))
        y = -permutedims(directions) .* reshape(k .- rho, m, 1)
        centers_feasible = count(i -> all(
            sum(abs2, view(y, i, :) .- view(y, j, :)) <= k[j]^2
            for j in 1:m), 1:m)
        centers_feasible == 0 || continue

        x_star = zeros(n)
        slater_value = maximum(sum(abs2, view(y, i, :)) - k[i]^2 for i in 1:m)
        # Reverse triangle inequality ensures distance(x0, C_i) >= gap.
        reach = maximum(norm(view(y, i, :)) + k[i] for i in 1:m)
        x0 = (reach + gap) .* normalize(randn(rng, n))
        initial_min = minimum(sum(abs2, x0 .- view(y, i, :)) - k[i]^2 for i in 1:m)
        slater_value < 0 && initial_min > 0 || error("Numerical geometry check failed")
        Q = zeros(m, n, n)
        for i in 1:m, j in 1:n
            Q[i, j, j] = 1.0
        end
        println("\nConstruction 6 | n=$n, m=$m, inner radius=$rho")
        println("Accepted attempt $attempt; centers in intersection = 0 / $m")
        println("Radius range = ", extrema(k), "; ||x0|| = ", norm(x0))
        return y, Q, k, x_star, x0, slater_value, initial_min
    end
    error("No instance accepted; review radius range or increase V6_MAX_ATTEMPTS")
end
# -------------------------------------------------------------------------
# Calls of the original methods
# -------------------------------------------------------------------------

function call_base_method(
    method::String,
    x0,
    y,
    Q,
    k,
    env,
    epsilon_back,
)
    common = (
        x0, y, Q, k,
        MAX_ITER_BALLS, env,
        STOP_RADIUS_EPS, TIME_LIMIT_BALLS,
    )

    finite = (
        x0, y, Q, k,
        MAX_ITER_BALLS,
        STOP_RADIUS_EPS, TIME_LIMIT_BALLS,
    )

    back = (
        x0, y, Q, k,
        MAX_ITER_BALLS, env, STOP_RADIUS_EPS,
        MAX_BACKTRACKING_BALLS,
        epsilon_back,
        BACKTRACK_STOP_BALLS,
        BACKTRACK_BETA_BALLS,
        TIME_LIMIT_BALLS,
    )

    if method == "Alt. Proj."
        return BallMethods.solve_alt_proj(common...)

    elseif method == "MV Alt. Proj."
        return BallMethods.solve_mv_alt_proj(common...)

    elseif method == "Cimmino"
        return BallMethods.solve_cimmino(common...)

    elseif method == "SCCRM"
        return BallMethods.solve_sc_crm(common...)

    elseif method == "OSCCRM"
        return BallMethods.solve_sc_crm_over(common...)

    elseif method == "BOSCCRM"
        return BallMethods.solve_sc_crm_over_back(back...)

    elseif method == "USCCRM"
        return BallMethods.solve_sc_crm_under(common...)

    elseif method == "BUSCCRM"
        return BallMethods.solve_sc_crm_under_back(back...)

    elseif method == "PACA"
        if PACA_STOP == :common
            return BallMethods.solve_paca2(finite...)
        elseif PACA_STOP == :original
            return BallMethods.solve_paca(finite...)
        else
            error("PACA_STOP desconhecido: $PACA_STOP")
        end

    elseif method == "PRSCCRM"
        return BallMethods.solve_prsccrm(finite...)

    elseif method == "OPRSCCRM"
        return BallMethods.solve_oprsccrm(finite...)

    elseif method == "UPRSCCRM"
        return BallMethods.solve_uprsccrm(finite...)
    end

    error("Método desconhecido: $method")
end

function run_one_method(
    method,
    x0,
    y,
    Q,
    k,
    env,
    epsilon_back,
)
    start = time()

    try
        x, history, elapsed, reported_iter =
            call_base_method(
                method, x0, y, Q, k, env, epsilon_back
            )

        # updates = length(history)

        raw = all(isfinite, x) ?
            BallMethods.max_violation(x, y, Q, k, 0.0) :
            NaN

        shifted = all(isfinite, x) ?
            BallMethods.max_violation(
                x, y, Q, k, STOP_RADIUS_EPS
            ) :
            NaN

        reached =
            method == "PACA" && PACA_STOP == :original ?
            raw <= STOP_RADIUS_EPS :
            shifted <= 0.0

        status = reached ? "OK" :
                 !isfinite(raw) ? "DIVERGED" :
                 elapsed >= TIME_LIMIT_BALLS ? "TL" :
                 "MI"

        @printf(
            "%-14s %10.4f s  reported_iter=%6d  raw=%11.3e  shifted=%11.3e  %s\n",
            method, elapsed, reported_iter, raw, shifted, status,
        )

        return (
            elapsed, reported_iter, raw, shifted,
            status, reported_iter,
        )

    catch err
        elapsed = time() - start

        @printf(
            "%-14s %10.4f s  status=FAILED: %s\n",
            method, elapsed, sprint(showerror, err),
        )

        return (elapsed, 0, NaN, NaN, "FAILED", 0)
    end
end

function main_balls()
    epsilon_back =
        BallMethods.configure_sequences!(SEQUENCE_MODE)

    # The original solvers require this argument in the signature
    env = BallMethods.Gurobi.Env()

    @printf(
        "Ball experiment | sequence=%s | max_iter=%d | time_limit=%.1f s\n",
        string(SEQUENCE_MODE),
        Int(MAX_ITER_BALLS),
        TIME_LIMIT_BALLS,
    )

    println(
        "SCCRM control rule inherited from base: ",
        BallMethods.CONTROL_RULE,
    )
    println("PACA stop: ", PACA_STOP)

    for n_uint in NS, m_uint in MS
        n, m = Int(n_uint), Int(m_uint)

        y, Q, k, x_star, x0, slater_value, initial_min =
            make_ball_instance_inner_spread(n, m)
        println("\nn=$n, m=$m, radius range=$(extrema(k))")
        

        @printf(
            "f_max(x_star)=%.6e; min_i f_i(x0)=%.6e; ||x0||=%.4f\n",
            slater_value, initial_min, norm(x0),
        )

        @printf(
            "Dense Q storage: approximately %.2f MB\n",
            8.0 * m * n * n / 1e6,
        )

        for method in METHODS_BALLS
            run_one_method(
                method, x0, y, Q, k, env, epsilon_back
            )
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_balls()
end

# To run this experiment, comment out the "main()" line in cutting_planes_with_finite.jl and run it using "if abspath(PROGRAM_FILE) == @__FILE__ main() end"
