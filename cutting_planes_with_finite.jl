using Gurobi
using JuMP
using LinearAlgebra
using Random
using Base.Threads

include("plot.jl")
include("util.jl")

println("Num threads: ", nthreads())

using Printf

const EPS = 1e-7
const mt = false

# -------------------------------------------------------------------------
# Control rule for SCCRM, ORSCCRM, URSCCRM, BORSCCRM and BURSCCRM.
#
# Change only this block when you want to test another rule for choosing
# the pair of sets used at each iteration.
#
# :almost_cyclic
#   Uses consecutive pairs of sets:
#   (C1,C2), (C2,C3), ..., (Cm,C1), ...
#
# :most_violated_function
#   Chooses the pair according to the largest function violation f_i(x).
#
# :most_violated_distance
#   Chooses the pair according to the largest distance d(x,C_i).
# -------------------------------------------------------------------------

# const CONTROL_RULE = :almost_cyclic
 const CONTROL_RULE = :most_violated_function
# const CONTROL_RULE = :most_violated_distance

# -------------------------------------------------------------------------
# Sequence choices.
#
# Change only this block whenever you want to test another sequence.
#
# approx_epsilon:
#   used by the relaxed SCCRM methods:
#   OSCCRM, USCCRM, BOSCCRM, BUSCCRM.
#
# paca_epsilon:
#   perturbation sequence used by PACA.
#
# prsccrm_eta:
#   perturbation sequence used to build the perturbed halfspaces in
#   PRSCCRM, OPRSCCRM and UPRSCCRM.
#
# prsccrm_lambda_plain:
#   relaxation parameter for PRSCCRM.
#
# prsccrm_lambda_over:
#   relaxation parameter for OPRSCCRM.
#
# prsccrm_lambda_under:
#   relaxation parameter for UPRSCCRM.
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Option 1: epsilon_k = 1/k
# -------------------------------------------------------------------------

approx_epsilon = (iter::Int) -> 1.0 / iter
paca_epsilon = (iter::Int) -> 1.0 / iter
prsccrm_eta = (iter::Int) -> 1.0 / (iter + 1)
prsccrm_lambda_over = (iter::Int) -> 1.0 + (1.0 / iter)
prsccrm_lambda_under = (iter::Int) -> 1.0 - (1.0 / iter)
prsccrm_lambda_plain = (iter::Int) -> 1.0 # keeps the perturbed halfspaces but does not over- or under-relax the circumcenter step.



# -------------------------------------------------------------------------
# Option 3: epsilon_k = 1/sqrt(k)
# -------------------------------------------------------------------------

# approx_epsilon = (iter::Int) -> 1.0 / sqrt(iter)
# paca_epsilon = (iter::Int) -> 1.0 / sqrt(iter)
# prsccrm_eta = (iter::Int) -> 1.0 / sqrt(iter)
# prsccrm_lambda_over = (iter::Int) -> 1.0 + (1.0 / sqrt(iter))
# prsccrm_lambda_under = (iter::Int) -> 1.0 - (1.0 / sqrt(iter))
# prsccrm_lambda_plain = (iter::Int) -> 1.0


# -------------------------------------------------------------------------
# Backtracking parameters.
#
# epsilonBack:
#   initial trial value used by the backtracking versions:
#   BOSCCRM and BUSCCRM.
#
# epsilonStop:
#   minimum accepted value for the backtracking parameter. 
#
# beta:
#   reduction factor used by backtracking. Usually 0 < beta < 1.
# -------------------------------------------------------------------------

const epsilonStop = 1e-6 
const beta = 0.5 
max_backtracking::UInt = 20 
const epsilonBack = 1.0 

# -------------------------------------------------------------------------

# Update: Cyclic Projections
"""
update_alt_proj!(x::AbstractVector, y::AbstractMatrix,
	Q::AbstractArray, k::AbstractVector,
	env::Gurobi.Env)

Updates the current solution using alternating projections

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real:              precision parameter
"""
function update_alt_proj!(x::AbstractVector, y::AbstractMatrix,
	Q::AbstractArray, k::AbstractVector,
	env::Gurobi.Env, ϵ::Real)
	m, n = size(y)

	for i in 1:m
		x .= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
	end

	return x
end

# Update: Most ViolatedCyclic Projections
function update_mv_alt_proj!(x::AbstractVector, y::AbstractMatrix,
    Q::AbstractArray, k::AbstractVector,
    env::Gurobi.Env, ϵ::Real)

    m, _ = size(y)

    ell = most_violated_ellipsoid_index(x, y, Q, k)

    for offset in 0:m-1
        i = ((Int(ell) + offset - 1) % m) + 1
        x .= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
    end

    return x
end

# Update Cimmino

"""
update_cimmino!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

Updates the current solution using cimmino's projection
- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function update_cimmino!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

	m, n = size(Q, 1), size(Q, 2)
	sum_y = zeros(Float64, n)

	for i in 1:m
		if g(x, y[i, :], Q[i, :, :], k[i]) <= 0.0
			sum_y .+= x
		else
			sum_y .+= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
		end
	end

	x .= (1.0 / m) .* sum_y

	return x
end

"""
update_cimmino_mt!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

Updates the current solution using cimmino's projection in parallel
- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function update_cimmino_mt!(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

	m, n = size(Q, 1), size(Q, 2)
	p = zeros(Float64, m, n)

	Threads.@threads for i in 1:m
		if g(x, y[i, :], Q[i, :, :], k[i]) <= 0.0
			p[i, :] .= x
		else
			p[i, :] .= elipsoid_projection(x, y[i, :], Q[i, :, :], k[i], env)
		end
	end

	x .= (1.0 / m) .* sum(p, dims = 1)
end

"""
composite_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)

Computes the composite projection P_B(P_A(x)) 
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real               precision parameter

"""
function composite_projection(
    x::AbstractVector, A::UInt, B::UInt,
    y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,
)

    xA = elipsoid_projection(x, y[A, :], Q[A, :, :], k[A], env) # First projection: xA = P_A(x).
    return elipsoid_projection(xA, y[B, :], Q[B, :, :], k[B], env) # Second projection: P_B(xA) = P_B(P_A(x)).
end

"""
average_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env, ϵ::Real)

Computes the average projection of A and B: Z = P_A(x) + P_B(x)
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function average_projection(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)
	return 0.5 * (elipsoid_projection(x, y[A, :], Q[A, :, :], k[A], env) +
				  elipsoid_projection(x, y[B, :], Q[B, :, :], k[B], env))
end

"""
find_circuncenter(x::AbstractMatrix)

Given m points x ∈ ℜ^{n}, finds the equidistant point to all points x.
- x::AbstractMatrix:    Set of points x ∈ ℜ^{n}
"""
function find_circuncenter(x::AbstractMatrix)
	m = size(x, 1) - 1
	M = similar(x, m, m)
	b = similar(x, m)
	x_0 = x[1, :]
	for i in 1:m
		for j in 1:m
			M[i, j] = dot(x[j+1, :] .- x_0, x[i+1, :] .- x_0)
		end
		b[i] = 0.5 * norm(x[i+1, :] .- x_0)^2
	end
	result = similar(x_0)
	try
		α = M \ b
		result = copy(x_0)
		for j in 1:m
			result .+= α[j] .* (x[j+1, :] .- x_0)
		end
	catch e
		if isa(e, SingularException) || isa(e, PosDefException)
			result .= sum(x, dims = 1)[:] ./ size(x, 1)
		else
			rethrow(e)
		end
	end
	return result
end


"""
Z(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

	Computes the projection operator ̄z(x) = average_projection(A,B) + composite_projection(A,B)
- x::AbstractVector:    size(x) = n, current x
- A::Uint:              Elipsoid index A
- B::Uint:              Elipsoid index B
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- env::Gurobi.Env:      Gurobi environment to solve projections
"""
function Z(x::AbstractVector, A::UInt, B::UInt, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)
	return average_projection(composite_projection(x, A, B, y, Q, k, env), A, B, y, Q, k, env)
end

"""
reflection(x::AbstractVector, projection::AbstractVector)

Computes the reflection of x given a projected point

- x::AbstractVector:                current point
- projection::AbstractVector:       projected point
"""
function reflection(x::AbstractVector, projection::AbstractVector)
	if size(x) != size(projection)
		return error("Dimension mismatch at reflection: size(x) = $(size(x)), size(projection) = $(size(projection))")
	end
	return 2 * projection .- x
end

# Update ScCRM (and versions over, under, back over, back under)

function update_sc_crm!(
    x::AbstractVector, A::UInt, B::UInt,
    y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,
)
    z_bar = Z(x, A, B, y, Q, k, env)

    R_A = reflection(
        z_bar,
        elipsoid_projection(z_bar, y[A, :], Q[A, :, :], k[A], env),
    )
    R_B = reflection(
        z_bar,
        elipsoid_projection(z_bar, y[B, :], Q[B, :, :], k[B], env),
    )

    _, n = size(y)
    P = zeros(3, n)
    P[1, :] .= z_bar
    P[2, :] .= R_A
    P[3, :] .= R_B

    x .= find_circuncenter(P)
    return x
end


function update_sc_crm_over!(
    x::AbstractVector, A::UInt, B::UInt,
    y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,
    approx_epsilon::Real,
)
    z_bar = Z(x, A, B, y, Q, k, env)

    R_A = reflection(
        z_bar,
        elipsoid_projection(z_bar, y[A, :], Q[A, :, :], k[A], env),
    )
    R_B = reflection(
        z_bar,
        elipsoid_projection(z_bar, y[B, :], Q[B, :, :], k[B], env),
    )

    _, n = size(y)
    P = zeros(3, n)
    P[1, :] .= z_bar
    P[2, :] .= R_A
    P[3, :] .= R_B

    aux = find_circuncenter(P)

    x .+= (1.0 + approx_epsilon) .* (aux .- x)
    return x
end


function update_sc_crm_under!(
    x::AbstractVector, A::UInt, B::UInt,
    y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,
    approx_epsilon::Real,
)
    z_bar = Z(x, A, B, y, Q, k, env)

    R_A = reflection(
        z_bar,
        elipsoid_projection(z_bar, y[A, :], Q[A, :, :], k[A], env),
    )
    R_B = reflection(
        z_bar,
        elipsoid_projection(z_bar, y[B, :], Q[B, :, :], k[B], env),
    )

    _, n = size(y)
    P = zeros(3, n)
    P[1, :] .= z_bar
    P[2, :] .= R_A
    P[3, :] .= R_B

    aux = find_circuncenter(P)

    x .+= (1.0 - approx_epsilon) .* (aux .- x)
    return x
end


function update_sc_crm_over_back!(
    x::AbstractVector, A::UInt, B::UInt,
    y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,
    max_backtracking::UInt,
    epsilonBack::Real, epsilonStop::Real, beta::Real,
)
    x_old = copy(x)
    xcrm = update_sc_crm!(x, A, B, y, Q, k, env)
    direction = xcrm .- x_old

    violation_crm = max_violation(xcrm, y, Q, k, 0.0)
    violation_crm_stop = max_violation(xcrm, y, Q, k, epsilonStop)

    if violation_crm_stop <= 0.0
        return x
    end

    epsilon_trial = epsilonBack
    nb_iter = 1

    while nb_iter <= max_backtracking
        x_over = xcrm .+ epsilon_trial .* direction
        violation_over = max_violation(x_over, y, Q, k, 0.0)

        if violation_over <= violation_crm
            x .= x_over
            return x
        end

        epsilon_trial *= beta
        nb_iter += 1
    end

    return x
end


function update_sc_crm_under_back!(
    x::AbstractVector, A::UInt, B::UInt,
    y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, env::Gurobi.Env,
    max_backtracking::UInt,
    epsilonBack::Real, epsilonStop::Real, beta::Real,
)
    x_old = copy(x) # Preserve the old iterate before update_sc_crm! modifies x.
    xcrm = update_sc_crm!(x, A, B, y, Q, k, env)
    direction = xcrm .- x_old

    violation_crm = max_violation(xcrm, y, Q, k, 0.0)
    violation_crm_stop = max_violation(xcrm, y, Q, k, epsilonStop)

    if violation_crm_stop <= 0.0
        return x
    end

    epsilon_trial = epsilonBack
    nb_iter = 1

    while nb_iter <= max_backtracking
        x_under = xcrm .- epsilon_trial .* direction
        violation_under = max_violation(x_under, y, Q, k, 0.0)

        if violation_under <= violation_crm
            x .= x_under # Store the accepted point in the solver's vector
            return x
        end

        epsilon_trial *= beta
        nb_iter += 1
    end

    return x  # No trial accepted: x still contains the ordinary SCCRM point
end

"""
average_product_space_projection(z::AbstractVector, m::UInt, n::UInt)

Computes the average projection in the product space.
- z::AbstractVector     Current solution in product-space, z ∈ ℜ^{mn}
- m::Uint               size parameter for z 
- n::Uint               size parameter for z
"""
function average_product_space_projection(z::AbstractVector, m::Int, n::Int)
	if size(z, 1) != m * n
		return error("z has wrong size $(size(z,1)) != m*n = $(m*n)")
	end
	Y = reshape(z, n, m)
	return (1.0 / m) * repeat(vec(sum(Y, dims = 2)), m)
end

function ellipsoid_subgradient(x::AbstractVector, y::AbstractVector, Q::AbstractMatrix)
	return 2.0 .* (Q * (x .- y))
end

function paca_step_vector(
    x::AbstractVector,
    y::AbstractVector,
    Q::AbstractMatrix,
    k::Real,
    epsilon_k::Real,
)
    fx = g(x, y, Q, k)
    grad = ellipsoid_subgradient(x, y, Q)
    residual = fx + epsilon_k
    denominator = dot(grad, grad)

    # Check for nonfinite values
    isfinite(residual) && isfinite(denominator) ||
        error("Nonfinite value in PACA correction.")

    # The perturbed constraint is satisfied: no correction is needed
    if residual <= 0.0
        return zeros(length(x))
    end

    # A violated perturbed constraint requires a valid gradient
    # Small positive denominators are not discarded
    denominator > 0.0 ||
        error("Zero or underflowed squared gradient norm in PACA correction.")

    v = (residual / denominator) .* grad

    all(isfinite, v) || error("Nonfinite PACA correction.")
    return v
end


function update_paca!(x::AbstractVector, y::AbstractMatrix,
    Q::AbstractArray, k::AbstractVector, epsilon_k::Real)

    m, n = size(y)
    m > 0 || error("No constraints.")
    v = zeros(Float64, m, n)

    for i in 1:m
        v[i, :] .= paca_step_vector(
            x, y[i, :], Q[i, :, :], k[i], epsilon_k)
    end

    w = vec(sum(v, dims = 1)) ./ m
    numerator = sum(dot(v[i, :], v[i, :]) for i in 1:m) / m
    denominator = dot(w, w)

    # Reject nonfinite values, without discarding small corrections
    isfinite(numerator) && isfinite(denominator) ||
        error("Nonfinite value in PACA extrapolation.")

    # Theoretical case: w^k = 0 implies x^{k+1} = x^k
    if all(iszero, w)
        return x
    end

    # A nonzero w with zero squared norm indicates numerical underflow
    denominator > 0.0 ||
        error("Squared norm underflow in PACA extrapolation.")

    alpha = numerator / denominator
    isfinite(alpha) || error("Nonfinite PACA extrapolation factor.")

    x_new = x .- alpha .* w
    all(isfinite, x_new) || error("Nonfinite PACA iterate.")
    x .= x_new

    return x
end

# -----------------------------------------------------------------
function most_violated_ellipsoid_index(x, y, Q, k; exclude=nothing)
    m = size(y, 1)
    best_i = 1
    best_val = -Inf

    for i in 1:m
        if exclude !== nothing && i == exclude && m > 1
            continue
        end

        diff = x .- y[i, :]
        val = diff' * Q[i, :, :] * diff - k[i]^2

        if val > best_val
            best_val = val
            best_i = i
        end
    end

    return UInt(best_i)
end
# ---
function most_violated_distance_ellipsoid_index(
    x::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    env::Gurobi.Env;
    exclude=nothing,
)
    m = size(y, 1)
    best_i = 1
    best_dist = -Inf

    for i in 1:m
        if exclude !== nothing && i == exclude && m > 1
            continue
        end

        px = elipsoid_projection(
            x,
            y[i, :],
            Q[i, :, :],
            k[i],
            env,
        )

        dist_i = norm(x - px)

        if dist_i > best_dist
            best_dist = dist_i
            best_i = i
        end
    end

    return UInt(best_i)
end
# ---
function choose_pair_indices(
    x::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    iter::Int,
    env::Gurobi.Env,
    ϵ::Real,
)
    m = size(y, 1)

    if CONTROL_RULE == :almost_cyclic
        A_int = (iter % m) + 1
        B_int = A_int == m ? 1 : A_int + 1

        A = UInt(A_int)
        B = UInt(B_int)

    elseif CONTROL_RULE == :most_violated_function
        # Select A using function values at the current iterate.
        A = most_violated_ellipsoid_index(x, y, Q, k)

        xA = elipsoid_projection(
            x, y[A, :], Q[A, :, :], k[A], env,
        )

        # Select a different B using function values at P_A(x).
        B = most_violated_ellipsoid_index(
            xA, y, Q, k; exclude = A,
        )

    elseif CONTROL_RULE == :most_violated_distance
        # Select A using distances at the current iterate.
        A = most_violated_distance_ellipsoid_index(x, y, Q, k, env)

        xA = elipsoid_projection(
            x, y[A, :], Q[A, :, :], k[A], env,
        )

        # Select a different B using distances at P_A(x).
        B = most_violated_distance_ellipsoid_index(
            xA, y, Q, k, env; exclude = A,
        )

    else
        error("Unknown CONTROL_RULE = $CONTROL_RULE")
    end

    # Convention: P_B P_A, so the update projects onto A first
    return A, B
end
# ---
function successor_index(index::UInt, m::Int)
	if m == 1
		return index
	end

	return index == m ? UInt(1) : UInt(index + 1)
end

# ---


function perturbed_halfspace_projection_from_base(
    x::AbstractVector,
    base::AbstractVector,
    y_i::AbstractVector,
    Q_i::AbstractMatrix,
    k_i::Real,
    eta_k::Real,
)
    all(isfinite, x) && all(isfinite, base) ||
        error("Nonfinite point in halfspace projection.")

    f_base = g(base, y_i, Q_i, k_i)
    base_violation = f_base + eta_k

    isfinite(base_violation) ||
        error("Nonfinite perturbed constraint value.")

    # If the base satisfies the perturbed constraint, H_i^k = R^n.
    if base_violation <= 0.0
        return copy(x)
    end

    # Define the halfspace using the unnormalized gradient at the base.
    grad_base = ellipsoid_subgradient(base, y_i, Q_i)
    all(isfinite, grad_base) ||
        error("Nonfinite subgradient.")

    all(iszero, grad_base) &&
        error("Empty perturbed halfspace: zero subgradient.")

    grad_norm_sq = dot(grad_base, grad_base)

    isfinite(grad_norm_sq) ||
        error("Nonfinite squared subgradient norm.")
    grad_norm_sq > 0.0 ||
        error("Squared subgradient norm underflow.")

    halfspace_violation =
        base_violation + dot(grad_base, x .- base)

    isfinite(halfspace_violation) ||
        error("Nonfinite halfspace evaluation.")

    # The intermediate point already belongs to the fixed halfspace.
    if halfspace_violation <= 0.0
        return copy(x)
    end

    projected =
        x .- (halfspace_violation / grad_norm_sq) .* grad_base

    all(isfinite, projected) ||
        error("Nonfinite projected point.")

    return projected
end

function update_perturbed_relaxed_sccrm!(
    z::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    eta_k::Real,
    lambda_k::Real,
)
    m, n = size(y)

    old = copy(z)

    ell = most_violated_ellipsoid_index(old, y, Q, k)

    P_A = x -> perturbed_halfspace_projection_from_base(
        x,
        old,
        y[ell, :],
        Q[ell, :, :],
        k[ell],
        eta_k,
    )

    PA_z = P_A(old)

    r = most_violated_ellipsoid_index(PA_z, y, Q, k; exclude = ell)

    P_B = x -> perturbed_halfspace_projection_from_base(
        x,
        old,
        y[r, :],
        Q[r, :, :],
        k[r],
        eta_k,
    )

    PB_PA_z = P_B(PA_z)
    PA_PB_PA_z = P_A(PB_PA_z)

    u = 0.5 .* (PB_PA_z .+ PA_PB_PA_z)

    v = reflection(u, P_A(u))
    w = reflection(u, P_B(u))

    points = zeros(3, n)
    points[1, :] .= u
    points[2, :] .= v
    points[3, :] .= w

    c = find_circuncenter(points)

    z .= old .+ lambda_k .* (c .- old)

    return z
end

"""
elipsoid_product_space_projection(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)

Computes the projection of z ∈ ℜ^{mn} into the product space of the elipsoids (y_i,Q_i,k_i), i in 1:m
- z::AbstractVector:        Current solution in product-space, z ∈ ℜ^{mn} 
- y::AbstractMatrix:        Elipsoid centers 
- Q::AbstractArray:         Elipsoid matrices
- k::AbstractVector:        Elipsoid radii
- env::Gurobi.Env:          Gurobi environment to solve projections
"""
function elipsoid_product_space_projection(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)
	m, n = size(y)

	if size(z, 1) != m * n
		return error("z has wrong size $(size(z,1)) != m*n = $(m*n)")
	end

	Y = reshape(z, n, m)
	Z = similar(Y)
	for i in 1:m
		Z[:, i] .= elipsoid_projection(Y[:, i], y[i, :], Q[i, :, :], k[i], env)
	end
	return vec(Z)
end


function update_crm!(z::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, env::Gurobi.Env)
	m, n = size(y)
	if size(z, 1) != m * n
		return error("z has wrong size $(size(z,1)) != m*n = $(m*n)")
	end
	P = zeros(3, m * n)
	P[1, :] .= z
	P[2, :] .= reflection(z, elipsoid_product_space_projection(z, y, Q, k, env))
	R_W = reflection(z, elipsoid_product_space_projection(z, y, Q, k, env))
	P[3, :] .= reflection(R_W, average_product_space_projection(R_W, m, n))
	z .= find_circuncenter(P)
	return z
end


"""
solve_alt_proj(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using alternating projections

- x0::AbstractVector:    size(x) = n, initial solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- max_iter::UInt:       max number of iterations
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real               precision parameter
"""
function solve_alt_proj(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = []

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        update_alt_proj!(x, y, Q, k, env, ϵ)

        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end

        iter += 1
    end

    println("alt iter = $iter")
    return x, violation, time() - t_start, iter
end


function solve_mv_alt_proj(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, max_iter::UInt, env::Gurobi.Env, ϵ::Real, time_limit::Real)

    t_start = time()
    x = copy(x0)
    iter = 1
    violation = Float64[]

    while iter < max_iter
		if time() - t_start >= time_limit
    println("time limit reached")
    break
end
        update_mv_alt_proj!(x, y, Q, k, env, ϵ)

        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end

        iter += 1
    end

    println("mv alt proj iter = $iter")

    return x, violation, time() - t_start, iter
end

"""
solve_cimmino(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using cimmino projections

- x0::AbstractVector:    size(x) = n, initial solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii
- max_iter::UInt:       max number of iterations
- env::Gurobi.Env:      Gurobi environment to solve projections
- ϵ::Real               precision parameter
"""
function solve_cimmino(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = Float64[]

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        if mt
            update_cimmino_mt!(x, y, Q, k, env)
        else
            update_cimmino!(x, y, Q, k, env)
        end

        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        iter += 1

        if current_violation <= 0.0
            break
        end
    end

    println("cimmino iter = $iter")
    return x, violation, time() - t_start, iter
end
"""
solve_sc_crm(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
	k::AbstractVector, max_iter::UInt, env::Gurobi.Env)
Finds a point x in the intersection of elipsoids (y_i,Q_i,k_i) using successive centralized CRMs

"""
function solve_sc_crm(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = []

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        A, B = choose_pair_indices(x, y, Q, k, iter, env, ϵ)
        update_sc_crm!(x, A, B, y, Q, k, env)

        iter += 1
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("sc_crm iter = $iter")
    return x, violation, time() - t_start, iter
end


function solve_sc_crm_over(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = []

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        A, B = choose_pair_indices(x, y, Q, k, iter, env, ϵ)
        update_sc_crm_over!(x, A, B, y, Q, k, env, approx_epsilon(iter))

        iter += 1
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("sc_crm iter = $iter")
    return x, violation, time() - t_start, iter
end


function solve_sc_crm_under(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = []

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        A, B = choose_pair_indices(x, y, Q, k, iter, env, ϵ)
        update_sc_crm_under!(x, A, B, y, Q, k, env, approx_epsilon(iter))

        iter += 1
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("sc_crm iter = $iter")
    return x, violation, time() - t_start, iter
end


function solve_sc_crm_under_back(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    max_backtracking::UInt,
    epsilonBack::Real,
    epsilonStop::Real,
    beta::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = []

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        A, B = choose_pair_indices(x, y, Q, k, iter, env, ϵ)
        update_sc_crm_under_back!(
            x, A, B, y, Q, k, env,
            max_backtracking, epsilonBack, epsilonStop, beta,
        )

        iter += 1
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("sc_crm iter = $iter")
    return x, violation, time() - t_start, iter
end


function solve_sc_crm_over_back(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
    max_backtracking::UInt,
    epsilonBack::Real,
    epsilonStop::Real,
    beta::Real,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = []

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        A, B = choose_pair_indices(x, y, Q, k, iter, env, ϵ)
        update_sc_crm_over_back!(
            x, A, B, y, Q, k, env,
            max_backtracking, epsilonBack, epsilonStop, beta,
        )

        iter += 1
        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("sc_crm iter = $iter")
    return x, violation, time() - t_start, iter
end

# PACA method
function solve_paca2(x0::AbstractVector, y::AbstractMatrix, Q::AbstractArray,
    k::AbstractVector, max_iter::UInt, ϵ::Real, time_limit::Real)

    t_start = time()
    x = copy(x0)
    iter = 1
    violation = Float64[]

    while iter < max_iter
		if time() - t_start >= time_limit
    println("time limit reached")
    break
end
        update_paca!(x, y, Q, k, paca_epsilon(iter))

        iter += 1

        current_violation = max_violation(x, y, Q, k, ϵ)
        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("paca2 iter = $iter")

    return x, violation, time() - t_start, iter
end

# PRSCCRM method and variants
function solve_perturbed_relaxed_sccrm(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    ϵ::Real,
    lambda_rule::Function,
    method_name::String,
    time_limit::Real,
)
    t_start = time()
    x = copy(x0)
    iter = 1
    violation = Float64[]

    while iter < max_iter
        if time() - t_start >= time_limit
            println("time limit reached")
            break
        end

        update_perturbed_relaxed_sccrm!(
            x,
            y,
            Q,
            k,
            prsccrm_eta(iter),
            lambda_rule(iter),
        )

        iter += 1

current_violation = max_violation(x, y, Q, k, ϵ)
push!(violation, current_violation)

if current_violation <= 0.0
    break
end
    end

println("$method_name iter = $iter")
return x, violation, time() - t_start, iter
end

#OPRSCCRM = over-relaxed perturbed relaxed SCCRM
function solve_oprsccrm(
	x0::AbstractVector,
	y::AbstractMatrix,
	Q::AbstractArray,
	k::AbstractVector,
	max_iter::UInt,
	ϵ::Real,
	time_limit::Real
)
	return solve_perturbed_relaxed_sccrm(
		x0,
		y,
		Q,
		k,
		max_iter,
		ϵ,
		prsccrm_lambda_over,
		"oprsccrm",
		time_limit,
	)
end

#UPRSCCRM = under-relaxed perturbed relaxed SCCRM:
function solve_uprsccrm(
	x0::AbstractVector,
	y::AbstractMatrix,
	Q::AbstractArray,
	k::AbstractVector,
	max_iter::UInt,
	ϵ::Real,
	time_limit::Real	
)
	return solve_perturbed_relaxed_sccrm(
		x0,
		y,
		Q,
		k,
		max_iter,
		ϵ, # 0.0
		prsccrm_lambda_under,
		"uprsccrm",
		time_limit,
	)
end

# PRSCCRM 

function solve_prsccrm(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    ϵ::Real,
    time_limit::Real
)
    return solve_perturbed_relaxed_sccrm(
        x0,
        y,
        Q,
        k,
        max_iter,
        ϵ,
        prsccrm_lambda_plain,
        "prsccrm",
		time_limit,
    )
end

function solve_crm(
    x0::AbstractVector,
    y::AbstractMatrix,
    Q::AbstractArray,
    k::AbstractVector,
    max_iter::UInt,
    env::Gurobi.Env,
    ϵ::Real,
)
    t_start = time()
    x = copy(x0)
    m, n = size(y)
    iter = 1
    z = repeat(x, m)
    violation = []

    while iter < max_iter
        update_crm!(z, y, Q, k, env)
        iter += 1

        Y = reshape(z, n, m)
        current_violation = -Inf

        for i in 1:m
            violation_i = max_violation(Y[:, i], y, Q, k, ϵ)

            if violation_i > current_violation
                current_violation = violation_i
            end
        end

        push!(violation, current_violation)

        if current_violation <= 0.0
            break
        end
    end

    println("iter crm $iter")
    return z, violation, time() - t_start, iter
end

function test_cutting_plane(n::UInt, m::UInt, max_iter::UInt,
    max_backtracking::UInt, epsilonBack::Real, epsilonStop::Real,
    beta::Real, time_limit::Real)

    env = Gurobi.Env()

    Random.seed!(0)
    lambda_original = 10.0
    y = zeros(m, n)
    Q = zeros(m, n, n)
    k = zeros(m)
    x_common = zeros(n)

    for i in 1:m
        # Symmetric positive-semidefinite matrix
        A = randn(n, n)
        q = A * A'

        # Positive regularization guarantees positive definiteness
        q += lambda_original * I(n)
        q = Matrix(Symmetric(q))
        isposdef(Symmetric(q)) ||
            error("Q_$i is not positive definite.")
        Q[i, :, :] .= q

        # Center coordinates are uniform in [-10, 10).
        y[i, :] .= rand(Float64, n) .* 20.0 .- 10.0

        # The origin is strictly feasible for every ellipsoid.
        k[i] = (1.0 + norm(y[i, :], 2)) * sqrt(norm(q, 2))
    end

    println("ellipsoid generation mode = original")
    println("Slater/common point violation = ",
        max_violation(x_common, y, Q, k, 0.0))

    centers_in_intersection = count(
        i -> max_violation(y[i, :], y, Q, k, 0.0) <= 0.0,
        1:m,
    )
    println("number of centers in intersection = ",
        centers_in_intersection, " / ", m)

    # Double x0 until it is outside every tolerance-expanded ellipsoid
    kfactor = 2
    x0 = ones(n)
    ϵ = 1e-7
    while min_violation(x0, y, Q, k, ϵ) <= 0.0
        x0 .*= kfactor
    end

    # Optional additional displacement:
    # x0 .*= 10.0

    println("violation x0 = ", max_violation(x0, y, Q, k, ϵ))


function safe_run_method(run_method::Function, method_name::String)
    t_start = time_ns()
    local x, violation, elapsed_time, iter

    try
        x, violation, elapsed_time, iter = run_method()
    catch err
        err isa InterruptException && rethrow()
        elapsed_time = (time_ns() - t_start) / 1e9
        println("$method_name status = NUMERICAL_FAILURE")
        println("Error: ", sprint(showerror, err))
        println("original_intersection = NOT_CHECKED")
        return fill(NaN, length(x0)), Float64[], elapsed_time, 0,
               "NUMERICAL_FAILURE"
    end

    # Post-run verification: excluded from the solver's elapsed_time.
    valid = all(isfinite, x)
    v_original = -Inf
    v_tolerance = -Inf

    if valid
        for i in axes(y, 1)
            diff = x .- y[i, :]
            quadratic = dot(diff, Q[i, :, :] * diff)
            original_i = quadratic - k[i]^2
            tolerance_i = quadratic - (k[i] + ϵ)^2

            # Check every constraint before computing the maxima.
            if !all(isfinite, (quadratic, original_i, tolerance_i))
                valid = false
                break
            end

            v_original = max(v_original, original_i)
            v_tolerance = max(v_tolerance, tolerance_i)
        end
    end

    valid = valid && isfinite(v_original) && isfinite(v_tolerance)
    if !valid
        v_original = NaN
        v_tolerance = NaN
    end

    # Common stopping criterion, including PACA.2.
    reached = valid && v_tolerance <= 0.0

    status = if !valid
        "NUMERICAL_FAILURE"
    elseif reached
        "OK"
    elseif iter >= max_iter
        "MAX_ITER"
    elseif elapsed_time >= time_limit
        "TIME_LIMIT"
    else
        "NUMERICAL_FAILURE"
    end

    membership = !valid ? "NOT_CHECKED" :
                 v_original <= 0.0 ? "YES" : "NO"

    println("$method_name time = $elapsed_time seconds, iter = $iter, status = $status")
    println("  original_violation = $v_original")
    println("  tolerance_violation = $v_tolerance")
    println("  original_intersection = $membership")

    return x, violation, elapsed_time, iter, status
end


	# -------------- TESTED METHOD  --------------

	x_alt_proj, violation_alt, time_alt_proj, iter_alt_proj, status_alt =
    safe_run_method("alt_proj") do
        solve_alt_proj(x0, y, Q, k, max_iter, env, ϵ, time_limit)
    end

x_mv_alt_proj, violation_mv_alt, time_mv_alt_proj, iter_mv_alt_proj, status_mv_alt =
    safe_run_method("mv_alt_proj") do
        solve_mv_alt_proj(x0, y, Q, k, max_iter, env, ϵ, time_limit)
    end

	x_cimmino, violation_cimmino, time_cimmino, iter_cimmino, status_cimmino =
    safe_run_method("cimmino") do
        solve_cimmino(x0, y, Q, k, max_iter, env, ϵ, time_limit)
    end


x_sccrm, violation_sccrm, time_sccrm, iter_sccrm, status_sccrm =
    safe_run_method("sccrm") do
        solve_sc_crm(x0, y, Q, k, max_iter, env, ϵ, time_limit)
    end

x_sc_crm_over, violation_sc_crm_over, time_sc_crm_over, iter_sc_crm_over, status_sc_crm_over =
safe_run_method("sc_crm_over") do
    solve_sc_crm_over(x0, y, Q, k, max_iter, env, ϵ, time_limit)
end

x_sc_crm_over_back, violation_sc_crm_over_back, time_sc_crm_over_back, iter_sc_crm_over_back, status_sc_crm_over_back =
safe_run_method("sc_crm_over_back") do
    solve_sc_crm_over_back(
        x0, y, Q, k, max_iter, env, ϵ,
        max_backtracking, epsilonBack, epsilonStop, beta,
        time_limit,
    )
end

x_sc_crm_under, violation_sc_crm_under, time_sc_crm_under, iter_sc_crm_under, status_sc_crm_under =
safe_run_method("sc_crm_under") do
    solve_sc_crm_under(x0, y, Q, k, max_iter, env, ϵ, time_limit)
end

x_sc_crm_under_back, violation_sc_crm_under_back, time_sc_crm_under_back, iter_sc_crm_under_back, status_sc_crm_under_back =
safe_run_method("sc_crm_under_back") do
    solve_sc_crm_under_back(
        x0, y, Q, k, max_iter, env, ϵ,
        max_backtracking, epsilonBack, epsilonStop, beta,
        time_limit,
    )
end

x_paca2, violation_paca2, time_paca2, iter_paca2, status_paca2 =
    safe_run_method("paca2") do
        solve_paca2(x0, y, Q, k, max_iter, ϵ, time_limit)
    end

x_prsccrm, violation_prsccrm, time_prsccrm, iter_prsccrm, status_prsccrm =
    safe_run_method("prsccrm") do
        solve_prsccrm(x0, y, Q, k, max_iter, ϵ, time_limit)
    end

x_oprsccrm, violation_oprsccrm, time_oprsccrm, iter_oprsccrm, status_oprsccrm =
    safe_run_method("oprsccrm") do
        solve_oprsccrm(x0, y, Q, k, max_iter, ϵ, time_limit)
    end

x_uprsccrm, violation_uprsccrm, time_uprsccrm, iter_uprsccrm, status_uprsccrm =
    safe_run_method("uprsccrm") do
        solve_uprsccrm(x0, y, Q, k, max_iter, ϵ, time_limit)
    end


end

function test_find_circuncenter(m, n)
    Random.seed!(0)

    # Generate m random points in n-dimensional space.
    x = rand(m, n)

    # Calculate the circumcenter.
    c = find_circuncenter(x)
    @show c

    # Print its distance to each point.
    for i in 1:m
        println("Distance to point $i = ", norm(c .- x[i, :]))
    end

    return c
end

function main()
    max_iter::UInt = 500000
	time_limit = 100.0

	 ms = UInt[2, 5, 10, 20, 50]
	 ns = UInt[20, 50, 100, 200, 500, 1000]


    println("Num threads: ", Threads.nthreads())

    for n in ns
        for m in ms
            println("Testing n = $n, m = $m")
            test_cutting_plane(n, m, max_iter, max_backtracking, epsilonBack, epsilonStop, beta, time_limit)
        end
    end
end

# main() # for this script to be run as a module, comment out this line and call test_cutting_plane() from another script
 if abspath(PROGRAM_FILE) == @__FILE__
     main()
 end
