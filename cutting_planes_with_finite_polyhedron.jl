using Gurobi
using JuMP
using LinearAlgebra
using Random
using Printf
import MathOptInterface as MOI

const METHODS = [
    #"Alt. Proj.",
    # "MV Alt. Proj.",
    # "Cimmino",
    # "SCCRM",
    # "OSCCRM",
    # "BOSCCRM",
    # "USCCRM",
    "PACA",
    "PRSCCRM",
    "OPRSCCRM",
    "UPRSCCRM",
]

# -------------------------------------------------------------------------
# Sequence choices.
# -------------------------------------------------------------------------
# approx_epsilon = (iter::Int) -> 1.0 / iter
# paca_epsilon = (iter::Int) -> 1.0 / iter
# prsccrm_eta = (iter::Int) -> 1.0 / iter
# prsccrm_lambda_over = (iter::Int) -> 1.0 + (1.0 / iter)
# prsccrm_lambda_under = (iter::Int) -> 1.0 - (1.0 / iter)
# prsccrm_lambda_plain = (iter::Int) -> 1.0
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
 approx_epsilon = (iter::Int) -> 1.0 / sqrt(iter)
 paca_epsilon = (iter::Int) -> 1.0 / sqrt(iter)
 prsccrm_eta = (iter::Int) -> 1.0 / sqrt(iter)
 prsccrm_lambda_over = (iter::Int) -> 1.0 + (1.0 / sqrt(iter))
 prsccrm_lambda_under = (iter::Int) -> 1.0 - (1.0 / sqrt(iter))
 prsccrm_lambda_plain = (iter::Int) -> 1.0

struct MethodResult
    time::Float64
    iterations::Int
    violation::Float64
    status::String
end

function constraints_per_polyhedron(n::Int)
    return 2 * n
end

# function constraints_per_polyhedron(n::Int)
#     return n
# end

function max_violation_poly(x::AbstractVector, b::AbstractMatrix, A::AbstractArray)
    all(isfinite, x) || throw(DomainError(x, "Nonfinite iterate."))
    number_sets = size(A, 1)
    number_sets > 0 || throw(ArgumentError("No polyhedra."))
    value = -Inf
    for i in 1:number_sets
        residuals = A[i, :, :] * x .- b[i, :]
        all(isfinite, residuals) ||
            throw(DomainError(i, "Nonfinite polyhedron residual."))
        value = max(value, maximum(residuals))
    end
    return value
end

function polyhedron_value_and_subgradient(
    x::AbstractVector,
    b_i::AbstractVector,
    A_i::AbstractMatrix,
)
    violations = A_i * x .- b_i
    row_index = argmax(violations)

    return violations[row_index], vec(A_i[row_index, :])
end

function polyhedron_projection_safe(
    x::AbstractVector,
    b::AbstractVector,
    A::AbstractMatrix,
    env::Gurobi.Env,
)
    if all(A * x .<= b)
        return copy(x)
    end

    n = length(x)
    bound = max(1e4, 100.0 * maximum(abs.(x)), 100.0 * maximum(abs.(b)), 100.0)

    for attempt in 1:3
        model = direct_model(Gurobi.Optimizer(env))
        set_silent(model)

        set_optimizer_attribute(model, "TimeLimit", 600.0)
        set_optimizer_attribute(model, "NumericFocus", 3)
        set_optimizer_attribute(model, "DualReductions", 0)
        set_optimizer_attribute(model, "ScaleFlag", 2)

        @variable(model, -bound <= y_var[1:n] <= bound)

        @objective(model, Min, 0.5 * sum((y_var[j] - x[j])^2 for j in 1:n))
        @constraint(model, A * y_var .<= b)

        optimize!(model)

        status = termination_status(model)

        if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
            return value.(y_var)
        end

        if result_count(model) > 0
            return value.(y_var)
        end

        if status == MOI.INFEASIBLE_OR_UNBOUNDED ||
           status == MOI.INFEASIBLE ||
           status == MOI.NUMERICAL_ERROR
            bound *= 100.0
            continue
        end

        error("Projection failed with status = $status")
    end

    error("Projection failed after retries. max violation = $(maximum(A * x .- b))")
end

function generate_polyhedra(
    number_sets::Int,
    n::Int,
    p::Int;
    alpha::Float64 = 1.0,
    seed::Int = 0,
)
    Random.seed!(seed)

    x_star = randn(n)
    A = zeros(number_sets, p, n)
    b = zeros(number_sets, p)

    for i in 1:number_sets
        Ai = randn(p, n)

        for row in 1:p
            row_norm = norm(Ai[row, :])
            if row_norm > 0.0
                Ai[row, :] ./= row_norm
            end
        end

        A[i, :, :] .= Ai
        b[i, :] .= Ai * x_star .+ alpha .* rand(p)
    end

    x0 = 20.0 .* ones(n)

# Ensure that the initial point is outside the intersection.
# It is enough that x0 violates at least one inequality of at least one polyhedron.
scale_attempts = 0
while max_violation_poly(x0, b, A) <= 0.0
    x0 .*= 2.0
    scale_attempts += 1

    if scale_attempts > 20
        error("Could not generate an initial point outside the intersection.")
    end
end

return x0, b, A
end

# function generate_polyhedra(
#     number_sets::Int,
#     n::Int,
#     p::Int;
#     alpha::Float64 = 1.0,
#     seed::Int = 0,
# )
#     Random.seed!(seed)

#     x_star = randn(n)
#     A = zeros(number_sets, p, n)
#     b = zeros(number_sets, p)

#     for i in 1:number_sets
#         # Gaussian coefficients without row normalization.
#         Ai = randn(p, n)

#         A[i, :, :] .= Ai
#         b[i, :] .= Ai * x_star .+ alpha .* rand(p)
#     end

#     x0 = 20.0 .* ones(n)

#     # Double x0 until it lies outside the intersection.
#     scale_attempts = 0
#     while max_violation_poly(x0, b, A) <= 0.0
#         x0 .*= 2.0
#         scale_attempts += 1

#         if scale_attempts > 20
#             error("Could not generate an initial point outside the intersection.")
#         end
#     end

#     return x0, b, A
# end

function projection_to_set(x, b, A, index, env)
    return polyhedron_projection_safe(x, b[index, :], A[index, :, :], env)
end

function update_alt_proj!(x, b, A, env)
    number_sets, _, _ = size(A)

    for i in 1:number_sets
        x .= projection_to_set(x, b, A, i, env)
    end

    return x
end

function update_mv_alt_proj!(x, b, A, env)
    number_sets, _, _ = size(A)

    ell = most_violated_polyhedron_index(x, b, A)

    for offset in 0:number_sets-1
        i = ((ell + offset - 1) % number_sets) + 1
        x .= projection_to_set(x, b, A, i, env)
    end

    return x
end

function update_cimmino!(x, b, A, env)
    number_sets, _, n = size(A)
    sum_proj = zeros(n)

    for i in 1:number_sets
        if all(A[i, :, :] * x .<= b[i, :])
            sum_proj .+= x
        else
            sum_proj .+= projection_to_set(x, b, A, i, env)
        end
    end

    x .= sum_proj ./ number_sets

    return x
end

function composite_projection(x, first_set, second_set, b, A, env)
    projected = projection_to_set(x, b, A, first_set, env)
    return projection_to_set(projected, b, A, second_set, env)
end

function average_projection(x, first_set, second_set, b, A, env)
    return 0.5 .* (
        projection_to_set(x, b, A, first_set, env) .+
        projection_to_set(x, b, A, second_set, env)
    )
end

function z_operator(x, first_set, second_set, b, A, env)
    y = composite_projection(x, first_set, second_set, b, A, env)
    return average_projection(y, first_set, second_set, b, A, env)
end

reflection(x, projection) = 2.0 .* projection .- x

function find_circumcenter(points::AbstractMatrix)
    m = size(points, 1) - 1
    M = similar(points, m, m)
    rhs = similar(points, m)
    x0 = points[1, :]

    for i in 1:m
        for j in 1:m
            M[i, j] = dot(points[j + 1, :] .- x0, points[i + 1, :] .- x0)
        end
        rhs[i] = 0.5 * norm(points[i + 1, :] .- x0)^2
    end

    try
        alpha = M \ rhs
        result = copy(x0)

        for j in 1:m
            result .+= alpha[j] .* (points[j + 1, :] .- x0)
        end

        return result
    catch err
        if isa(err, SingularException) || isa(err, PosDefException)
            return vec(sum(points, dims = 1)) ./ size(points, 1)
        end

        rethrow(err)
    end
end

function sc_pair(iter::Int, number_sets::Int)
    first_set = (iter % number_sets) + 1
    second_set = first_set == number_sets ? 1 : first_set + 1

    return first_set, second_set
end

function update_sc_crm!(x, first_set, second_set, b, A, env)
    z_bar = z_operator(x, first_set, second_set, b, A, env)

    r_first = reflection(z_bar, projection_to_set(z_bar, b, A, first_set, env))
    r_second = reflection(z_bar, projection_to_set(z_bar, b, A, second_set, env))

    _, _, n = size(A)

    points = zeros(3, n)
    points[1, :] .= z_bar
    points[2, :] .= r_first
    points[3, :] .= r_second

    x .= find_circumcenter(points)

    return x
end

function update_sc_crm_over!(x, first_set, second_set, b, A, env, epsilon_k)
    old = copy(x)

    update_sc_crm!(x, first_set, second_set, b, A, env)

    x .= old .+ (1.0 + epsilon_k) .* (x .- old)

    return x
end

function update_sc_crm_under!(x, first_set, second_set, b, A, env, epsilon_k)
    old = copy(x)

    update_sc_crm!(x, first_set, second_set, b, A, env)

    x .= old .+ (1.0 - epsilon_k) .* (x .- old)

    return x
end

function update_sc_crm_over_back!(
    x,
    first_set,
    second_set,
    b,
    A,
    env,
    max_backtracking,
    epsilon_back,
    epsilon_stop,
    beta,
)
    old = copy(x)
    crm_point = copy(x)

    update_sc_crm!(crm_point, first_set, second_set, b, A, env)

    crm_violation = max_violation_poly(crm_point, b, A)

    if crm_violation <= epsilon_stop
        x .= crm_point
        return x
    end

    step = epsilon_back

    for _ in 1:max_backtracking
        candidate = crm_point .+ step .* (crm_point .- old)

        if max_violation_poly(candidate, b, A) <= crm_violation
            x .= candidate
            return x
        end

        step *= beta
    end

    x .= crm_point

    return x
end

function update_sc_crm_under_back!(
    x,
    first_set,
    second_set,
    b,
    A,
    env,
    max_backtracking,
    epsilon_back,
    epsilon_stop,
    beta,
)
    old = copy(x)
    crm_point = copy(x)

    update_sc_crm!(crm_point, first_set, second_set, b, A, env)

    crm_violation = max_violation_poly(crm_point, b, A)

    if crm_violation <= epsilon_stop
        x .= crm_point
        return x
    end

    step = epsilon_back

    for _ in 1:max_backtracking
        candidate = crm_point .- step .* (crm_point .- old)

        if max_violation_poly(candidate, b, A) <= crm_violation
            x .= candidate
            return x
        end

        step *= beta
    end

    x .= crm_point

    return x
end

# -------------------------------------------------------------------------
# PACA for polyhedra.
#
# Each polyhedron is represented by the convex function
# f_i(x) = max_j {a_ij' x - b_ij}.
# Then P_i = {x : f_i(x) <= 0}.
# -------------------------------------------------------------------------
# function paca_step_vector_poly(
#     x::AbstractVector,
#     b_i::AbstractVector,
#     A_i::AbstractMatrix,
#     epsilon_k::Real,
# )
#     fx, grad = polyhedron_value_and_subgradient(x, b_i, A_i)
#     grad_norm_sq = dot(grad, grad)

#     if grad_norm_sq <= eps(Float64)
#         return zeros(length(x))
#     end

#     return (max(0.0, fx + epsilon_k) / grad_norm_sq) .* grad
# end

function paca_step_vector_poly(
    x::AbstractVector, b_i::AbstractVector,
    A_i::AbstractMatrix, epsilon_k::Real,
)
    fx, grad = polyhedron_value_and_subgradient(x, b_i, A_i)
    residual = fx + epsilon_k
    denominator = dot(grad, grad)
    isfinite(residual) && isfinite(denominator) ||
        throw(DomainError(denominator, "Nonfinite PACA correction data."))
    denominator > 0.0 ||
        throw(DomainError(denominator, "Zero denominator in PACA correction."))
    v = (max(0.0, residual) / denominator) .* grad
    all(isfinite, v) || throw(DomainError(v, "Nonfinite PACA correction."))
    return v
end

function update_paca_poly!(
    x::AbstractVector, b::AbstractMatrix,
    A::AbstractArray, epsilon_k::Real,
)
    number_sets, _, n = size(A)
    number_sets > 0 || throw(ArgumentError("No polyhedra."))
    v = zeros(Float64, number_sets, n)
    for i in 1:number_sets
        v[i, :] .= paca_step_vector_poly(x, b[i, :], A[i, :, :], epsilon_k)
    end
    w = vec(sum(v, dims = 1)) ./ number_sets
    numerator = sum(dot(v[i, :], v[i, :]) for i in 1:number_sets) / number_sets
    denominator = dot(w, w)
    isfinite(numerator) && isfinite(denominator) ||
        throw(DomainError(denominator, "Nonfinite PACA extrapolation data."))
        if all(iszero, w)
    return x
end
    denominator > 0.0 ||
        throw(DomainError(denominator, "Zero denominator in PACA extrapolation."))
    alpha = numerator / denominator
    isfinite(alpha) || throw(DomainError(alpha, "Nonfinite PACA factor."))
    x_new = x .- alpha .* w
    all(isfinite, x_new) || throw(DomainError(x_new, "Nonfinite PACA iterate."))
    x .= x_new
    return x
end

# -------------------------------------------------------------------------
# Perturbed relaxed successive cCRM for polyhedra.
# -------------------------------------------------------------------------

function most_violated_polyhedron_index(
    x::AbstractVector, b::AbstractMatrix, A::AbstractArray; exclude = nothing,
)
    number_sets = size(A, 1)
    number_sets > 0 || throw(ArgumentError("No polyhedra."))
    best_index = 0
    best_value = -Inf
    for i in 1:number_sets
        if number_sets > 1 && i == exclude
            continue
        end
        residuals = A[i, :, :] * x .- b[i, :]
        all(isfinite, residuals) ||
            throw(DomainError(i, "Nonfinite value in most-violated selection."))
        value_i = maximum(residuals)
        if best_index == 0 || value_i > best_value
            best_value = value_i
            best_index = i
        end
    end
    best_index > 0 || error("No index available.")
    return best_index
end

function successor_index(index::Int, number_sets::Int)
    if number_sets == 1
        return index
    end

    return index == number_sets ? 1 : index + 1
end


function perturbed_halfspace_projection_from_poly_base(
    x::AbstractVector, base::AbstractVector,
    b_i::AbstractVector, A_i::AbstractMatrix, eta_k::Real,
)
    all(isfinite, x) && all(isfinite, base) ||
        throw(DomainError(base, "Nonfinite point in halfspace projection."))
    residuals = A_i * base .- b_i
    all(isfinite, residuals) ||
        throw(DomainError(residuals, "Nonfinite base residuals."))
    row_index = argmax(residuals)
    base_violation = residuals[row_index] + eta_k
    isfinite(base_violation) ||
        throw(DomainError(base_violation, "Nonfinite perturbed value."))

    # Decide using base = z^k, once for this fixed halfspace.
    # If base satisfies the perturbed constraint, H_i^k = R^n.
    if base_violation <= 0.0
        return copy(x)
    end

    # A maximizing row is a subgradient of the polyhedral max function
    grad_base = vec(A_i[row_index, :])
    all(isfinite, grad_base) ||
        throw(DomainError(grad_base, "Nonfinite subgradient."))
    grad_norm = norm(grad_base)
    isfinite(grad_norm) && grad_norm > 0.0 ||
        throw(DomainError(grad_norm, "Invalid normal or empty perturbed halfspace."))
    normal = grad_base ./ grad_norm
    h = base_violation / grad_norm + dot(normal, x .- base)
    isfinite(h) || throw(DomainError(h, "Nonfinite halfspace evaluation."))
    if h <= 0.0
        return copy(x)
    end
    projected = x .- h .* normal
    all(isfinite, projected) ||
        throw(DomainError(projected, "Nonfinite projected point."))
    return projected
end

function update_perturbed_relaxed_sccrm_poly!(
    z::AbstractVector, b::AbstractMatrix, A::AbstractArray,
    eta_k::Real, lambda_k::Real,
)
    isfinite(eta_k) && eta_k > 0.0 ||
        throw(DomainError(eta_k, "eta_k must be positive and finite."))
    isfinite(lambda_k) ||
        throw(DomainError(lambda_k, "Nonfinite relaxation parameter."))
    old = copy(z)
    ell = most_violated_polyhedron_index(old, b, A)
    P_A = x -> perturbed_halfspace_projection_from_poly_base(
        x, old, b[ell, :], A[ell, :, :], eta_k,
    )
    PA_z = P_A(old)

    # Choose r at P_A(z^k), excluding ell when there is another set
    r = most_violated_polyhedron_index(PA_z, b, A; exclude = ell)
    # Both halfspaces are still constructed at old = z^k
    P_B = x -> perturbed_halfspace_projection_from_poly_base(
        x, old, b[r, :], A[r, :, :], eta_k,
    )
    PB_PA_z = P_B(PA_z)
    PA_PB_PA_z = P_A(PB_PA_z)
    u = 0.5 .* (PB_PA_z .+ PA_PB_PA_z)
    v = reflection(u, P_A(u))
    w = reflection(u, P_B(u))
    points = zeros(3, length(z))
    points[1, :] .= u
    points[2, :] .= v
    points[3, :] .= w
    all(isfinite, points) ||
        throw(DomainError(points, "Nonfinite circumcenter input."))
    c = find_circumcenter(points)
    z_new = old .+ lambda_k .* (c .- old)
    all(isfinite, z_new) ||
        throw(DomainError(z_new, "Nonfinite PRSCCRM iterate."))
    z .= z_new
    return z
end

function solve_polyhedron_method(
    x0, b, A, env, method;
    max_iter = 100, tol = 1e-2, time_limit = 600.0,
    max_backtracking = 20, epsilon_back = 1.0,
    epsilon_stop = 1e-6, beta = 0.5,
)
    supported = ("Alt. Proj.", "MV Alt. Proj.", "Cimmino", "SCCRM",
        "OSCCRM", "BOSCCRM", "USCCRM", "BUSCCRM",
        "PACA", "PRSCCRM", "OPRSCCRM", "UPRSCCRM")
    method in supported || throw(ArgumentError("Unknown method: $method"))
    max_iter isa Integer && max_iter >= 0 ||
        throw(ArgumentError("max_iter must be a nonnegative integer."))
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative."))
    !isnan(time_limit) && time_limit >= 0 ||
        throw(ArgumentError("time_limit must be nonnegative."))

    t_start = time_ns()
    elapsed() = Float64(time_ns() - t_start) / 1e9
    x = copy(x0)
    number_sets = size(A, 1)
    iter_done = 0
    violation = NaN
    status = "NUMERICAL_FAILURE"
    failure = nothing

    try
        violation = max_violation_poly(x, b, A)
        while true
            # OK means the numerical stopping tolerance was met.
            if violation <= tol
                status = "OK"
                break
            elseif iter_done >= max_iter
                status = "MAX_ITER"
                break
            elseif elapsed() >= time_limit
                status = "TIME_LIMIT"
                break
            end

            iter = iter_done + 1
            if method == "Alt. Proj."
                update_alt_proj!(x, b, A, env)
            elseif method == "MV Alt. Proj."
                update_mv_alt_proj!(x, b, A, env)
            elseif method == "Cimmino"
                update_cimmino!(x, b, A, env)
            elseif method == "SCCRM"
                first_set, second_set = sc_pair(iter, number_sets)
                update_sc_crm!(x, first_set, second_set, b, A, env)
            elseif method == "OSCCRM"
                first_set, second_set = sc_pair(iter, number_sets)
                update_sc_crm_over!(x, first_set, second_set, b, A, env, approx_epsilon(iter))
            elseif method == "USCCRM"
                first_set, second_set = sc_pair(iter, number_sets)
                update_sc_crm_under!(x, first_set, second_set, b, A, env, approx_epsilon(iter))
            elseif method == "BOSCCRM"
                first_set, second_set = sc_pair(iter, number_sets)
                update_sc_crm_over_back!(x, first_set, second_set, b, A, env,
                    max_backtracking, epsilon_back, epsilon_stop, beta)
            elseif method == "BUSCCRM"
                first_set, second_set = sc_pair(iter, number_sets)
                update_sc_crm_under_back!(x, first_set, second_set, b, A, env,
                    max_backtracking, epsilon_back, epsilon_stop, beta)
            elseif method == "PACA"
                update_paca_poly!(x, b, A, paca_epsilon(iter))
            else
                lambda_k = method == "PRSCCRM" ? prsccrm_lambda_plain(iter) :
                    method == "OPRSCCRM" ? prsccrm_lambda_over(iter) :
                    prsccrm_lambda_under(iter)
                update_perturbed_relaxed_sccrm_poly!(x, b, A, prsccrm_eta(iter), lambda_k)
            end
            # Count completed outer updates, not backtracking trials.
            iter_done = iter
            violation = max_violation_poly(x, b, A)
        end
    catch err
        err isa InterruptException && rethrow()
        # Do not hide programming/installation errors as numerical failures.
        if !(err isa DomainError || err isa OverflowError ||
             err isa SingularException || err isa PosDefException)
            rethrow()
        end
        status = "NUMERICAL_FAILURE"
        failure = err
    end

    # Freeze the reported elapsed time BEFORE the independent final check.
    elapsed_seconds = elapsed()
    original_violation = NaN
    tolerance_violation = NaN
    membership = "UNVERIFIED"
    try
        original_violation = max_violation_poly(x, b, A)
        tolerance_violation = original_violation - tol
        membership = original_violation <= 0.0 ? "YES" : "NO"
        if status == "OK" && !(original_violation <= tol)
            status = "NUMERICAL_FAILURE"
        end
    catch err
        err isa InterruptException && rethrow()
        err isa DomainError || rethrow()
        status = "NUMERICAL_FAILURE"
        failure === nothing && (failure = err)
    end

    println("    ", method, " | post-run check (excluded from timing)")
    println("        original_violation = ", original_violation)
    println("        tolerance_violation = ", tolerance_violation)
    println("        original_intersection = ", membership)
    if failure !== nothing
        println("        numerical failure: ", sprint(showerror, failure))
    end
    return MethodResult(elapsed_seconds, iter_done, original_violation, status)
end

function run_instance(
    number_sets,
    n;
    max_iter = 100,
    tol = 1e-6,
    alpha = 1.0,
    seed = 0,
    time_limit = 600.0,
    max_backtracking = 20,
    epsilon_back = 1.0,
    epsilon_stop = 1e-6,
    beta = 0.5,
)
    p = constraints_per_polyhedron(n)

    x0, b, A = generate_polyhedra(number_sets, n, p; alpha = alpha, seed = seed)

    env = Gurobi.Env()

    println("Testing m = $number_sets, n = $n, p = $p")
    println("\tinitial violation = $(max_violation_poly(x0, b, A))")

    results = Dict{String, MethodResult}()

    for method in METHODS
        results[method] = solve_polyhedron_method(
            x0,
            b,
            A,
            env,
            method;
            max_iter = max_iter,
            tol = tol,
            time_limit = time_limit,
            max_backtracking = max_backtracking,
            epsilon_back = epsilon_back,
            epsilon_stop = epsilon_stop,
            beta = beta,
        )

        result = results[method]

        @printf(
            "\t%-10s time = %.4f s, iter = %d, violation = %.4e, status = %s\n",
            method,
            result.time,
            result.iterations,
            result.violation,
            result.status,
        )
    end

    return results
end

function print_latex_table(table_results)
    println("\\begin{table}[htbp]")
    println("\\centering")
    println("\\scriptsize")
    println("\\setlength{\\tabcolsep}{2pt}")
    println("\\caption{Polyhedron feasibility experiments. Each method reports running time \$t\$ in seconds and number of iterations.}")
    println("\\resizebox{\\textwidth}{!}{%")
    println("\\begin{tabular}{", repeat("r", 2 + 2 * length(METHODS)), "}")
    println("\\hline")

    print("\\multirow{2}{*}{\$m\$} & \\multirow{2}{*}{\$n\$}")
    for method in METHODS
        print(" & \\multicolumn{2}{c}{", method, "}")
    end
    println(" \\\\")

    for i in 1:length(METHODS)
        first_col = 2 * i + 1
        second_col = first_col + 1
        print("\\cline{$first_col-$second_col} ")
    end
    println()

    print(" & ")
    for _ in METHODS
        print("& \$t\$ (s) & it. ")
    end
    println("\\\\")
    println("\\hline")

    for ((number_sets, n), results) in table_results
        @printf("%d & %d", number_sets, n)

        for method in METHODS
            result = results[method]
            @printf(" & %.4f & %d", result.time, result.iterations)
        end

        println(" \\\\")
    end

    println("\\hline")
    println("\\end{tabular}%")
    println("}")
    println("\\end{table}")
end


function main()

    set_counts = [100, 200]
    dimensions = [20, 50, 100]

    max_iter = 30000
    tol = 1e-6 # stopping tolerance for the maximum violation of the constraints: stop criterion tolerance
    time_limit = 600.0


    max_backtracking = 20
    epsilon_back = 1.0
    epsilon_stop = 1e-6
    beta = 0.5

    table_results = []

    for number_sets in set_counts
        for n in dimensions
            results = run_instance(
                number_sets,
                n;
                max_iter = max_iter,
                tol = tol,
                time_limit = time_limit,
                max_backtracking = max_backtracking,
                epsilon_back = epsilon_back,
                epsilon_stop = epsilon_stop,
                beta = beta,
            )

            push!(table_results, ((number_sets, n), results))
        end
    end

    println()
    println("LATEX TABLE:")
    print_latex_table(table_results)
end

 main()
# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end



# We normalize the rows of A_i so that the most_violated rule compares
# violations on a common geometric scale, avoiding bias toward an
# inequality simply because its coefficients have a larger norm.
#
# With ||a_ij|| = 1, max(0, a_ij' * x - b_ij) is the distance to the
# corresponding half-space. Therefore, when f_i(x) > 0,
# f_i(x) = max_j(a_ij' * x - b_ij) represents the greatest distance
# to the half-spaces defining C_i, rather than necessarily the distance
# to the entire polyhedron C_i.
    