# =========================================================================
# Two-ellipsoid experiments with a controlled intersection angle
#
# The prescribed angle is the angle between the outward normal vectors
# of the two ellipsoids at the common boundary point p = 0
#
# =========================================================================

include("cutting_planes_with_finite.jl")

using LinearAlgebra
using Printf
using Plots
using Gurobi

# -------------------------------------------------------------------------
# Angles to be tested.
# -------------------------------------------------------------------------

const ANGLE_CASES = [
    ("acute_10", 10.0),
    ("acute_35", 35.0),
    ("acute_50", 50.0),
    ("right_90", 90.0),
    ("obtuse_120", 120.0),
    ("obtuse_160", 160.0),
]

# Use dimension 3 for the numerical experiments => plotting code can be used to visualize the geometry in the first two coordinates
const AMBIENT_DIMENSION = 3

# -------------------------------------------------------------------------
# General experiment parameters
# -------------------------------------------------------------------------

const MAX_ITER = UInt(50_000)
const TIME_LIMIT = 600.0
const VIOLATION_TOL = 1e-8

# Ellipsoid geometry
const NORMAL_SEMIAXIS = 1.0
const TANGENT_SEMIAXIS = 5.0
const EXTRA_SEMIAXIS = 2.0

# Initial point and Slater point distances from the common boundary point
const INITIAL_DISTANCE = 10.0
const SLATER_DISTANCE = 0.05

# Backtracking parameters
const ANGLE_MAX_BACKTRACKING = UInt(100)
const ANGLE_EPSILON_BACK = 1.0
const ANGLE_EPSILON_STOP = 1e-6
const ANGLE_BETA = 0.5

# Methods to be tested
const ANGLE_METHODS = [
    "Alt. Proj.",
    "MV Alt. Proj.",
    "Cimmino",
    "SCCRM",
    "OSCCRM",
    "BOSCCRM",
    "USCCRM",
    "BUSCCRM",
    "PACA.2",
    "PRSCCRM",
    "OPRSCCRM",
    "UPRSCCRM",
]

# -------------------------------------------------------------------------
# Construct one ellipsoid with a prescribed outward normal at the common
# boundary point.
# -------------------------------------------------------------------------

function ellipsoid_from_normal(
    normal::Vector{Float64},
    tangent::Vector{Float64},
    common_point::Vector{Float64},
)
    n = length(normal)
    Q = zeros(Float64, n, n)

    Q .+= (normal * normal') ./ NORMAL_SEMIAXIS^2
    Q .+= (tangent * tangent') ./ TANGENT_SEMIAXIS^2

    for j in 3:n
        Q[j, j] = 1.0 / EXTRA_SEMIAXIS^2
    end

    center = common_point .- NORMAL_SEMIAXIS .* normal
    radius = 1.0

    return center, Matrix(Symmetric(Q)), radius
end

# -------------------------------------------------------------------------
# Construct two ellipsoids meeting at the prescribed angle.
#
# Both boundaries contain common_point = 0.
# Their outward normal vectors at this point form angle_degrees.
#
# x_slater is strictly inside both ellipsoids.
# x0 is outside both ellipsoids along the external angle bisector.
# -------------------------------------------------------------------------

function generate_angle_instance(
    angle_degrees::Real;
    n::Int = AMBIENT_DIMENSION,
)
    if n < 2
        error("The ambient dimension must be at least 2.")
    end

    if !(0.0 < angle_degrees < 180.0)
        error("The angle must belong to (0, 180) degrees.")
    end

    #half_angle = deg2rad(angle_degrees / 2.0)
    # angle_degrees is now the internal angle of the feasible intersection.
normal_angle_degrees = 180.0 - angle_degrees
half_angle = deg2rad(normal_angle_degrees / 2.0)

    normal_1 = zeros(Float64, n)
    normal_2 = zeros(Float64, n)
    tangent_1 = zeros(Float64, n)
    tangent_2 = zeros(Float64, n)

    normal_1[1:2] .= [
        cos(half_angle),
        -sin(half_angle),
    ]

    normal_2[1:2] .= [
        cos(half_angle),
        sin(half_angle),
    ]

    tangent_1[1:2] .= [
        -normal_1[2],
        normal_1[1],
    ]

    tangent_2[1:2] .= [
        -normal_2[2],
        normal_2[1],
    ]

    common_point = zeros(Float64, n)

    center_1, Q_1, radius_1 = ellipsoid_from_normal(
        normal_1,
        tangent_1,
        common_point,
    )

    center_2, Q_2, radius_2 = ellipsoid_from_normal(
        normal_2,
        tangent_2,
        common_point,
    )

    y = zeros(Float64, 2, n)
    Q = zeros(Float64, 2, n, n)
    k = zeros(Float64, 2)

    y[1, :] .= center_1
    y[2, :] .= center_2

    Q[1, :, :] .= Q_1
    Q[2, :, :] .= Q_2

    k[1] = radius_1
    k[2] = radius_2

    external_bisector = normal_1 .+ normal_2
    external_bisector ./= norm(external_bisector)

    # x0 = common_point .+ INITIAL_DISTANCE .* external_bisector
    # new x0 begin:

    # Direction of x0 relative to the external bisector.
    # Positive/negative values move x0 to opposite sides.
    initial_direction_degrees = 70.0 

# Perpendicular direction in the plane of the first two coordinates.
side_direction = zeros(Float64, n)
side_direction[1] = -external_bisector[2]
side_direction[2] =  external_bisector[1]

phi = deg2rad(initial_direction_degrees)

start_direction = cos(phi) .* external_bisector .+
                  sin(phi) .* side_direction

# Increase the distance if necessary to keep x0 outside BOTH ellipsoids.
distance = INITIAL_DISTANCE
x0 = common_point .+ distance .* start_direction

for attempt in 1:40
    outside_both = all(
        g(x0, y[i, :], Q[i, :, :], k[i]) > 0.0
        for i in 1:2
    )
    outside_both && break

    attempt == 40 && error("Could not place x0 outside both ellipsoids.")
    distance *= 1.5
    x0 = common_point .+ distance .* start_direction
end

    #new x0 end 
    x_slater = common_point .- SLATER_DISTANCE .* external_bisector

    boundary_errors = [
        abs(g(common_point, y[i, :], Q[i, :, :], k[i]))
        for i in 1:2
    ]

    gradient_1 = ellipsoid_subgradient(
        common_point,
        y[1, :],
        Q[1, :, :],
    )

    gradient_2 = ellipsoid_subgradient(
        common_point,
        y[2, :],
        Q[2, :, :],
    )

    cosine_angle = dot(gradient_1, gradient_2) /
                   (norm(gradient_1) * norm(gradient_2))

    measured_angle = rad2deg(
        acos(clamp(cosine_angle, -1.0, 1.0)),
    )

    principal_angle = min(
        measured_angle,
        180.0 - measured_angle,
    )

    slater_violation = max_violation(
        x_slater,
        y,
        Q,
        k,
        0.0,
    )

    initial_violation = max_violation(
        x0,
        y,
        Q,
        k,
        0.0,
    )

    if maximum(boundary_errors) > 1e-10
        error("The common point is not on both ellipsoid boundaries.")
    end

    if slater_violation >= 0.0
        error("The constructed instance does not satisfy strict Slater.")
    end

    if initial_violation <= 0.0
        error("The initial point is not outside the intersection.")
    end

    return (
        x0 = x0,
        x_slater = x_slater,
        common_point = common_point,
        y = y,
        Q = Q,
        k = k,
        prescribed_angle = Float64(angle_degrees),
        measured_angle = measured_angle,
        principal_angle = principal_angle,
        slater_violation = slater_violation,
        initial_violation = initial_violation,
    )
end

# -------------------------------------------------------------------------
# Call the methods implemented in cutting_planes_with_finite.jl.
# -------------------------------------------------------------------------

function call_angle_method(
    method::String,
    x0,
    y,
    Q,
    k,
    env;
    max_iter::UInt = MAX_ITER,
    time_limit::Real = TIME_LIMIT,
)
    if method == "Alt. Proj."
        return solve_alt_proj(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "MV Alt. Proj."
        return solve_mv_alt_proj(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "Cimmino"
        return solve_cimmino(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "SCCRM"
        return solve_sc_crm(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "OSCCRM"
        return solve_sc_crm_over(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "BOSCCRM"
        return solve_sc_crm_over_back(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL,
            ANGLE_MAX_BACKTRACKING,
            ANGLE_EPSILON_BACK,
            ANGLE_EPSILON_STOP,
            ANGLE_BETA,
            time_limit,
        )

    elseif method == "USCCRM"
        return solve_sc_crm_under(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "BUSCCRM"
        return solve_sc_crm_under_back(
            x0, y, Q, k, max_iter, env,
            VIOLATION_TOL,
            ANGLE_MAX_BACKTRACKING,
            ANGLE_EPSILON_BACK,
            ANGLE_EPSILON_STOP,
            ANGLE_BETA,
            time_limit,
        )

    elseif method == "PACA.2"
        return solve_paca2(
            x0, y, Q, k, max_iter,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "PRSCCRM"
        return solve_prsccrm(
            x0, y, Q, k, max_iter,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "OPRSCCRM"
        return solve_oprsccrm(
            x0, y, Q, k, max_iter,
            VIOLATION_TOL, time_limit,
        )

    elseif method == "UPRSCCRM"
        return solve_uprsccrm(
            x0, y, Q, k, max_iter,
            VIOLATION_TOL, time_limit,
        )
    end

    error("Unknown method: $method")
end

# -------------------------------------------------------------------------
# Run one method and determine its final status.
# -------------------------------------------------------------------------

function run_angle_method(method, instance, env)
    try
        x, violation_history, elapsed_time, iterations =
            call_angle_method(
                method,
                instance.x0,
                instance.y,
                instance.Q,
                instance.k,
                env,
            )

        final_violation = max_violation(
            x,
            instance.y,
            instance.Q,
            instance.k,
            VIOLATION_TOL,
        )

        # Check original constraints after the solver has returned
original_residuals = [
    dot(
        x - instance.y[i, :],
        instance.Q[i, :, :] * (x - instance.y[i, :]),
    ) - instance.k[i]^2
    for i in axes(instance.y, 1)
]

valid_original = all(isfinite, x) &&
                 !isempty(original_residuals) &&
                 all(isfinite, original_residuals)

original_violation = valid_original ?
                     maximum(original_residuals) : NaN

original_intersection = !valid_original ? "NOT_CHECKED" :
                        original_violation <= 0.0 ? "YES" : "NO"

println("$method: original_violation = $original_violation")
println("  original_intersection = $original_intersection")

        status =
            final_violation <= 0.0 ? "OK" :
            elapsed_time >= 0.99 * TIME_LIMIT ? "TIME_LIMIT" :
            iterations >= Int(MAX_ITER) - 1 ? "MAX_ITER" :
            "STOPPED"

        return (
            method = method,
            time = elapsed_time,
            iterations = iterations,
            violation = final_violation,
            status = status,
        )
    catch err
        println("$method failed: ", err)

        return (
            method = method,
            time = NaN,
            iterations = 0,
            violation = NaN,
            status = "FAILED",
        )
    end
end

# -------------------------------------------------------------------------
# Plot the equivalent two-dimensional geometry.
#
# -------------------------------------------------------------------------

function plot_angle_geometry(case_name, angle_degrees)
    instance = generate_angle_instance(angle_degrees; n = 2)

    y = instance.y
    Q = instance.Q
    k = instance.k

    plot_limit = max(
        2.0,
        maximum(abs.(y)),
        maximum(abs.(instance.x0)),
        TANGENT_SEMIAXIS,
    ) + 1.0

    grid = range(
        -plot_limit,
        plot_limit,
        length = 500,
    )

    figure = plot(
        aspect_ratio = :equal,
        xlabel = "x₁",
        ylabel = "x₂",
        title = "Tested angle = $(angle_degrees)°",
        legend = :topright,
    )

    colors = [:blue, :red]

    for i in 1:2
        contour!(
            figure,
            grid,
            grid,
            (a, b) -> g(
                [a, b],
                y[i, :],
                Q[i, :, :],
                k[i],
            ),
            levels = [0.0],
            linewidth = 2,
            color = colors[i],
            label = "Ellipsoid $i",
        )
    end

    scatter!(
        figure,
        [instance.common_point[1]],
        [instance.common_point[2]],
        color = :black,
        marker = :star5,
        markersize = 7,
        label = "Common boundary point",
    )

    scatter!(
        figure,
        [instance.x_slater[1]],
        [instance.x_slater[2]],
        color = :green,
        markersize = 5,
        label = "Strict Slater point",
    )

    scatter!(
        figure,
        [instance.x0[1]],
        [instance.x0[2]],
        color = :orange,
        markersize = 6,
        label = "Initial point",
    )

    output_file = joinpath(
        @__DIR__,
        "ellipsoid_angle_$(case_name).pdf",
    )

    savefig(figure, output_file)
end

# -------------------------------------------------------------------------
# Main experiment.
# -------------------------------------------------------------------------

function main_angle_experiment()
    env = Gurobi.Env()
    results = NamedTuple[]

    println("Num threads: ", Threads.nthreads())
    println("No warm-up is being performed.")
    println("Angles to be tested: ", last.(ANGLE_CASES))

    for (case_name, angle_degrees) in ANGLE_CASES
        instance = generate_angle_instance(angle_degrees)

        println()
        println("============================================================")
        println("Case                   = $case_name")

        @printf(
            "Tested angle           = %.2f degrees\n",
            instance.prescribed_angle,
        )

        @printf(
            "Measured normal angle  = %.8f degrees\n",
            instance.measured_angle,
        )

        @printf(
            "Principal angle        = %.8f degrees\n",
            instance.principal_angle,
        )

        @printf(
            "Slater violation       = %.6e\n",
            instance.slater_violation,
        )

        @printf(
            "Initial violation      = %.6e\n",
            instance.initial_violation,
        )

        println("============================================================")

        for method in ANGLE_METHODS
            result = run_angle_method(
                method,
                instance,
                env,
            )

            @printf(
    "angle = %6.2f, %-14s time = %10.4f s, iter = %7d, violation = %12.4e, status = %s\n",
    instance.prescribed_angle,
    result.method,
    result.time,
    result.iterations,
    result.violation,
    result.status,
)

            push!(
                results,
                (
                    case = case_name,
                    prescribed_angle = instance.prescribed_angle,
                    measured_angle = instance.measured_angle,
                    principal_angle = instance.principal_angle,
                    method = result.method,
                    time = result.time,
                    iterations = result.iterations,
                    violation = result.violation,
                    status = result.status,
                ),
            )
        end

        plot_angle_geometry(
            case_name,
            angle_degrees,
        )
    end

    output_csv = joinpath(
        @__DIR__,
        "ellipsoid_angle_results.csv",
    )

    open(output_csv, "w") do io
        println(
            io,
            "case,prescribed_angle,measured_angle," *
            "principal_angle,method,time,iterations,violation,status",
        )

        for row in results
            println(
                io,
                "$(row.case)," *
                "$(row.prescribed_angle)," *
                "$(row.measured_angle)," *
                "$(row.principal_angle)," *
                "$(row.method)," *
                "$(row.time)," *
                "$(row.iterations)," *
                "$(row.violation)," *
                "$(row.status)",
            )
        end
    end

    println()
    println("Results saved to: $output_csv")
    println("Geometry plots saved in: $(@__DIR__)")
end

function plot_all_angle_geometries_v2()
   # angles = [15.0, 45.0, 90.0, 135.0, 165.0]
   angles = last.(ANGLE_CASES)
    colors = [:teal, :darkorange]
    full_panels, zoom_panels = [], []
    theta = range(0, 2pi; length = 1000)
    circle = [cos.(theta)'; sin.(theta)']

    for angle in angles
        inst = generate_angle_instance(angle; n = 2)
        p = inst.common_point
        curves, gradients = [], []
        for i in 1:2
            Qi, center = inst.Q[i,:,:], inst.y[i,:]
            F = eigen(Symmetric(Qi))
            curve = center .+ F.vectors *
                Diagonal(inst.k[i] ./ sqrt.(F.values)) * circle
            push!(curves, curve)
            push!(gradients, 2 .* Qi * (p - center))
        end

        cosine = dot(gradients[1], gradients[2]) /
                 (norm(gradients[1]) * norm(gradients[2]))
        alpha = 180.0 - rad2deg(acos(clamp(cosine, -1.0, 1.0)))
        @assert isapprox(alpha, angle; atol = 1e-8)
        extent = max(maximum(abs.(inst.x0)),
                     maximum(maximum(abs.(c)) for c in curves))

        for (zoom, panels) in ((false, full_panels), (true, zoom_panels))
            L = zoom ? 0.8 * NORMAL_SEMIAXIS : 1.2 * extent
            heading = zoom ? "Tangent construction" : "α = $(Int(angle))°"
            fig = plot(; aspect_ratio=:equal, xlims=(-L,L), ylims=(-L,L),
                title=heading, titlefontsize=11, legend=false,
                xlabel="x₁", ylabel="x₂", tickfontsize=7, gridalpha=0.12)

            for i in 1:2
                c = curves[i]
                plot!(fig, c[1,:], c[2,:]; color=colors[i], linewidth=2)
                # Each tangent is perpendicular to the gradient at p
                normal = normalize(gradients[i])
                tangent = [-normal[2], normal[1]]
                s = zoom ? [-2L, 2L] : [-0.25L, 0.25L]
                plot!(fig, p[1] .+ s .* tangent[1],
                    p[2] .+ s .* tangent[2]; color=colors[i],
                    linestyle=:dash, linewidth=1.5)
            end

            # The feasible wedge opens toward the negative x₁ direction
            t = range(pi-deg2rad(alpha)/2, pi+deg2rad(alpha)/2; length=100)
            r = zoom ? 0.30L : 0.10L
            plot!(fig, p[1] .+ r*cos.(t), p[2] .+ r*sin.(t);
                color=:black, linewidth=2)
            annotate!(fig, p[1]-1.65r, p[2], text("α", 14))

            if !zoom
                scatter!(fig, [inst.x0[1]], [inst.x0[2]];
                    color=:black, markersize=4)
                annotate!(fig, inst.x0[1], inst.x0[2]+0.12L,
                    text("Initial point", 9))
            end
            push!(panels, fig)
        end
    end

    figure = plot(full_panels..., zoom_panels...;
    layout = (2, length(angles)),
    size = (400 * length(angles), 850),
    dpi = 300,
)
        #layout=(2,5), size=(2000,850), dpi=300)
        #layout=(2,length(angles)), size=(400*length(angles),850), dpi=300)
    savefig(figure, joinpath(@__DIR__, "ellipsoids_five_anglesv2.png"))
    savefig(figure, joinpath(@__DIR__, "ellipsoids_five_anglesv2.pdf"))
    display(figure)
    return figure
end

main_angle_experiment()

# To run this experiment, comment out the "main()" line in cutting_planes_with_finite.jl and run it using "if abspath(PROGRAM_FILE) == @__FILE__ main() end"
