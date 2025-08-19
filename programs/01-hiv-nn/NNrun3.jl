# ----------------------------------------------------- #
# Four NN for characterizing :
### 2 different types of infection
### fraction between infection types
### production of HIV #
# ----------------------------------------------------- #


## Libraries

using Dates
using CairoMakie
using DataFramesMeta
using PharmaDatasets
using CSV
using Pumas
using PumasUtilities
using DataFrames
using Random
using Statistics
using CategoricalArrays
using Chain
using BenchmarkTools
using AlgebraOfGraphics
using SummaryTables
using XLSX
using DeepPumas
using Serialization
using Flux: softmax

# Set working directory to the script's location
cd(@__DIR__)

# ---------------------------
# 1. Load and Prepare Data
# ---------------------------

# Read the simulated population data
pdData = CSV.read(joinpath(@__DIR__, "hiv-ipp-datax.csv"), DataFrame; missingstring = "", stringtype = String)

# Create a Pumas Population object from the DataFrame
_pop = read_pumas(
    pdData;
    id = :id,
    time = :time,
    observations = [:DV],
    evid = :evid,
    amt = :amt,
    cmt = :cmt,
    covariates = [:xKa, :xCL, :xVc, :xQ, :xVp, :xDur],
)

# Split into training and validation populations
_tpop = _pop[1:40]   # Training set: first 40 subjects
_vpop = _pop[41:end] # Validation set: remaining subjects

# Visualize the first 8 training subjects
plotgrid(_tpop[1:8]; data = (; color=:blue))

# ---------------------------
# 2. Define the HIV PKPD Model
# ---------------------------

nn_hiv = @model begin
    @metadata begin
        desc = "HIV PD Only"
        timeu = u"d" # day
    end
    @param begin
        # Neural networks for different mechanistic components
        NN1 ∈ MLPDomain(5, 7, 7, (1, identity); reg = L2(1.0))
        NN2 ∈ MLPDomain(5, 3, 3, (1, identity); reg = L2(1.0))
        NN3 ∈ MLPDomain(3, 3, 3, (1, identity); reg = L2(1.0))
        NN4 ∈ MLPDomain(3, 3, 3, (1, identity); reg = L2(1.0))

        # PKPD parameters
        tvR0   ∈ RealDomain(lower=1.0, init=6.5)
        tvλ    ∈ RealDomain(lower=0., init=0.46236)
        tvd    ∈ RealDomain(lower=0., init=0.006)
        tvic50 ∈ RealDomain(lower=0., init=200.0)
        tvap    ∈ RealDomain(lower=0., init=0.63911)
        tvas    ∈ RealDomain(lower=0., init=0.63911)
        tvu    ∈ RealDomain(lower=0., init=0.011)
        Ω      ∈ PDiagDomain(4)
        ωR0    ∈ RealDomain(lower=0.)
        σ_add  ∈ RealDomain(lower=0.)
    end
    @random begin
        η ~ MvNormal(Ω)
        η_nn ~ MvNormal(2, 0.1)
    end
    @covariates xKa xCL xVc xQ xVp xDur
    @pre begin
        # Assign covariates to PK parameters
        Ka = xKa
        CL = xCL
        Vc = xVc
        Q  = xQ
        Vp = xVp

        # PD parameters with random effects
        R0 = tvR0 * exp(10 * ωR0 * η_nn[1])
        λ  = tvλ * exp(η[1])
        d  = tvd
        ap  = tvap * exp(η[2])
        as  = tvas 
        u  = tvu * exp(η[3])
        IC50 = tvic50 * exp(η[4])

        # Initial conditions for T and V
        T0 = λ / d / R0
        V0 = λ / u * (1 - 1 / R0)

        # Fix neural networks with random effects
        iNN_INF   = fix(NN1, η_nn)
        iNN_GATE  = fix(NN2, η_nn)
        iNN_PRODP = fix(NN3, η_nn)
        iNN_PRODS = fix(NN4, η_nn)

        # Gate for initial infected populations
        z0   = iNN_GATE(T0, V0, 1.0)[1]
        f_s0 = 1.0 / (1.0 + exp(-z0))
        f_p0 = 1.0 - f_s0
    end
    @dosecontrol begin
        duration = (; Depot = xDur)
    end
    @init begin
        # Initial conditions for HIV dynamics
        T  = λ / d / R0
        Ip = f_p0 * λ / ap * (1 - (1 / R0))
        Is = f_s0 * λ / as * (1 - (1 / R0))
        V  = λ / u * (1 - (1 / R0))
    end
    @vars begin
        # Calculate concentration and inhibition
        Conc = Central / Vc / 1000
        INH = Conc / (Conc + IC50)
        β = R0 * d / λ / V

        # Gate for infected populations
        zgate = iNN_GATE(T, V, (1.0 - INH))[1]
        f_s  = 1.0 / (1.0 + exp(-zgate))   # sigmoid
        f_p  = 1.0 - f_s

        # Infection and production terms
        infect_term = iNN_INF(T, V, (1 - INH))[1]
        prodp = iNN_PRODP(Ip)[1]
        prods = iNN_PRODS(Is)[1]
    end
    @dynamics begin
        # PK: Two-compartment model with first-order absorption
        Depot'   = -Ka * Depot
        Central' = Ka * Depot + (Q / Vp) * Periph - (Q / Vc) * Central - CL / Vc * Central
        Periph'  = -(Q / Vp) * Periph + (Q / Vc) * Central

        # HIV dynamics
        T'  = λ - d * T - infect_term
        Ip' = f_p * infect_term - ap * Ip
        Is' = f_s * infect_term - as * Is
        V'  = prodp + prods - u * V
    end
    @derived begin
        Concentration = Conc
        Virus = V
        DV ~ @. Normal(Virus, σ_add)
    end
end

# ---------------------------
# 3. Fit the Model
# ---------------------------

fit_pkpd = fit(
    nn_hiv,
    _tpop,
    init_params(nn_hiv),
    JointMAP(),
    optim_options = (; iterations=100, f_tol=1e-6),
)

# Save and reload the fit for reproducibility
serialize("run3_nn.jls", fit_pkpd)
mdl_fit = deserialize("run3_nn.jls")

# ---------------------------
# 4. Diagnostics and Goodness-of-Fit
# ---------------------------

# Inspect the fitted model
mdl_insp = inspect(fit_pkpd)

# Overall goodness-of-fit summary
goodness_of_fit(mdl_insp)

# Key diagnostic plots
observations_vs_ipredictions(mdl_insp)
# Additional diagnostics (uncomment as needed)
# observations_vs_predictions(mdl_insp)
# wresiduals_vs_time(mdl_insp)
# wresiduals_vs_predictions(mdl_insp)
# wresiduals_dist(mdl_insp)
# empirical_bayes_dist(mdl_insp)

# ---------------------------
# 5. Predict on Validation Set and Plot
# ---------------------------

# Predict on validation population for days 0 to 42
model_pred = predict(fit_pkpd, _vpop; obstimes=0:1:42)

# Plot predictions for the first 8 validation subjects
plotgrid(
    model_pred[1:8],
    observation = :DV,
    pred = (; label = "model pred", linestyle=:dash),
    ipred = (; label = "model ipred"),
    axis = (; limits = ((0., 45.), nothing))
)

# Plot predictions for validation subjects 20 to 31
plotgrid(
    model_pred[20:31],
    observation = :DV,
    pred = (; label = "model pred", linestyle=:dash),
    ipred = (; label = "model ipred"),
    axis = (; limits = ((0., 45.), nothing))
)