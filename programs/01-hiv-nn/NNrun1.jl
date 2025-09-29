# --------------------------- #
# One NN for HIV RNA dynamics #
# --------------------------- #

## Libraries

using Dates
using CairoMakie
using DataFramesMeta
using PharmaDatasets
using CSV
using Pumas
using Random
using Statistics
using CategoricalArrays
using Chain
using AlgebraOfGraphics
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
#    One NN - for HIV RNA dynamics (simplest hiv model)
# ---------------------------

nn_hiv = @model begin
    @metadata begin
        desc = "HIV PD Only"
        timeu = u"d" # day
    end
    @param begin
        # Neural network for PD dynamics
        NN ∈ MLPDomain(5, 7, 7, (1, identity); reg = L2(1.0))
        # Baseline virus production rate
        tvR0 ∈ RealDomain(lower=0., init=6.5)
        # Inter-individual variability on R0
        ωR0 ∈ RealDomain(lower=0.)
        # Additive error for virus measurement
        σ_add ∈ RealDomain(lower=0.)
    end
    @random begin
        # Random effects for the neural network
        η_nn ~ MvNormal(3, 0.1)
    end
    # Covariates for PK parameters and absorption duration
    @covariates xKa xCL xVc xQ xVp xDur
    @pre begin
        # Assign covariates to PK parameters
        Ka = xKa
        CL = xCL
        Vc = xVc
        Q  = xQ
        Vp = xVp
        # Baseline virus production with random effect
        R0 = tvR0 * exp(10 * ωR0 * η_nn[1])
        # Fix the neural network with random effects
        iNN = fix(NN, η_nn)
    end
    @dosecontrol begin
        # Set absorption duration for Depot compartment
        duration = (; Depot = xDur)
    end
    @init begin
        # Initial virus amount
        R = R0
    end
    @vars begin
        # Calculate concentration in central compartment
        Conc = Central / Vc
    end
    @dynamics begin
        # PK: Two-compartment model with first-order absorption
        Depot'   = -Ka * Depot
        Central' = Ka * Depot + (Q/Vp) * Periph - (Q/Vc) * Central - (CL/Vc) * Central
        Periph'  = -(Q/Vp) * Periph + (Q/Vc) * Central
        # PD: Virus dynamics via neural network
        R' = iNN(Conc, R)[1]
    end
    @derived begin
        # Output concentration in ng/mL
        Concentration = Conc * 1000
        # Output virus (could be log-transformed if desired)
        Virus = R
        # Observed variable with additive error
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
    # JointMAP(),  # Use JointMAP for mixed effects and neural network
    MAP(FOCE());
    optim_options = (; iterations=100),
)

# Save and reload the fit for reproducibility
serialize("run1_nn.jls", fit_pkpd)
mdl_fit = deserialize("run1_nn.jls")

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
    figure = (; resolution=(700, 700)),
    pred = (; label = "model pred", linestyle=:dash),
    ipred = (; label = "model ipred"),
    axis = (; limits = ((0., 42.), nothing))
)