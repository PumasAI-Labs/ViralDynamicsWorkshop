############################################################################################
# Workshop: Neural-Embedded NLME Modeling with DeepPumas
# Goal: Generate synthetic PK/PD data → Fit neural-embedded NLME model → Validate predictions
############################################################################################

############################
# 0) Environment & Packages
############################

using DeepPumas            # For neural-embedded NLME modeling
using CairoMakie           # Plotting backend
using StableRNGs           # Stable random number generator for reproducibility
using PumasPlots           # Convenient plotting utilities for Pumas
set_mlp_backend(:staticflux)  # Use StaticFlux backend for neural networks
set_theme!(deep_light())       # Set a light theme for plots (try deep_dark() for dark mode)

############################################################################################
# 1) Generate synthetic data from an Indirect Response (IDR) model
############################################################################################

## Define the data-generating model (true underlying PK/PD system)
datamodel = @model begin
    @param begin
        # PK parameters
        tvKa ∈ RealDomain()
        tvCL ∈ RealDomain()
        tvVc ∈ RealDomain()
        # PD parameters
        tvSmax ∈ RealDomain()
        tvn ∈ RealDomain()
        tvSC50 ∈ RealDomain()
        tvKout ∈ RealDomain()
        tvKin ∈ RealDomain()
        # Variability and residual error
        Ω ∈ PDiagDomain(5)
        σ ∈ RealDomain()      # PD residual error
        σ_pk ∈ RealDomain()   # PK residual error
    end

    @random begin
        η ~ MvNormal(Ω)       # Random effects for inter-individual variability
    end

    @pre begin
        # Apply random effects to parameters
        Smax = tvSmax * exp(η[1])
        SC50 = tvSC50 * exp(η[2])
        Ka   = tvKa   * exp(η[3])
        Vc   = tvVc   * exp(η[4])
        Kout = tvKout * exp(η[5])
        Kin  = tvKin
        CL   = tvCL
        n    = tvn
    end

    @init begin
        # Initial response value based on Kin/Kout
        R = Kin / Kout
    end

    @vars begin
        # PK concentration (ensure non-negative)
        _cp = max(Central / Vc, 0.)
        # Drug effect using Emax model
        EFF = Smax * _cp^n / (SC50^n + _cp^n)
    end

    @dynamics begin
        # PK compartments
        Depot'   = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        # PD compartment (indirect response)
        R' = Kin * (1 + EFF) - Kout * R
    end

    @derived begin
        # Observations: PK and PD
        cp ~ @. Normal(Central / Vc, σ_pk)
        dv ~ @. Normal(R, σ)
    end
end

## Parameter values for synthetic data generation
p_data = (;
    tvKa    = 0.5,
    tvCL    = 1.,
    tvVc    = 1.,
    tvSmax  = 2.9,
    tvn     = 1.5,
    tvSC50  = 0.05,
    tvKout  = 2.2,
    tvKin   = 0.8,
    Ω       = Diagonal(fill(0.1, 5)),
    σ       = 0.1,    # PD noise
    σ_pk    = 0.02    # PK noise
)

## Simulation settings
obstimes = 0:24       # Observation times (hours)
ntrain   = 10         # Number of training subjects
ntest    = 12         # Number of test subjects

## Generate synthetic population with random dosing regimens
pop = map(1:ntrain + ntest) do i
    rng = StableRNG(i)  # Ensure reproducibility per subject
    dose_1 = DosageRegimen(1.)  # First dose
    dose_2 = DosageRegimen(1.; time = rand(rng, Gamma(40, 5/40)))  # Random second dose
    sim = simobs(
        datamodel,
        Subject(; id = i, events = DosageRegimen(dose_1, dose_2)),
        p_data;
        obstimes,
        rng
    )
    Subject(sim)
end

## Split into training and test sets
trainpop = pop[1:ntrain]
testpop  = pop[(ntrain+1):end]

## Visualize synthetic data and predictions from the data-generating model
pred_datamodel = predict(datamodel, testpop, p_data; obstimes = 0:0.1:24)
plotgrid(pred_datamodel; observation = :cp)  # PK profiles
plotgrid(pred_datamodel; observation = :dv)  # PD profiles

############################################################################################
# 2) Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD dynamics are learned by a neural network (MLP).
# The NN takes PK states and individual parameters as inputs and predicts PD change.

model = @model begin
    @param begin
        # Neural network: maps 5 inputs → 1 output
        # Inputs: 2 state variables + 3 individual parameters
        NN ∈ MLPDomain(5, 7, 7, (1, identity); reg = L2(1.0))  # L2 regularization
        # PK parameters
        tvKa ∈ RealDomain(; lower = 0)
        tvCL ∈ RealDomain(; lower = 0)
        tvVc ∈ RealDomain(; lower = 0)
        # PD initial condition
        tvR₀ ∈ RealDomain(; lower = 0)
        ωR₀ ∈ RealDomain(; lower = 0)
        # Variability and residual error
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(; lower = 0)
        σ_pk ∈ RealDomain(; lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)          # Random effects for PK
        η_nn ~ MvNormal(3, 0.1)  # Random effects for NN inputs
    end

    @pre begin
        # PK parameters with random effects
        Ka = tvKa * exp(η[1])
        Vc = tvVc * exp(η[2])
        CL = tvCL

        # Initial PD state depends on random effect
        R₀ = tvR₀ * exp(10 * ωR₀ * η_nn[1])

        # Fix random effects as NN inputs → individual neural network
        iNN = fix(NN, η_nn)
    end

    @init begin
        R = R₀
    end

    @dynamics begin
        # PK compartments
        Depot'   = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        # PD dynamics predicted by NN
        R' = iNN(Central / Vc, R)[1]
    end

    @derived begin
        cp ~ @. Normal(Central / Vc, σ_pk)
        dv ~ @. Normal(R, σ)
    end
end

############################################################################################
# 3) Fit the neural-embedded model
############################################################################################

fpm = fit(
    model,
    trainpop,
    init_params(model),
    MAP(FOCE());  # MAP estimation with FOCE
    optim_options = (; iterations = 200, f_tol = 1e-6)  # Speed up demo
)

# Predict on training data
pred_traindata = predict(fpm; obstimes = 0:0.1:24)
plotgrid(pred_traindata; observation = :dv)

# Inspect goodness-of-fit
ins = inspect(fpm)
goodness_of_fit(ins; observations = [:dv])

############################################################################################
# 4) Validate on test data
############################################################################################

pred_test = predict(model, testpop, coef(fpm); obstimes = 0:0.1:24)
plotgrid(pred_test; ylabel = "Outcome (Test data)", observation = :dv)

############################################################################################
# 5) Out-of-sample dosing scenario
############################################################################################

# New regimen: three low doses, then two high doses
dr2 = DosageRegimen(0.3, ii = 3, addl = 2)
dr3 = DosageRegimen(1.5, time = 25, ii = 8, addl = 1)

# Generate synthetic data for new regimen
testpop2 = synthetic_data(datamodel, DosageRegimen(dr2, dr3), p_data; nsubj = 12, obstimes = 0:2:48)

# Predict with neural-embedded model
pred2 = predict(model, testpop2, coef(fpm); obstimes = 0:0.01:48)
plotgrid(pred2; 
        observation = :dv,
        pred        = (; label = "neuralModel pred"),
        ipred       = (; label = "neuralModel ipred")
        )

# Overlay true data-generating model predictions
pred_truth = predict(datamodel, testpop2, p_data; obstimes = 0:0.01:48)
plotgrid!(pred_truth; pred = false, ipred = (; color = Cycled(3), label = "DataModel ipred"), observation = :dv)

############################################################################################
# 6) Compare estimated vs true parameters
############################################################################################

# println("Estimated vs True PK parameters:")
println("Ka: ", coef(fpm).tvKa, " vs ", p_data.tvKa)
println("CL: ", coef(fpm).tvCL, " vs ", p_data.tvCL)
println("Vc: ", coef(fpm).tvVc, " vs ", p_data.tvVc)
println("σ: ", coef(fpm).σ, " vs ", p_data.σ)
println("σ_pk: ", coef(fpm).σ_pk, " vs ", p_data.σ_pk)



############################################################################################
# Exercises for participants:
#=
Exercises:

Explore freely, but if you want some ideas for what you can look at then here's a list

- How many subjects do you need for training? Re-train with different numbers of training
  subjects and see how the model performs on test data. Can you make the model better than 
  it is here? Can you break it?
  
- How noise-sensitive is this? Increase the noisiness (σ) in your training and
  test data and re-fit. Can you compensate for noise with a larger training
  population?

- Is there some out-of-sample dose regimen that the model fails for? Why?
  
- Rewrite the model to include more knowledge. Flesh out the indirect response model and let
  the NN capture only what's called EFF in the data generating model. You can stop using R as
  an input but you'll need to change the MLPDomain definition for that.
  
- Change the number of random effects that's passed to the neural network. What happens if
  the DeepNLME model has fewer random effects than the data generating model? What happens if
  it has more?
=#