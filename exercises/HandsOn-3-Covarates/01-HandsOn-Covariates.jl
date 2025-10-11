############################################################################################
# Workshop: Advanced Neural-Embedded NLME Modeling with Covariates
# Goal: Generate synthetic PK/PD data with complex covariates → Fit neural-embedded NLME model
#       → Augment with ML for covariate prediction → Refine Ω estimates → Explore joint fitting
############################################################################################

############################
# 0) Environment & Packages
############################

using DeepPumas            # Neural-embedded NLME modeling
using StableRNGs           # Stable RNG for reproducibility
using CairoMakie           # Plotting backend
using Serialization         # Save/load fitted models
using Latexify              # Render model equations in LaTeX
using PumasUtilities        # Utilities for Pumas workflows

# Backend for neural networks and plotting theme
set_mlp_backend(:staticflux)
set_theme!(deep_light())

# Set working directory and artifacts folder
cd(@__DIR__)
ASSETS_DIR = joinpath(@__DIR__, "assets")

############################################################################################
## Generate synthetic data from an indirect response model (IDR) with complicated covariates
############################################################################################

## Define the data-generating model
datamodel = @model begin
  @param begin
    tvKa ∈ RealDomain(; lower=0, init=0.5)
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvSmax ∈ RealDomain(; lower=0, init=0.9)
    tvn ∈ RealDomain(; lower=0, init=1.5)
    tvSC50 ∈ RealDomain(; lower=0, init=0.2)
    tvKout ∈ RealDomain(; lower=0, init=1.2)
    Ω ∈ PDiagDomain(; init=fill(0.05, 5))
    σ ∈ RealDomain(; lower=0, init=5e-2)
    σ_pk ∈ RealDomain(; lower=0, init=5e-2)
  end
  @random begin
    η ~ MvNormal(Ω)
  end
  @covariates R_eq c1 c2 c3 c4 c5 c6
  @pre begin
    # Complex covariate relationships influencing PK/PD parameters
    Smax = tvSmax * exp(η[1]) + 3 * c1 / (12.0 + c1) # Additional nonlinear covariate effect
    SC50 = tvSC50 * exp(η[2] + 0.2 * (c2 / 20)^0.75)
    Ka = tvKa * exp(η[3] + 0.3 * c3 * c4)
    Vc = tvVc * exp(η[4] + 0.3 * c3)
    Kout = tvKout * exp(η[5] + 0.3 * c5 / (c6 + c5))
    Kin = R_eq * Kout
    CL = tvCL
    n = tvn
  end
  @init begin
    R = Kin / Kout
  end
  @vars begin
    _cp = max(Central / Vc, 0.0)
    EFF = Smax * _cp^n / (SC50^n + _cp^n)
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + EFF) - Kout * R
  end
  @derived begin
    cp ~ @. Normal(Central/Vc, σ_pk)
    dv ~ @. Normal(R, σ)
  end
end

# Render LaTeX representation of @pre block for documentation
render(latexify(datamodel, :pre))

## Generate synthetic data.
p_data = (;
  tvKa=0.5,
  tvCL=1.0,
  tvVc=1.0,
  tvSmax=1.2,
  tvn=1.5,
  tvSC50=0.02,
  tvKout=2.2,
  Ω=Diagonal(fill(0.05, 5)),
  σ=0.1,                         ## <-- tune the observational noise of the data here
  σ_pk=0.02                      ## <-- tune the observational noise of the data here
)

# Dosing regimen: one dose + additional doses
dr = DosageRegimen(0.5, ii=8, addl=1)

# Generate population with covariates drawn from distributions
pop = synthetic_data(
  datamodel,
  dr,
  p_data;
  covariates=(;
    R_eq=Gamma(50, 1 / (50)),
    c1=Gamma(5, 2),
    c2=Gamma(21, 1),
    c3=Normal(),
    c4=Normal(),
    c5=Gamma(11, 1),
    c6=Gamma(11, 1)
  ),
  nsubj=1020,
  rng=StableRNG(123),
  obstimes=0:2:24
)

# Visualize covariate distributions
covariates_dist(pop)

# Split the data into different training/test populations
trainpop_small = pop[1:50]
trainpop_large = pop[1:1000]
testpop = pop[1001:end]

# Visualize predictions from data-generating model
pred_datamodel = predict(datamodel, testpop, p_data; obstimes=0:0.1:24)
plotgrid(pred_datamodel; observation=:cp)
plotgrid(pred_datamodel; observation=:dv)

############################################################################################
## Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD is entirely determined by a neural network.
# At this point, we're not trying to explain how patient data may inform individual
# parameters.

model = @model begin
  @param begin
    # Define a multi-layer perceptron (a neural network) which maps from 5 inputs
    # (2 state variables + 3 individual parameters) to a single output.
    # Apply L2 regularization (equivalent to a Normal prior).
    NN ∈ MLPDomain(5, 7, 7, (1, identity); reg = L2(1.0))
    tvKa ∈ RealDomain(; lower=0)
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvR₀ ∈ RealDomain(; lower=0)
    ωR₀ ∈ RealDomain(; lower=0)
    Ω ∈ PDiagDomain(2)
    Ω_nn ∈ PDiagDomain(3)
    σ ∈ RealDomain(; lower=0)
    σ_pk ∈ RealDomain(; lower=0)
  end
  @random begin
    η ~ MvNormal(Ω)
    η_nn ~ MvNormal(Ω_nn)
  end
  @pre begin
    Ka = tvKa * exp(η[1])
    Vc = tvVc * exp(η[2])
    CL = tvCL

    # Letting the initial value of R depend on a random effect enables
    # its identification from observations. Note how we're using this 
    # random effect in both R₀ and as an input to the NN.
    # This is because the same information might be useful for both
    # determining the initial value and for adjusting the dynamics.
    R₀ = tvR₀ * exp(10 * ωR₀ * η_nn[1])

    # Fix random effects as non-dynamic inputs to the NN and return an "individual"
    # neural network:
    iNN = fix(NN, η_nn)
  end
  @init begin
    R = R₀
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = iNN(Central / Vc, R)[1]
  end
  @derived begin
    cp ~ @. Normal(Central/Vc, σ_pk)
    dv ~ @. Normal(R, σ)
  end
end

# Fit the neural-embedded model
fpm = fit(
  model,
  trainpop_small,
  init_params(model),
  MAP(FOCE());
  optim_options=(; iterations=200),
  constantcoef = (:Ω_nn,)
)

# Save and reload fit for reproducibility
serialize(joinpath(ASSETS_DIR, "deep_pumas_fpm.jls"), fpm)
# fpm = deserialize(joinpath(ASSETS_DIR, "deep_pumas_fpm.jls"))

# Predict on test population
pred = predict(fpm, testpop; obstimes=0:0.1:24)
plotgrid(pred; observation=:cp)
plotgrid(pred; observation=:dv)

############################################################################################
## 'Augment' the model to predict heterogeneity from data
############################################################################################
# All patient heterogeneity of our recent model was captured by random effects and can thus
# not be predicted by the model. Here, we 'augment' that model with ML that's trained to 
# capture this heterogeneity from data.

# Generate a target for the ML fitting from a Normal approximation of the posterior η
# distribution.
target = preprocess(fpm)

nn = MLPDomain(numinputs(target), 9, 9, (numoutputs(target), identity); reg=L2(10.0))

fnn = fit(nn, target; training_fraction=0.9, optim_options = (; loss = l2))

@time augmented_fpm = augment(fpm, fnn)

# Predict with augmented model
pred_augment = predict(augmented_fpm.model, testpop, coef(augmented_fpm); obstimes=0:0.1:24)

# Overlay predictions for comparison
plotgrid(pred_datamodel; ipred=false, pred=(; color=(:black, 0.4), label="Best possible pred"), observation=:dv)
plotgrid!(pred; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"), observation=:dv)
plotgrid!(pred_augment; ipred=false, pred=(; linestyle=:dash), observation=:dv)

pred_datamodel

# Define a function to compare pred values so that we can see how close our preds were to
# the preds of the datamodel
function pred_residuals(pred1, pred2)
  mapreduce(vcat, pred1, pred2) do p1, p2
    p1.pred.dv .- p2.pred.dv
  end
end


residuals = pred_residuals(pred_datamodel, pred_augment)
mean(abs, residuals)

# residuals between the preds of no covariate model and the preds of the datamodel 
residuals_base = pred_residuals(pred_datamodel, pred)
mean(abs, residuals_base)

############################################################################################
## Scale up: Train covariate model on 1000 subjects
############################################################################################
# We should now have gotten some improvement over not using covariates at all. However,
# training covariate models well requires more data than fitting the neural networks
# embedded in dynamical systems. With UDEs, every observation is a data point. With
# prognostic factor models, every subject is a data point. We've (hopefully) managed to
# improve our model using only 50 subjects, but let's try using data from 1000 patients
# instead.

target_large = preprocess(model, trainpop_large, coef(fpm), FOCE())
nn_large = MLPDomain(numinputs(target), 9, 9, (numoutputs(target), identity); reg=L2(1e-3))
fnn_large = fit(nn_large, target_large; training_fraction=0.9, optim_options = (; loss = l2))
augmented_fpm_large = augment(fpm, fnn_large)


pred_augment_large = predict(augmented_fpm_large.model, testpop, coef(augmented_fpm_large); obstimes=0:0.1:24);

plotgrid(
  pred_datamodel;
  ipred=false,
  pred=(; color=(:black, 0.4), label="Best possible pred"),
  observation = :dv
)
plotgrid!(pred; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"), observation = :dv)
plotgrid!(pred_augment_large; ipred=false, pred=(; linestyle=:dash), observation = :dv)

# residuals between the preds of no covariate model and the preds of the datamodel 
residuals_large = pred_residuals(pred_datamodel, pred_augment_large)
mean(abs, residuals_large)

############################################################################################
## Further refinement: Refit Ω after augmentation
############################################################################################

# After augmenting the model, we could keep on fitting everything in concert. We'd start the
# fit from our sequentially attained parameter values but this would still take time and for
# larger models than this is might be unfeasible.

# However, even if we don't re-fit every parameter, it would be good to fit the Ω_nn such
# that we don't overestimate the unaccounted for between-subject variability now that we've
# taken care of some of that with the covariates.

fpm_refit_Ω = fit(
  augmented_fpm_large.model,
  trainpop_large,
  coef(augmented_fpm_large),
  MAP(FOCE()); 
  constantcoef = Tuple(setdiff(keys(coef(augmented_fpm_large)), (:Ω_nn, :Ω))),
  optim_options = (; time_limit=3*60)
)

# Save and reload fit for reproducibility
serialize(joinpath(ASSETS_DIR, "deep_pumas_fpm-refit.jls"), fpm_refit_Ω)
# fpm_refit_Ω = deserialize(joinpath(ASSETS_DIR, "deep_pumas_fpm-refit.jls"))

# Compare Ω before and after refit
coef(fpm_refit_Ω).Ω_nn ./ coef(augmented_fpm).Ω_nn
coef(fpm_refit_Ω).Ω ./ coef(augmented_fpm).Ω

# Visual predictive check
_vpc = vpc(fpm_refit_Ω; observations = [:dv])

vplt = vpc_plot(_vpc, include_legend = false);
figurelegend(vplt, position=:b, orientation=:horizontal, nbanks=3, tellwidth=true);

vplt


############################################################################################
## Joint deep fit (optional, computationally expensive)
############################################################################################

# Finally, when we don't have the luxury of just increasing the size of our population to
# 1000, there's still one more thing one can do to improve what we can get out of the 50
# patients we trained on. We can jointly fit everything. For large models this may be
# computationally intense, but for this model we should be fine.


fpm_deep = fit(
  augmented_fpm.model,
  trainpop_small,
  coef(augmented_fpm),
  MAP(FOCE());
  optim_options=(; time_limit= 5 * 60), # Note that this will take 5 minutes.
)


# Save and reload fit for reproducibility
serialize(joinpath(ASSETS_DIR, "deep_pumas_fpm-deep.jls"), fpm_deep)
# fpm_deep = deserialize(joinpath(ASSETS_DIR, "deep_pumas_fpm-deep.jls"))


pred_deep = predict(fpm_deep.model, testpop, coef(fpm_deep); obstimes=0:0.1:24);
plotgrid(
  pred_datamodel;
  ipred=false,
  pred=(; color=(:black, 0.4), label="Best possible pred")
)
plotgrid!(pred_augment; ipred=false)
plotgrid!(pred_deep; ipred=false, pred=(; color=Cycled(2), label = "Deep fit pred"))


# Compare the deviation from the best possible pred. 
mean(abs, pred_residuals(pred_datamodel, pred_augment))
mean(abs, pred_residuals(pred_datamodel, pred_deep))


############################################################################################
############################################################################################
