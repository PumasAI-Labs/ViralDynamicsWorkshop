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
using DataFramesMeta

# Backend for neural networks and plotting theme
set_mlp_backend(:staticflux)
set_theme!(deep_light())

# Set working directory and artifacts folder
cd(@__DIR__)
ASSETS_DIR = joinpath(@__DIR__, "assets")


############################################################################################
## Load covariate data
############################################################################################

df_cov = CSV.read("hiv-ipp-cov.csv", DataFrame; missingstring = "", stringtype = String)

pop = read_pumas(
  df_cov;
    id = :id,
    time = :time,
    observations = [:Virus], #
    covariates=[:R_eq,:c1,:c2,:c3,:c4,:c5,:c6, :iKa, :iCL, :iVc, :iQ, :iVp, :iDur],
    evid = :evid,
    amt = :amt,
    cmt = :cmt,
    rate = :rate    
)

# Visualize covariate distributions
covariates_dist(pop, covariates = [:R_eq,:c1,:c2,:c3,:c4,:c5,:c6])

# Split the data into different training/test populations
trainpop_small = pop[1:50]
trainpop_large = pop[1:1000]
testpop = pop[1001:end]


# Load previously fitted NLME model (used for data generation)
fit_pd = deserialize(joinpath(ASSETS_DIR, "fit_pd_foce.jls"))

# Define model from the fit_pd
datamodel = fit_pd.model

# Visualize predictions from data-generating model
pred_datamodel = predict(datamodel, testpop, init_params(datamodel); obstimes=0:0.1:42)
plotgrid(pred_datamodel; observation=:Virus)

############################################################################################
## Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD is entirely determined by a neural network.
# At this point, we're not trying to explain how patient data may inform individual
# parameters.

model = @model begin
  @metadata begin
      desc = "DeepNLME for HIV Dynamics"
      timeu = u"d"  # Model time unit = days
  end

  @param begin
      # Neural network: maps inputs to viral dynamics
      NN ∈ MLPDomain(4, 8, 8, (1, identity); reg = L2(1.0))

      # PD parameters
      tvr0 ∈ RealDomain(lower = 0.001, init = 0.5)

      # Drug effect parameters
      tvec50 ∈ RealDomain(lower = 0.001, init = 150.0)

      # Random effects
      Ω ∈ PDiagDomain(2)

      # Residual error
      σ ∈ RealDomain(lower = 0.001, init = 0.1)
  end

  @random begin
      η ~ MvNormal(Ω)
      η_nn ~ MvNormal(2, 0.1)
  end

  @covariates iKa iCL iVc iQ iVp iDur R_eq c1 c2 c3 c4 c5 c6

  @pre begin
      # PK parameters from covariates
      KA = iKa
      CL = iCL
      VC = iVc
      Q  = iQ
      VP = iVp

      # Baseline viral RNA
      R0 = tvr0 * exp(η[1] * η_nn[1])

      # Individual EC50 value
      EC50 = tvec50 * exp(η[2])

      # Fix random effects as non-dynamic inputs to the NN and return an "individual" NN
      iNN = fix(NN, η_nn)
  end

  @dosecontrol begin
      duration = (; Depot = iDur)
      # Sequential zero and first-order absorption codes
  end

  @init begin
      Depot   = 0.0
      Central = 0.0
      R       = R0
  end

  @vars begin
      CP   = 1000 * Central / VC
      Conc = Central / VC / 10
      EFF  = CP / (EC50 + CP)
      INH  = 1.0 - EFF

      # Neural infection term
      viralDynamics = iNN(R / 10, EFF)[1]

      # For the concentration effect input to the NN, we could use Conc, EFF, or INH.
      # These will give a similar description of the data.

      V = 10 * R
  end

  @dynamics begin
      Depot'   = -KA * Depot
      Central' = KA * Depot - (CL / VC) * Central - (Q / VC) * Central + (Q / VP) * Periph
      Periph'  = (Q / VC) * Central - (Q / VP) * Periph

      R' = viralDynamics
  end

  @derived begin
      Virus ~ @. Normal(V, σ)
  end
end

# Fit the neural-embedded model
fpm = fit(
  model,
  trainpop_small,
  init_params(model),
  MAP(FOCE());
  optim_options=(; iterations=100)
)

# Save and reload fit for reproducibility
serialize(joinpath(ASSETS_DIR, "deep_pumas_hiv_fpm.jls"), fpm)
# fpm = deserialize(joinpath(ASSETS_DIR, "deep_pumas_hiv_fpm.jls"))

# Predict on test population
pred = predict(fpm, testpop; obstimes=0:0.1:42)
plotgrid(pred; observation=:Virus)

############################################################################################
## 'Augment' the model to predict heterogeneity from data
############################################################################################
# All patient heterogeneity of our recent model was captured by random effects and can thus
# not be predicted by the model. Here, we 'augment' that model with ML that's trained to 
# capture this heterogeneity from data.

# Generate a target for the ML fitting from a Normal approximation of the posterior η
# distribution.
target = preprocess(fpm, covs= (:R_eq,:c1,:c2,:c3,:c4,:c5,:c6))

nn = MLPDomain(numinputs(target), 9, 9, (numoutputs(target), identity); reg=L2(10.0))

fnn = fit(nn, target; training_fraction=0.9, optim_options = (; loss = l2))

@time augmented_fpm = augment(fpm, fnn)

# Predict with augmented model
pred_augment = predict(augmented_fpm.model, testpop, coef(augmented_fpm); obstimes=0:0.1:42)

# Overlay predictions for comparison
plotgrid(pred_datamodel; ipred=false, pred=(; color=(:black, 0.4), label="Best possible pred"), observation=:Virus)
plotgrid!(pred; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"), observation=:Virus)
plotgrid!(pred_augment; ipred=false, pred=(; linestyle=:dash), observation=:Virus)

# Define a function to compare pred values so that we can see how close our preds were to
# the preds of the datamodel
function pred_residuals(pred1, pred2)
  mapreduce(vcat, pred1, pred2) do p1, p2
    p1.pred.Virus .- p2.pred.Virus
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

target_large = preprocess(model, trainpop_large, coef(fpm), FOCE(), covs= (:R_eq,:c1,:c2,:c3,:c4,:c5,:c6))
nn_large = MLPDomain(numinputs(target), 9, 9, (numoutputs(target), identity); reg=L2(1e-3))
fnn_large = fit(nn_large, target_large; training_fraction=0.9, optim_options = (; loss = l2))
augmented_fpm_large = augment(fpm, fnn_large)


pred_augment_large = predict(augmented_fpm_large.model, testpop, coef(augmented_fpm_large); obstimes=0:0.1:42);

plotgrid(
  pred_datamodel;
  ipred=false,
  pred=(; color=(:black, 0.4), label="Best possible pred"),
  observation = :Virus
)
plotgrid!(pred; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"), observation = :Virus)
plotgrid!(pred_augment_large; ipred=false, pred=(; linestyle=:dash), observation = :Virus)

# residuals between the preds of no covariate model and the preds of the datamodel 
residuals_large = pred_residuals(pred_datamodel, pred_augment_large)
mean(abs, residuals_large)

############################################################################################
# End of the Script
############################################################################################
