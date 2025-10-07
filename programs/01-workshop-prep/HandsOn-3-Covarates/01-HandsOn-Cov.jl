# =============================================================================
# Workshop: Fitting Dynamical PKPD Models using Pumas (Satellite Course)
# Hands-On 2: DeepNLME modeling in Pumas 
# Goal: Read IPP data → build DeepNLME model → estimate → diagnose → validate
# =============================================================================

############################
# 0) Environment & Packages
############################

using Pumas
using PumasUtilities
using DeepPumas
using CairoMakie

using CSV
using DataFrames
using DataFramesMeta
using Chain                # for @chain macro
using LinearAlgebra        # for Diagonal()
using Unitful              # for u"d" time units
using Random               # for reproducibility
using Serialization        # for saving/loading model fits

# Set a readable plotting theme (try deep_dark() for dark backgrounds)
set_theme!(deep_light())

# Set working directory to script folder 
ARTIFACTS_DIR = joinpath(@__DIR__, "artifacts")


########################################
# 1) Load & validate the source dataset
########################################

# Expect a file "hiv-ipp-data.csv" in ARTIFACTS_DIR
DATA_PATH = joinpath(ARTIFACTS_DIR, "hiv-ipp-cov.csv") #"hiv-ipp-cov.csv")

df_pkpd = CSV.read(DATA_PATH, DataFrame; missingstring = "", stringtype = String)

# Filter for dosing and viral data (evid == 1 and Virus not missing)
df_pd = @chain df_pkpd begin
    @rsubset((:evid == 1) | (!ismissing(:Virus)))
end

# Inspect basic structure (first few rows)
vscodedisplay(df_pd)

#############################################################
# 2) Create Pumas Population object from DataFrame
#############################################################

pop_pd = read_pumas(
    df_pd;
    id         = :id,
    time       = :time,
    observations = [:Virus],
    evid       = :evid,
    amt        = :amt,
    cmt        = :cmt,
    covariates = [:iKa, :iCL, :iVc, :iQ, :iVp, :iDur, :cont_cov]
)

# Split into training and validation populations
_tpop = pop_pd[1:20]   # Training set: first 20 subjects
_vpop = pop_pd[21:end] # Validation set: subjects 41 onward

# Visualize the first 8 training subjects
plotgrid(_tpop[1:8]; data = (; color=:blue))

############################################
# 3) Define the DeepNLME HIV Model 
############################################

nn_hiv = @model begin
  @metadata begin
      desc = "DeepNLME HIV model"
      timeu = u"d" # day
  end
  @param begin
      # Neural networks for PD dynamics
      NN1 ∈ MLPDomain(5, 2, 2, (1, identity); reg = L2(1.0))
      NN2 ∈ MLPDomain(3, 2, 2, (1, identity); reg = L2(1.0))
      # PKPD parameters
      tvR0   ∈ RealDomain(lower=1.0, init=6.5)
      tvλ    ∈ RealDomain(lower=0., init=0.46236)
      tvd    ∈ RealDomain(lower=0., init=0.006)
      tvic50 ∈ RealDomain(lower=0., init=200.0)
      tva    ∈ RealDomain(lower=0., init=0.63911)
      tvu    ∈ RealDomain(lower=0., init=0.011)
      Ω      ∈ PDiagDomain(4)
      ωR0    ∈ RealDomain(lower=0.)
      σ_add  ∈ RealDomain(lower=0.)
  end
  @random begin
      η ~ MvNormal(Ω)
      η_nn1 ~ MvNormal(2, 0.1)
      η_nn2 ~ MvNormal(2, 0.1)
  end
  @covariates iKa iCL iVc iQ iVp iDur cont_cov
  @pre begin
      # Assign covariates to PK parameters
      Ka = iKa
      CL = iCL
      Vc = iVc
      Q  = iQ
      Vp = iVp

      # PD parameters with random effects
      R0 = tvR0 * exp(ωR0 * η_nn1[1])
      λ  = tvλ * exp(η[1])
      d  = tvd
      a  = tva * exp(η[2])
      u  = tvu * exp(η[3])
      IC50 = tvic50 * exp(η[4])

      # Fix neural networks with random effects
      iNN_INF = fix(NN1, η_nn1)
      iNN_VIR = fix(NN2, η_nn2)
  end
  @dosecontrol begin
      duration = (; Depot = iDur)
  end
  @init begin
      # Initial conditions for HIV dynamics
      T = λ / d / R0
      I = λ / a * (1 - (1 / R0))
      V = λ / u * (1 - (1 / R0))
  end
  @vars begin
      # Calculate concentration and inhibition
      Conc = Central / Vc / 1000
      INH = Conc / (Conc + IC50)
      β = R0 * d / λ / V
      infect_term = iNN_INF(T, V, (1 - INH))[1]
      prod_term   = iNN_VIR(I)[1]
  end
  @dynamics begin
      # PK: Two-compartment model with first-order absorption
      Depot'   = -Ka * Depot
      Central' = Ka * Depot + (Q / Vp) * Periph - (Q / Vc) * Central - CL / Vc * Central
      Periph'  = -(Q / Vp) * Periph + (Q / Vc) * Central
      # HIV dynamics
      T' = λ - d * T - infect_term
      I' = infect_term - a * I
      V' = prod_term - u * V
  end
  @derived begin
      Concentration = Conc
      ipred = @. log10(2*abs(V)) + 3
      # ipred = V
      Virus ~ @. Normal(ipred, σ_add)
  end
end
# ---------------------------
# 3. Loglikelihood Check
# ---------------------------

# Check the loglikelihood at initial parameters
# loglikelihood(nn_hiv, _tpop, init_params(nn_hiv), FOCE())

# ---------------------------
# 4. Fit the Model
# ---------------------------

fit_pkpd = fit(
    nn_hiv,
    _tpop,
    init_params(nn_hiv),
    MAP(FOCE()),
    # JointMAP(),
    optim_options = (; iterations=100, f_tol=1e-6),
)

# Save and reload the fit for reproducibility
serialize(joinpath(ARTIFACTS_DIR, "hiv_pkpd_cov.jls"), fit_pkpd)
mdl_fit = deserialize(joinpath(ARTIFACTS_DIR, "hiv_pkpd_cov.jls"))

# ---------------------------
# 5. Diagnostics and Goodness-of-Fit
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
# 6. Predict on Validation Set and Plot
# ---------------------------

# Predict on validation population for days 0 to 42
model_pred = predict(fit_pkpd, _vpop; obstimes=0:1:56)

# Plot predictions for validation subjects 11 to 30
plotgrid(
    model_pred[1:12],
    observation = :Virus,
    pred = (; label = "model pred", linestyle=:dash),
    ipred = (; label = "model ipred"),
    axis = (; limits = ((0., 60.), nothing))
)
