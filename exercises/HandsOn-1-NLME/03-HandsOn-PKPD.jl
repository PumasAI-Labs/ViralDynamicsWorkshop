# =============================================================================
# Workshop: Fitting Dynamical PKPD Models using Pumas (Satellite Course)
# Hands-On 1.3: NLME modeling in Pumas (PKPD)
# Goal: Read IPP data → build PKPD models → estimate → diagnose → validate
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
ASSESTS_DIR = joinpath(@__DIR__, "assests")


########################################
# 1) Load & validate the source dataset
########################################

# Expect a file "hiv-ipp-data.csv" in ASSESTS_DIR
DATA_PATH = joinpath(ASSESTS_DIR, "hiv-ipp-data.csv")

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
    rate       = :rate,
    covariates = [:iKa, :iCL, :iVc, :iQ, :iVp, :iDur]
)

# Visualize the first 8 training subjects
plotgrid(pop_pd[1:8]; data=(; color=:blue))

############################################
# 3) Define the HIV PKPD Model (PD Only)
############################################

model_pd = @model begin
    @metadata begin
      desc = "HIV PK–PD Model"
      timeu = u"d"
    end
  
    @param begin
 
      # PD parameters (log10 biological rates)
      log10_lambda ∈ RealDomain(init= 2.47)
      log10_d      ∈ RealDomain(init= -0.74)
      log10_beta   ∈ RealDomain(init= -5.41)
      log10_a      ∈ RealDomain(init= -0.17)
      log10_gamma  ∈ RealDomain(init= 3.77)
      log10_omega  ∈ RealDomain(init= 1.27)
  
      # Drug effect parameters
      tvec50 ∈ RealDomain(lower=0, init= 136.0)
  
      # Random effects
      Ω ∈ PDiagDomain(2)
  
      # Residual error
      σ ∈ RealDomain(lower=0.001, init= 0.318)
    end
  
    @random begin
      η ~ MvNormal(Ω)
    end
  
    @covariates iKa iCL iVc iQ iVp iDur

    @pre begin
        # PK parameters from covariates
        KA = iKa
        CL = iCL
        VC = iVc
        Q  = iQ
        VP = iVp 
  
      # PD (convert log10 to linear)
      LAMBDA = exp(log10_lambda * log(10)) * exp(η[1])
      D      = exp(log10_d      * log(10))
      BETA   = exp(log10_beta   * log(10))
      A      = exp(log10_a      * log(10))
      GAMMA  = exp(log10_gamma  * log(10))
      OMEGA  = exp(log10_omega  * log(10))
  
      # Drug effect parameters
      EC50 = tvec50 * exp(η[2])
  
      # Basic reproduction number
      R0 = BETA * LAMBDA * GAMMA / (D * A * OMEGA)
    end
  
    @dosecontrol begin
        duration = (; Depot = iDur)
      #sequential zero and first order abs codes
      end
  
    @init begin
      Depot   = 0.0
      Central = 0.0
      T       = LAMBDA / (D * R0)                          # x0
      I       = LAMBDA / A * (1.0 - 1.0 / R0)              # y0
      V       = GAMMA * LAMBDA / (A * OMEGA) * (1.0 - 1.0 / R0)  # v0
    end
  
    @vars begin
      CP  = 1000 * Central / VC
      EFF = CP / (EC50 + CP)  # fractional reduction of infectivity
    end
  
    @dynamics begin
      Depot'   = -KA * Depot
      Central' =  KA * Depot + (Q/VP)*Periph - (CL / VC) * Central - (Q/VC)*Central
      Periph'  =  (Q/VC)*Central - (Q/VP)*Periph
  
      T' = LAMBDA - D * T - (1.0 - EFF) * BETA * T * V
      I' = (1.0 - EFF) * BETA * T * V - A * I
      V' = GAMMA * I - OMEGA * V
    end
  
    @derived begin
      log10V = @.log10(V + 1e-6)
      R0_out = R0
      Virus ~ @. Normal(log10V, σ)
    end
  end

############################################
# 4) Fit the Model and Check Loglikelihood
############################################

# Check the loglikelihood at initial parameters
loglikelihood(model_pd, pop_pd, init_params(model_pd), FOCE())

# Fit the model using FOCE
fit_pd_foce = fit(
    model_pd,
    pop_pd,
    init_params(model_pd),
    FOCE()
)

# Save and reload the fit for reproducibility
# serialize(joinpath(ASSESTS_DIR, "fit_pd_foce.jls"), fit_pd_foce)
# fit_pd_foce_loaded = deserialize(joinpath(ASSESTS_DIR, "fit_pd_foce.jls"))

# Infer parameter uncertainty
infer(fit_pd_foce)

############################################
# 5) Diagnostics and Goodness-of-Fit
############################################

# Inspect the fitted model
insp_pd = inspect(fit_pd_foce)

# Overall goodness-of-fit summary (4-panel plot)
goodness_of_fit(insp_pd)

# Key diagnostic plots
observations_vs_ipredictions(insp_pd)

# Additional diagnostics (uncomment as needed)
# observations_vs_predictions(insp_pd)
# wresiduals_vs_time(insp_pd)
# wresiduals_vs_predictions(insp_pd)
# wresiduals_dist(insp_pd)
# empirical_bayes_dist(insp_pd)

############################################
# 6) Predict on Validation Set and Plot
############################################

# Predict on validation population for days 0 to 30
pred_pd_valid = predict(fit_pd_foce, pop_pd; obstimes=0:0.5:30)

# Plot predictions for validation subjects 1 to 12
plotgrid(
    pred_pd_valid[21:28],
    observation = :Virus,
    pred        = (; label = "model pred", linestyle=:dash),
    ipred       = (; label = "model ipred"),
    axis        = (; limits = ((0., 30.), nothing))
)


# VPC
vpc_pd    = vpc(fit_pd_foce)

vpcfig_pd = vpc_plot(
  vpc_pd;
  simquantile_medians = true, observations = false, include_legend = false,
  axis = (xlabel = "Time (h)", ylabel = "Log10 Viral RNA", xticks = 0:50:200)
);

figurelegend(vpcfig_pd, position=:b, orientation=:horizontal, nbanks=3, tellwidth=true);

vpcfig_pd
# =============================================================================
# End of the script
# =============================================================================