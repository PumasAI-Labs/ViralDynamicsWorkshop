# =============================================================================
# Workshop: Fitting Dynamical models using Pumas (Satellite Course)
# Hands On-1.1 : NLME modeling in Pumas (PK)
# Goal: Read NM-TRAN–style data → build PK models → estimate → diagnose → compare
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
using Random                     # reproducibility
using Serialization              # serialize/deserialize fits
# using PharmaDatasets           # optional helper (not required)

# Choose a readable theme; try deep_dark() if presenting on dark slides
set_theme!(deep_light())

# All outputs (plots/tables/serialized fits) go here
ASSESTS_DIR = joinpath(@__DIR__, "assests/")


########################################
# 1) Load & validate the source dataset
########################################
# Expect a file `hiv-pkpd-data.csv` in the same folder as this script
DATA_PATH = joinpath(@__DIR__, "hiv-pkpd-data.csv")

df_pk = CSV.read(DATA_PATH, DataFrame; missingstring = "", stringtype = String)

# Inspect basic structure
vscodedisplay(df_pk)

unique(df_pk.time)  # observation time grid 

#############################################################
# 2) Parse NM-TRAN style table → Pumas Population{Subject}
#############################################################
pop_pk = read_pumas(
    df_pk;
    id           = :id,
    time         = :time,
    amt          = :amt,
    observations = [:CP],
    cmt          = :cmt,
    evid         = :evid
)

# Indexing examples (a Population is a Vector{Subject} and can be indexed or sliced accordingly)
pop_pk[1]
pop_pk[5:10]
pop_pk[begin:30]
pop_pk[30:end]

# Converting back to a DataFrame
reconstructed_pkdata   = DataFrame(pop_pk)
reconstructed_subject1 = DataFrame(pop_pk[1])

# Plot a manageable subset to avoid over-plotting
plotgrid(pop_pk[1:8]; data=(; color=:blue))

############################################
# 3) Model 1 — 2CMT, first-order absorption
############################################
# Unit convention: timeu = days; multiply *24 for hourly rates.
model_pk2cmt = @model begin
  @metadata begin
    description = "PK: 2 compartment:first-order absorption"
    timeu = u"d"
  end

  @param begin
    tvka      ∈ RealDomain(lower=0.0001, init=2.10)
    tvcl      ∈ RealDomain(lower=0.0001, init=15.3)
    tvvc      ∈ RealDomain(lower=0.001,  init=80.3)
    tvq       ∈ RealDomain(lower=0.0001, init=4.9)
    tvvp      ∈ RealDomain(lower=0.0001, init=5.5)

    Ω         ∈ PDiagDomain(3)
    σ_proppk  ∈ RealDomain(lower=0, init=0.1)
    σ_addpk   ∈ RealDomain(lower=0, init=10)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    Ka = tvka * exp(η[1])
    CL = tvcl * exp(η[2])
    Vc = tvvc * exp(η[3])
    Q  = tvq  
    Vp = tvvp
  end


  @init begin
    AUC = 0
  end

  @vars begin
    Conc = (Central / (Vc/1000))
  end

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot + (Q/Vp)*Periph - (Q/Vc)*Central - CL/Vc*Central
    Periph'  = -(Q/Vp)*Periph + (Q/Vc)*Central
    AUC'     =  Conc
  end

  @derived begin
    CP ~ @. Normal(Conc, sqrt(σ_addpk^2 + (abs(Conc) * σ_proppk)^2))
  end
end

# Reasonable initial parameters
param_pk2cmt = (
  tvka = 1.408,
  tvcl = 15.63,
  tvvc = 74.3,
  tvq = 15.0,
  tvvp = 22.0,
  Ω         = Diagonal([0.1, 0.1, 0.1]),
  σ_proppk  = 0.2,
  σ_addpk   = 10.0
)

# Initial parameters can also be extracted from the model
init_params(model_pk2cmt)

#########################################
# 4) Estimation for Model 1 (2CMT first-order absorption)
#########################################
fit_pk2cmt_foce        = fit(model_pk2cmt, pop_pk, param_pk2cmt, FOCE())
fit_pk2cmt_np          = fit(model_pk2cmt, pop_pk, param_pk2cmt, NaivePooled(); omegas = (:Ω,))
fit_pk2cmt_laplace     = fit(model_pk2cmt, pop_pk, param_pk2cmt, LaplaceI())
fit_pk2cmt_foce_fixed  = fit(model_pk2cmt, pop_pk, param_pk2cmt, FOCE(); constantcoef = (:tvcl,))

# Coefficients/tables 
coef(fit_pk2cmt_foce)
coeftable(fit_pk2cmt_foce)     # DataFrame of estimates, SE, etc.

# Individual parameter estimates as a DataFrame
DataFrame(icoef(fit_pk2cmt_foce))

#########################################################
# 5) Predictions (obs-time grid and custom grids)
#########################################################
pred_pk2cmt_obs   = predict(fit_pk2cmt_foce)           # at observed times
DataFrame(pred_pk2cmt_obs)

pred_pk2cmt_dense = predict(fit_pk2cmt_foce; obstimes = 0:0.1:18)

plotgrid(pred_pk2cmt_obs[1:6])
plotgrid(pred_pk2cmt_dense[1:6])

#############################################
# 6) Alternative data with :rate 
#############################################
# Many NONMEM workflows convey zero-order input via :rate. Here we mark dose rows
# with -2 (a common duration-coded convention); observation rows remain missing.

df_pk_seq = @chain df_pk begin
  @rtransform(:rate = :evid == 1 ? -2 : missing)
end

vscodedisplay(df_pk_seq)

pop_pk_seq = read_pumas(
  df_pk_seq;
  id           = :id,
  time         = :time,
  amt          = :amt,
  observations = [:CP],
  cmt          = :cmt,
  evid         = :evid,
  rate         = :rate
)

#############################################################
# 7) Model 2 — sequential zero + first-order absorption
#############################################################
model_pkseq = @model begin
  @metadata begin
    description = "PK: sequential zero + first-order absorption"
    timeu = u"d"
  end

  @param begin
    tvka      ∈ RealDomain(lower=0.0001, init=2.10)
    tvcl      ∈ RealDomain(lower=0.0001, init=15.3)
    tvvc      ∈ RealDomain(lower=0.001,  init=80.3)
    tvq       ∈ RealDomain(lower=0.0001, init=4.9)
    tvvp      ∈ RealDomain(lower=0.0001, init=5.5)
    tvd       ∈ RealDomain(lower=0.0001, init=5.83)  # zero-order duration
    Ω         ∈ PDiagDomain(4)
    σ_proppk  ∈ RealDomain(lower=0, init=0.1)
    σ_addpk   ∈ RealDomain(lower=0, init=10)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    Ka = tvka * exp(η[1])
    CL = tvcl * exp(η[2])
    Vc = tvvc * exp(η[3])
    Q  = tvq  
    Vp = tvvp
  end

  @dosecontrol begin
    # Duration for zero-order input into Depot (convert hours → days to match timeu)
    duration = (; Depot = tvd * exp(η[4]))
  end

  @init begin
    AUC = 0
  end

  @vars begin
    Conc = (Central / (Vc/1000))
  end

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot + (Q/Vp)*Periph - (Q/Vc)*Central - CL/Vc*Central
    Periph'  = -(Q/Vp)*Periph + (Q/Vc)*Central
    AUC'     =  Conc
  end

  @derived begin
    CP ~ @. Normal(
      Conc,
      sqrt(σ_addpk^2 + (abs(Conc) * σ_proppk)^2)
    )
  end
end

fit_pkseq_foce    = fit(model_pkseq, pop_pk_seq, init_params(model_pkseq), FOCE())

##########################################
# 8) Model comparison & diagnostics
##########################################
# Metrics tables (AIC/BIC/LL/shrinkage summaries)
tbl_metrics_pk2cmt = metrics_table(fit_pk2cmt_foce)
tbl_metrics_pkseq  = metrics_table(fit_pkseq_foce)

# Individual calls are also available
loglikelihood(fit_pkseq_foce)
aic(fit_pkseq_foce)
bic(fit_pkseq_foce)
ηshrinkage(fit_pkseq_foce)
ϵshrinkage(fit_pkseq_foce)

# GOF panels (4 plots): obs vs pred, obs vs ipred, wres vs time, wres vs ipred
ins_pk2cmt = inspect(fit_pk2cmt_foce)
ins_pkseq  = inspect(fit_pkseq_foce)

goffig_pk2cmt = goodness_of_fit(ins_pk2cmt)
goffig_pkseq  = goodness_of_fit(ins_pkseq; figure = (; fontsize = 15))

# Overlay individual prediction plots for quick visual comparison on early window
pred_pkseq_dense = predict(fit_pkseq_foce; obstimes=0:0.1:18)


plt_overlay = plotgrid(pred_pk2cmt_dense[1:6]);
plotgrid!(
  plt_overlay,
  pred_pkseq_dense[1:6];
  pred  = (; color=:green, label="model 2", linestyle=:dash),
  ipred = (; color=:red,   label="model 2", linestyle=:dash),
)

#########################################
# 9) Visual Predictive Checks (VPC)
#########################################
# Model 1
vpc_pk2cmt   = vpc(fit_pk2cmt_foce)

vpcfig_pk2cmt = vpc_plot(
  vpc_pk2cmt;
  simquantile_medians = true, observations = false, include_legend = false,
  axis = (xlabel = "Time (h)", ylabel = "Concentration (ng/mL)", xticks = 0:50:200)
);

figurelegend(vpcfig_pk2cmt, position=:b, orientation=:horizontal, nbanks=3, tellwidth=true);

vpcfig_pk2cmt

# Model 2
vpc_pkseq    = vpc(fit_pkseq_foce)

vpcfig_pkseq = vpc_plot(
  vpc_pkseq;
  simquantile_medians = true, observations = false, include_legend = false,
  axis = (xlabel = "Time (h)", ylabel = "Concentration (ng/mL)", xticks = 0:50:200)
);

figurelegend(vpcfig_pkseq, position=:b, orientation=:horizontal, nbanks=3, tellwidth=true);

vpcfig_pkseq


###### Covarience step ######
infer(fit_pkseq_foce)

###########################################################
# 10) Persist ASSESTS: tables, plots, and serialized fits
###########################################################
# Save key tables
# CSV.write(joinpath(ASSETS_DIR, "coef_pk2cmt_foce.csv"),  coeftable(fit_pk2cmt_foce))
# CSV.write(joinpath(ASSETS_DIR, "coef_pkseq_foce.csv"),   coeftable(fit_pkseq_foce))
# CSV.write(joinpath(ASSETS_DIR, "metrics_pk2cmt_foce.csv"), tbl_metrics_pk2cmt)
# CSV.write(joinpath(ASSETS_DIR, "metrics_pkseq_foce.csv"),  tbl_metrics_pkseq)

# # Save figures
# save(joinpath(ASSETS_DIR, "gof_pk2cmt.png"), goffig_pk2cmt)
# save(joinpath(ASSETS_DIR, "gof_pkseq.png"),  goffig_pkseq)
# save(joinpath(ASSETS_DIR, "overlay_pred.png"), plt_overlay)
# save(joinpath(ASSETS_DIR, "vpc_pk2cmt.png"), vpcfig_pk2cmt)
# save(joinpath(ASSETS_DIR, "vpc_pkseq.png"),  vpcfig_pkseq)

# Serialize fits (built-in Serialization; robust, no extra deps)

# open(joinpath(ASSETS_DIR, "fit_pkseq_foce.jls"), "w") do io
#     serialize(io, fit_pkseq_foce)
# end

serialize(joinpath(ASSETS_DIR, "fit_pkseq_foce.jls"), fit_pkseq_foce)

# Example: how to deserialize later
# deserialize(joinpath(ASSETS_DIR, "fit_pkseq_foce.jls"))

# =============================================================================
# End of the script
# =============================================================================


