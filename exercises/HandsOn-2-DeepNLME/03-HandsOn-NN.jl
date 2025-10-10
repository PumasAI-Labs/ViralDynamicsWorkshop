# =============================================================================
# Workshop: Fitting Dynamical PKPD Models using Pumas (Satellite Course)
# Hands-On 2: DeepNLME modeling in Pumas
# Goal: Read IPP data → build DeepNLME model → estimate → diagnose → validate
# =============================================================================


########################################
# 0) Environment & Packages
########################################

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

# Set plotting theme (deep_dark for better contrast)
set_theme!(deep_dark(); figure = (; size = (1000, 1000)))

# Set backend for neural networks
DeepPumas.set_mlp_backend(:staticflux)

# Set working directory and artifacts folder
cd(@__DIR__)
ASSETS_DIR = joinpath(@__DIR__, "assets")


########################################
# 1) Load & Validate the Source Dataset
########################################

# Read HIV IPP dataset
df_pkpd = CSV.read("hiv-ipp-data.csv", DataFrame; missingstring = "", stringtype = String);

# Filter for dosing and viral data (evid == 1 and Virus not missing)
df_pd = @chain df_pkpd begin
    @rsubset((:evid == 1) | (!ismissing(:Virus)))
end;

# Inspect basic structure (first few rows)
vscodedisplay(df_pd)


#############################################################
# 2) Create Pumas Population Object from DataFrame
#############################################################

pop_pd = read_pumas(
    df_pd;
    id           = :id,
    time         = :time,
    observations = [:Virus],
    evid         = :evid,
    amt          = :amt,
    cmt          = :cmt,
    rate         = :rate,
    covariates   = [:iKa, :iCL, :iVc, :iQ, :iVp, :iDur]
)

# Split into training and validation populations
trainpop = pop_pd[1:20]    # Training set: first 20 subjects
validpop = pop_pd[21:end]  # Validation set: remaining subjects

# Visualize the first 8 training subjects
plotgrid(trainpop[1:8]; data = (; color = :blue))


############################################
# 3) Define the DeepNLME HIV Model
############################################

model_hiv = @model begin
    @metadata begin
      desc = "Simple HIV PK–PD (Luo-style log10 parametrization) + explicit R0 + Kin/Kout initials"
      timeu = u"d"
    end
  
    @param begin
      NN1 ∈ MLPDomain(2 + 1, 7, 7, (1, softplus); reg=L2(1.0, input=false, output=false))
      tvλ ∈ RealDomain(lower=0)
      d ∈ RealDomain(lower=0)
      a ∈ RealDomain(lower=0)
    #   γ ∈ RealDomain(lower=0, init=1e5)
      x0 ∈ RealDomain(lower=0)
    #   ω ∈ RealDomain(lower=0)

    # Drug effect parameters
    tvec50 ∈ RealDomain(lower = 0.001, init = 150.0)

      Ω ∈ PSDDomain(2)
      EC50_Ω  ∈ PDiagDomain(1)
      σ ∈ RealDomain(lower=0.)
    end
  
    @random begin
      η ~ MvNormal(Ω)
      EC50_η ~ MvNormal(EC50_Ω) 
      η_nn ~ Normal()
    end
  
    @covariates iKa iCL iVc iQ iVp iDur
  
    @pre begin
      λ = tvλ * exp(η[1])
  
      totcells = λ / d
      T0 = totcells * logistic(x0 + η[2])
      I0 = totcells * (1 - logistic(x0 + η[2]))
    #   V0 = I0 * γ / ω # assume quasi-equilibirum

    EC50 = tvec50 * exp(EC50_η[1])
  
    iNN_INF = fix(NN1, η_nn)
    end
  
    @dosecontrol begin
      duration = (; Depot=iDur)
    end
  
    @init begin
      Depot = 0.0
      Central = 0.0
      T = T0
      I = I0
    #   V = V0
    end
  
    @vars begin
    CP   = 1000 * Central / iVc
    EFF  = CP / (EC50 + CP)
    V = I * 1e5
    infect_term = iNN_INF(V / 1e5, EFF)[1]
    end
  
    @dynamics begin
      Depot' = -iKa * Depot
      Central' = iKa * Depot - (iQ / iVc) * Central + (iQ / iVp) * Periph - (iCL / iVc) * Central
      Periph' = (iQ / iVc) * Central - (iQ / iVp) * Periph
  
      T' = λ - d * T - infect_term * T
      I' = infect_term * T - a * I
    #   V' = γ * I - ω * V
    end
  
    @derived begin
      log10V = @.log10(abs(V))
      Virus ~ @. Normal(log10V, σ)
    end
  end

########################################
# 4) Fit the Model
########################################

fit_hiv = fit(
    model_hiv,
    trainpop,
    init_params(model_hiv),
    MAP(FOCE());
    optim_options = (; iterations = 50, f_tol = 1e-6) #<---- only 50 iterations, could be increased to 100 ~ 12mins
)

# Save and reload the fit for reproducibility
# serialize(joinpath(ASSETS_DIR, "hiv_DeepNLME2_nn.jls"), fit_hiv)
mdl_fit = deserialize(joinpath(ASSETS_DIR, "hiv_DeepNLME2_nn.jls"))

########################################
# 5) Predict on Validation Set and Plot
########################################

# Predict on validation population for days 0 to 30
model_pred = predict(mdl_fit, validpop; obstimes = 0:0.5:30)

# Plot predictions for validation subjects 11 to 22
plotgrid(
    model_pred[11:22];
    observation = :Virus,
    pred        = (; label = "model pred", linestyle = :dash),
    ipred       = (; label = "model ipred"),
    axis        = (; limits = ((0., 30.), nothing))
)

########################################
# 6) Visual Predictive Check (VPC)
########################################

vpc_hiv = vpc(fit_hiv);

vpcfig_hiv = vpc_plot(
    vpc_hiv;
    simquantile_medians = true,
    observations         = false,
    include_legend       = false,
    axis = (
        xlabel = "Time (h)",
        ylabel = "Log10 HIV RNA",
        xticks = 0:50:200
    )
);

figurelegend(vpcfig_hiv, position = :b, orientation = :horizontal, nbanks = 3, tellwidth = true);
vpcfig_hiv

########################################################################################################################
# End of the Script
########################################################################################################################
