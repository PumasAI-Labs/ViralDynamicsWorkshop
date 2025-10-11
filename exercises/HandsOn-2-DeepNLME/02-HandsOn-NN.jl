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
        desc = "DeepNLME for HIV Dynamics"
        timeu = u"d"  # Model time unit = days
    end

    @param begin
        # Neural network: maps inputs to viral dynamics
        NN ∈ MLPDomain(4, 8, 8, (1, identity); reg = L2(1.0))

        # PD parameters
        tvV0 ∈ RealDomain(lower = 0.001, init = 0.5)

        # Random effects
        ωV0  ∈ RealDomain(; lower = 0)

        # Residual error
        σ ∈ RealDomain(lower = 0.001, init = 0.1)
    end

    @random begin
        η ~ MvNormal(I(2))
    end

    @covariates iKa iCL iVc iQ iVp iDur

    @pre begin
        # PK parameters from covariates
        KA = iKa
        CL = iCL
        VC = iVc
        Q  = iQ
        VP = iVp

        # Baseline viral RNA
        V0 = tvV0 * exp(ωV0 * η[1])

        # Fix random effects as non-dynamic inputs to the NN and return an "individual" NN
        iNN = fix(NN, η)
    end

    @dosecontrol begin
        duration = (; Depot = iDur)
        # Sequential zero and first-order absorption codes
    end

    @init begin
        Depot   = 0.0
        Central = 0.0
        V       = V0
    end

    @dynamics begin
        Depot'   = -KA * Depot
        Central' = KA * Depot - (CL / VC) * Central - (Q / VC) * Central + (Q / VP) * Periph
        Periph'  = (Q / VC) * Central - (Q / VP) * Periph

        V' = iNN(V, Central/VC)[1] * V
    end

    @derived begin
        Virus ~ @. Normal(log10(V) + 4, σ)
    end
end

# Note: instead of Central/VC, we could have Emax/Imax function
########################################
# 4) Fit the Model
########################################

fit_hiv = fit(
    model_hiv,
    trainpop,
    init_params(model_hiv),
    MAP(FOCE());
    optim_options = (; iterations = 200, f_tol = 1e-6)
)

# Save and reload the fit for reproducibility
serialize(joinpath(ASSETS_DIR, "hiv_DeepNLME_nn.jls"), fit_hiv)
# fit_hiv = deserialize(joinpath(ARTIFACTS_DIR, "hiv_pkpd_nn.jls"))


########################################
# 5) Predict on Validation Set and Plot
########################################

# Predict on validation population for days 0 to 30
model_pred = predict(fit_hiv, validpop; obstimes = 0:0.5:30)

# Plot predictions for validation subjects 11 to 22
plotgrid(
    model_pred[1:12];
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
# External Validation
########################################################################################################################

fit_hiv = deserialize(joinpath(ASSETS_DIR, "hiv_DeepNLME_nn.jls"))

########################################
# 7) Validation dataset
########################################
# Instead of single 500mg dose
## 300 mg dose given every 2 weeks
## 4 doses
## Obervation window instead of 28days, here 84 days


# Read HIV IPP dataset
df_mDos = CSV.read("hiv-ipp-multiple-dose.csv", DataFrame; missingstring = "", stringtype = String);

#############################################################
# 2) Create Pumas Population Object from DataFrame
#############################################################

pop_mDos= read_pumas(
    df_mDos;
    id           = :id,
    time         = :time,
    observations = [:Virus],
    evid         = :evid,
    amt          = :amt,
    cmt          = :cmt,
    rate         = :rate,
    covariates   = [:iKa, :iCL, :iVc, :iQ, :iVp, :iDur]
)

# For tests lets use a first 150 subjects from the validation populations
testpop = pop_mDos[1:150]    # Test set: first 150 subjects

# Visualize the first 8 training subjects
plotgrid(testpop[1:8]; data = (; color = :blue))


########################################
# 5) Prediction using developed NN model
########################################

# Predict on validation population for days 0 to 30
model_pred = predict(fit_hiv, testpop; obstimes = 0:0.1:90)

# Plot predictions for validation subjects 11 to 22
plotgrid(
    model_pred[1:24];
    observation = :Virus,
    pred        = (; label = "model pred", linestyle = :dash),
    ipred       = (; label = "model ipred"),
    axis        = (; limits = ((0., 90.), nothing))
)

########################################################################################################################
# End of the Script
########################################################################################################################