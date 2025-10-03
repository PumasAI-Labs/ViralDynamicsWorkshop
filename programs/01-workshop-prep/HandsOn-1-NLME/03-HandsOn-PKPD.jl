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
ARTIFACTS_DIR = joinpath(@__DIR__, "artifacts")


########################################
# 1) Load & validate the source dataset
########################################

# Expect a file "hiv-ipp-data.csv" in ARTIFACTS_DIR
DATA_PATH = joinpath(ARTIFACTS_DIR, "hiv-ipp-data.csv")

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
    covariates = [:iKa, :iCL, :iVc, :iQ, :iVp, :iDur]
)

# Visualize the first 8 training subjects
plotgrid(pop_pd[1:8]; data=(; color=:blue))

############################################
# 3) Define the HIV PKPD Model (PD Only)
############################################

model_pd = @model begin
    @metadata begin
        desc = "HIV PD Only"
        timeu = u"d"   # time in days
    end

    @param begin
        """ Viral production rate multiplier """
        tvpro    ∈ RealDomain(lower=1, init=6.1)
        """ Death rate of actively infected cells (1/day) """
        tvdelta  ∈ RealDomain(lower=0, init=0.63)
        """ Source rate of uninfected cells (cells/day) """
        tvlambda ∈ RealDomain(lower=0, init=0.46)
        """ IC50 for drug effect (ng/mL) """
        tvic50   ∈ RealDomain(lower=0, init=300.0)
        """ Inter-individual variability """
        Ω        ∈ PDiagDomain(4)
        """ Additive residual error """
        σ_add    ∈ RealDomain(lower=0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates iKa iCL iVc iQ iVp iDur

    @pre begin
        # PK parameters from covariates
        Ka = iKa
        CL = iCL
        Vc = iVc
        Q  = iQ
        Vp = iVp

        # PD parameters with random effects
        RR0    = tvpro   * exp(η[1])   # Basic reproductive ratio
        DELTA  = tvdelta * exp(η[2])   # Death rate of active infected cells
        LAMBDA = tvlambda* exp(η[3])   # Source rate of uninfected cells
        IC50   = tvic50  * exp(η[4])   # Drug potency

        # Fixed biological constants
        DU    = 0.006    # Death rate of uninfected cells (1/day)
        DL    = 0.04     # Death rate of latently infected cells (1/day)
        AL    = 0.036    # Conversion rate from latent to active (1/day)
        POVC  = 35.4     # Ratio of production to clearance (Funk 2001)
        DLL   = 0.01     # Death rate of long-lived infected cells (1/day)
        PLLC  = 0.374    # Ratio of birth/death for long-lived infected cells
        QLL   = 0.001    # Fraction of long-lived infected cells
        QA    = 0.97     # Fraction actively infected
        QL    = 0.029    # Fraction latently infected
    end

    @dosecontrol begin
        duration = (; Depot = iDur)
    end

    @init begin
        # Initial conditions for cell populations
        UNINFECTED = LAMBDA / (DU * RR0)
        ACTIVEIC   = (QA + QL*AL/(DL+AL)) * LAMBDA/DELTA * (1 - 1/RR0)
        LATENT     = QL * LAMBDA/(DL+AL) * (1 - 1/RR0)
        LLIC       = QLL * LAMBDA/DLL * (1 - 1/RR0)
        # PK/PD tracking
        AUC        = 0
    end

    @vars begin
        # Fraction accounting for latent infection
        LFAC = QA + QL*AL/(AL+DL)
        # Infection rate constant (β)
        BETA = RR0*DU / LAMBDA / (POVC*LFAC/DELTA + PLLC*QLL/DLL)
        # Drug concentration (ng/mL)
        Conc = Central / (Vc/1000)
        # Drug inhibition function
        INH  = Conc / (Conc + IC50)
        # Viral load proxy (from active + long-lived infected cells)
        V    = abs(POVC*ACTIVEIC + PLLC*LLIC)
    end

    @dynamics begin
        # PK compartments
        Depot'    = -Ka*Depot
        Central'  = Ka*Depot + (Q/Vp)*Periph - (Q/Vc)*Central - (CL/Vc)*Central
        Periph'   = -(Q/Vp)*Periph + (Q/Vc)*Central

        # PD compartments
        UNINFECTED' = LAMBDA - BETA*V*(1-INH)*UNINFECTED - DU*UNINFECTED
        ACTIVEIC'   = QA*BETA*V*(1-INH)*UNINFECTED - DELTA*ACTIVEIC + AL*LATENT
        LATENT'     = QL*BETA*V*(1-INH)*UNINFECTED - (DL+AL)*LATENT
        LLIC'       = QLL*BETA*V*(1-INH)*UNINFECTED - DLL*LLIC

        # PK exposure metric
        AUC'       = Central / (Vc/1000)
    end

    @derived begin
        """ Plasma concentration (ng/mL) """
        Concentration = Conc
        """ Viral load (log10 scale) """
        ipred = @. log10(2*V) + 3
        """ Observed viral load """
        Virus ~ @. Normal(ipred, σ_add)
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
serialize(joinpath(ARTIFACTS_DIR, "fit_pd_foce.jls"), fit_pd_foce)
fit_pd_foce_loaded = deserialize(joinpath(ARTIFACTS_DIR, "fit_pd_foce.jls"))

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

# Predict on validation population for days 0 to 42
pred_pd_valid = predict(fit_pd_foce, pop_pd; obstimes=0:0.5:42)

# Plot predictions for validation subjects 1 to 12
plotgrid(
    pred_pd_valid[1:12],
    observation = :Virus,
    pred        = (; label = "model pred", linestyle=:dash),
    ipred       = (; label = "model ipred"),
    axis        = (; limits = ((0., 45.), nothing))
)

# =============================================================================
# End of the script
# =============================================================================