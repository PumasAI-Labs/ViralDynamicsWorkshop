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
DATA_PATH = joinpath(ARTIFACTS_DIR, "hiv-ipp-data-long.csv")

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

# Split into training and validation populations
_tpop = pop_pd[1:20]   # Training set: first 20 subjects
_vpop = pop_pd[21:30] # Validation set: subjects 41 onward

# Visualize the first 8 training subjects
plotgrid(_tpop[1:8]; data = (; color=:blue))

############################################
# 3) Define the DeepNLME HIV Model 
############################################
mdl_fit = deserialize(joinpath(ARTIFACTS_DIR, "hiv_pkpd_nn.jls"))

# Predict on validation population for days 0 to 56
model_pred = predict(mdl_fit, _vpop; obstimes=0:1:120)

# Plot predictions for validation subjects 11 to 30
plotgrid(
    model_pred[1:12],
    observation = :Virus,
    pred = (; label = "model pred", linestyle=:dash),
    ipred = (; label = "model ipred"),
    axis = (; limits = ((0., 120.), nothing))
)


############################################
# 3) Define the DeepNLME HIV Model 
############################################

nn_hiv2 = @model begin
    @metadata begin
        desc = "HIV PD Only"
        timeu = u"d" # day
    end
    @param begin
        # Neural networks for different mechanistic components
        NN1 ∈ MLPDomain(5, 4, 3, (1, identity); reg = L2(1.0))
        NN2 ∈ MLPDomain(4, 2, 2, (1, identity); reg = L2(1.0))
        NN3 ∈ MLPDomain(4, 2, 2, (2, identity); reg = L2(1.0))
        # NN4 ∈ MLPDomain(3, 2, 2, (1, identity); reg = L2(1.0))

        # PKPD parameters
        tvR0   ∈ RealDomain(lower=1.0, init=6.5)
        tvλ    ∈ RealDomain(lower=0., init=0.46236)
        tvd    ∈ RealDomain(lower=0., init=0.006)
        tvic50 ∈ RealDomain(lower=0., init=200.0)
        tvap    ∈ RealDomain(lower=0., init=0.63911)
        tvas    ∈ RealDomain(lower=0., init=0.63911)
        tvu    ∈ RealDomain(lower=0., init=0.011)
        Ω      ∈ PDiagDomain(4)
        ωR0    ∈ RealDomain(lower=0.)
        σ_add  ∈ RealDomain(lower=0., init=0.11)
    end
    @random begin
        η ~ MvNormal(Ω)
        η_nn1 ~ MvNormal(2, 0.1)
        η_nn2 ~ MvNormal(2, 0.1)
    end
    @covariates iKa iCL iVc iQ iVp iDur
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
        ap  = tvap * exp(η[2])
        as  = tvas 
        u  = tvu * exp(η[3])
        IC50 = tvic50 * exp(η[4])

        # Initial conditions for T and V
        T0 = λ / d / R0
        V0 = λ / u * (1 - 1 / R0)

        # Fix neural networks with random effects
        iNN_INF   = fix(NN1, η_nn1)
        iNN_GATE  = fix(NN2, η_nn1)
        # iNN_PRODP = fix(NN3, η_nn2)
        # iNN_PRODS = fix(NN4, η_nn2)
        iNN_PROD = fix(NN3, η_nn2)

        # Gate for initial infected populations
        z0   = iNN_GATE(T0, V0)[1]
        f_s0 = 1.0 / (1.0 + exp(-z0))
        f_p0 = 1.0 - f_s0
    end
    @dosecontrol begin
        duration = (; Depot = iDur)
    end
    @init begin
        # Initial conditions for HIV dynamics
        T  = λ / d / R0
        Ip = f_p0 * λ / ap * (1 - (1 / R0))
        Is = f_s0 * λ / as * (1 - (1 / R0))
        V  = λ / u * (1 - (1 / R0))
    end
    @vars begin
        # Calculate concentration and inhibition
        Conc = Central / Vc / 1000
        INH = Conc / (Conc + IC50)
        β = R0 * d / λ / V

        # Gate for infected populations
        zgate = iNN_GATE(T, V)[1]
        f_s  = 1.0 / (1.0 + exp(-zgate))   # sigmoid
        f_p  = 1.0 - f_s

        # Infection and production terms
        infect_term = iNN_INF(T, V, (1 - INH))[1]
        prodp = iNN_PROD(Ip, Is)[1]
        prods = iNN_PROD(Ip, Is)[2]
    end
    @dynamics begin
        # PK: Two-compartment model with first-order absorption
        Depot'   = -Ka * Depot
        Central' = Ka * Depot + (Q / Vp) * Periph - (Q / Vc) * Central - CL / Vc * Central
        Periph'  = -(Q / Vp) * Periph + (Q / Vc) * Central

        # HIV dynamics
        T'  = λ - d * T - infect_term
        Ip' = f_p * infect_term - ap * Ip
        Is' = f_s * infect_term - as * Is
        V'  = prodp + prods - u * V
    end
    @derived begin
        Concentration = Conc
        ipred = @. log10(2*abs(V)) + 3
        Virus ~ @. Normal(ipred, σ_add)
    end
end



nn_hiv = @model begin
    @metadata begin
        desc = "DeepNLME HIV model"
        timeu = u"d" # day
    end
    @param begin
        # Neural networks for PD dynamics
        NN1 ∈ MLPDomain(5, 2, 2, (1, identity); reg = L2(1.0))
        NN2 ∈ MLPDomain(3, 4, 4, (2, identity); reg = L2(1.0))
        # PKPD parameters
        tvR0   ∈ RealDomain(lower=1.0, init=6.5)
        tvλ    ∈ RealDomain(lower=0., init=0.46236)
        tvd    ∈ RealDomain(lower=0., init=0.006)
        tvic50 ∈ RealDomain(lower=0., init=200.0)
        tvap    ∈ RealDomain(lower=0., init=0.63911)
        tvas    ∈ RealDomain(lower=0., init=0.011)
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
    @covariates iKa iCL iVc iQ iVp iDur
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
        ap  = tvap * exp(η[2])
        as  = tvas 
        u  = tvu * exp(η[3])
        IC50 = tvic50 * exp(η[4])

        # Fix neural networks with random effects
        iNN_INF = fix(NN1, η_nn1)
        iNN_VIR = fix(NN2, η_nn2)


        # Initial conditions for T and V
        T0 = λ / d / R0
        V0 = λ / u * (1 - 1 / R0)

        Ip_s0 = iNN_INF(T0, V0, 0.0)[1]
        Is_s0 = iNN_INF(T0, V0, 0.0)[2]
    end
    @dosecontrol begin
        duration = (; Depot = iDur)
    end
    @init begin
        # Initial conditions for HIV dynamics
        T = λ / d / R0
        # I = λ / a * (1 - (1 / R0))

        Ip = Ip_s0
        Is = Is_s0

        V = λ / u * (1 - (1 / R0))
    end
    @vars begin
        # Calculate concentration and inhibition
        Conc = Central / Vc / 1000
        INH = Conc / (Conc + IC50)
        β = R0 * d / λ / V
        infect_termP = iNN_INF(T, V, (1 - INH))[1]
        infect_termS = iNN_INF(T, V, (1 - INH))[2]
        prod_term   = iNN_VIR(Ip, Is)[1]
    end
    @dynamics begin
        # PK: Two-compartment model with first-order absorption
        Depot'   = -Ka * Depot
        Central' = Ka * Depot + (Q / Vp) * Periph - (Q / Vc) * Central - CL / Vc * Central
        Periph'  = -(Q / Vp) * Periph + (Q / Vc) * Central
        # HIV dynamics
        T' = λ - d * T - infect_termP - infect_termS
        Ip' = infect_termP - ap * Ip
        Is' = infect_termS - as * Is
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
# loglikelihood(nn_hiv, _tpop, init_params(nn_hiv2), FOCE())

# ---------------------------
# 4. Fit the Model
# ---------------------------

fit_pkpd = fit(
    nn_hiv2,
    _tpop,
    init_params(nn_hiv2),
    MAP(FOCE()),
    # JointMAP(),
    optim_options = (; iterations=100, f_tol=1e-6),
)

# Save and reload the fit for reproducibility
serialize(joinpath(ARTIFACTS_DIR, "hiv_pkpd_nn_d2.jls"), fit_pkpd)
mdl_fit = deserialize(joinpath(ARTIFACTS_DIR, "hiv_pkpd_nn_d2.jls"))

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

# Predict on validation population for days 0 to 56
model_pred = predict(fit_pkpd, _vpop; obstimes=0:1:120)

# Plot predictions for validation subjects 11 to 32
plotgrid(
    model_pred[1:8],
    observation = :Virus,
    pred = (; label = "model pred", linestyle=:dash),
    ipred = (; label = "model ipred"),
    axis = (; limits = ((0., 120.), nothing))
)