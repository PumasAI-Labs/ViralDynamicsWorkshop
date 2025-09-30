using PumasUtilities

include("04-predict.jl")

# Create a second model for comparison, this time a two-compartment PK model
model_2cmt = @model begin
    @metadata begin
        desc = "2-compartment model"
        timeu = u"hr"
    end

    @param begin
        """
        Clearance (L/hr)
        """
        tvcl ∈ RealDomain(; lower = 0)
        """
        Volume Central Compartment (L)
        """
        tvvc ∈ RealDomain(; lower = 0)
        """
        Intercompartmental Clearance (L/hr)
        """
        tvq ∈ RealDomain(; lower = 0)
        """
        Volume Peripheral Compartment (L)
        """
        tvvp ∈ RealDomain(; lower = 0)
        """
          - ΩC
          - ΩV
        """
        Ω ∈ PDiagDomain(2)
        """
        Additive RUV
        """
        σ_add ∈ RealDomain(; lower = 0)
        """
        Proportional RUV
        """
        σ_prop ∈ RealDomain(; lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
        Q = tvq
        Vp = tvvp
    end

    @dynamics Central1Periph1

    @derived begin
        CONC := @. Central / Vc
        DV ~ @. Normal(CONC, sqrt(CONC^2 * σ_prop^2 + σ_add^2))
    end
end

params_2cmt = (;
    tvvc = 5,
    tvcl = 0.02,
    tvq = 0.01,
    tvvp = 10,
    Ω = Diagonal([0.01, 0.01]),
    σ_add = 0.1,
    σ_prop = 0.1,
)

fit_2cmt = fit(model_2cmt, pop, params_2cmt, FOCE())


fit_1cmt = fit_foce # Make the name more clear

# You can get all model metrics from a fit result with metrics_table
# It returns a DataFrame
metrics_table(fit_1cmt)
metrics_table(fit_2cmt)

# Additionally, everything in metrics_table can be individually retrieved with specialized functions
loglikelihood(fit_1cmt)
aic(fit_1cmt)
bic(fit_1cmt)
ηshrinkage(fit_1cmt)
ϵshrinkage(fit_1cmt)

# One highlight is the log-likelihood function
# It can compute a loglikelihood for any model given any population, parameter values, and estimation method
# This is helpful for model conversions from other software/tools
loglikelihood(
    model,
    pop,
    (; # NamedTuple of parameter values
        tvcl = 0.2,
        tvvc = 5,
        Ω = Diagonal([0.1, 0.1]),
        σ_add = 0.01,
        σ_prop = 0.05,
    ),
    FOCE(),
)

# You can use the inspect function to calculate in one go:
# - population predictions (pred) and individual predictions (ipred)
# - weighted residuals (wres)
# - empirical bayes estimates (ebe)
# - individual parameters (icoef)
# - dose control parameters (dcp), if applicable
# It takes as input a single fitted Pumas model
inspect_1cmt = inspect(fit_1cmt)
inspect_2cmt = inspect(fit_2cmt)

# There are several plotting functions available in Pumas
# We will not cover all of them but one deserves a highlight: goodness_of_fit
# It takes a Pumas result from inspect and outputs a 4-panel plot with:
# 1. observations versus pred
# 2. observation versus ipred
# 3. wres versus time
# 4. wres versus ipred
goodness_of_fit(inspect_1cmt)
goodness_of_fit(inspect_2cmt)


# We can overlay individual plots for additional comparisons
plt = plotgrid(preds_dense[1:6])
predict_2cmt_dense = predict(fit_2cmt; obstimes=0:0.5:180)
# The ! in a function name indicates that something is being modified rather than created.
plotgrid!(
    plt,
    predict_2cmt_dense[1:6]; 
    pred = (; color=:green, label = "2cmt pred", linestyle=:dash),
    ipred = (; color=:red, label = "2cmt ipred", linestyle=:dash),
    )