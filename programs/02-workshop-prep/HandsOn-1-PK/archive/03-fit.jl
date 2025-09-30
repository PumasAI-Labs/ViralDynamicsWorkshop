include("02-model.jl")

# Initial parameter values
params = (
    tvka= 0.408, 
    tvcl= 1.63,
    tvvc= 74.3,
    tvq = 0.989,
    tvvp= 4.24,
    tvd= 2.83,
    
    Ω = Diagonal([
        0.2798, 0.128, 0.1722, 0.1544 #0.25, 
        ]),
    σ_proppk = 0.190,
    σ_addpk = 17.3)

# Fit a the model with FOCE
fit_foce = fit(hivPKmodel, pop, params, FOCE())

# Fit a the model with NaivePooled
fit_naivepooled = fit(hivPKmodel, pop, params, NaivePooled(); omegas = (:Ω,))

# Fit a the model with LaplaceI
fit_laplace = fit(hivPKmodel, pop, params, LaplaceI())

# Fit a the model with FOCE and fixed parameters
fit_foce_fixed = fit(hivPKmodel, pop, params, FOCE(); constantcoef = (:tvcl,))

# Get a NamedTuple of the estimated parameter values
coef(fit_foce)
coef(fit_naivepooled)

# Get a DataFrame of the estimated parameter values
coeftable(fit_foce)
coeftable(fit_naivepooled)

# Get the icoefs from the model fit as a DataFrame
DataFrame(icoef(fit_foce))
