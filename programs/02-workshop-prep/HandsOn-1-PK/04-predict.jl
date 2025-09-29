include("03-fit.jl")

# Our original data has observations from 0 to 168 hours
unique(pkdata.time)

# predict generate pred/ipred and can take either:
# - fit result
# - fit result, population
# - model, population, parameters
preds = predict(fit_foce)
preds = predict(fit_foce, pop)
preds = predict(hivPKmodel, pop, coef(fit_foce))

DataFrame(preds)

# By default it will generate pred/ipred using the original data observation time profile
# Suppose you want a more richer/denser pred/ipred time profile
# You can do that with the keyword argument obstimes
# it will "extend" the original observation profile to encompass the desired obstimes
preds_custom = predict(fit_foce; obstimes = 168:172)
DataFrame(preds_custom)

# We can also plot the predictions
plotgrid(preds[1:6])

# Extending the observation times provides more detail on what happens between the 
# timepoints for which we have data
preds_dense = predict(fit_foce; obstimes = 0:172)
plotgrid(preds_dense[1:6])
