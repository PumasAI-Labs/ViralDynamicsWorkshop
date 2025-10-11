############################################################################################
# Workshop: Text Embeddings and Complex Covariates in NLME
# Goal: Load PK/PD data with text descriptions → Create embeddings → Link embeddings to 
#       patient parameters → Augment NLME model with predicted patient parameters
# Warning: This kind of integration is cutting-edge and does not yet have first-class support
#          It works, but here be dragons.
############################################################################################

############################
# 0) Environment & Packages
############################

# This time, we'll need packages that are not built-in to the app
# We manage package environments using Julia's built-in Pkg
using Pkg
Pkg.activate(@__DIR__() * "/../../lectures")
Pkg.instantiate()

using AlgebraOfGraphics
using DeepPumas            # Neural-embedded NLME modeling
using Pumas                # Core NLME functionality
using PumasUtilities       # Utilities for Pumas workflows
using CairoMakie           # Plotting backend
using CSV                  # Read CSV data
using DataFrames           # Dataframe operations
using DataFramesMeta       # Dataframe meta-programming
using Transformers         # Pre-trained language models
using Transformers.HuggingFace
using Transformers.TextEncoders
using MultivariateStats    # For PCA dimensionality reduction
using LinearAlgebra        # For matrix operations
using StableRNGs           # Stable RNG for reproducibility
using PumasPlots           # Plotting utilities

const AoG = AlgebraOfGraphics

# Load utility functions
include("utils.jl")

# Set neural network backend and plotting theme
set_mlp_backend(:staticflux)
set_theme!(deep_light())

# Create directory for outputs if it doesn't exist
ASSETS_DIR = joinpath(@__DIR__, "assests/")
if !isdir(ASSETS_DIR)
    mkdir(ASSETS_DIR)
end

############################################################################################
# 1) Load and explore the data
############################################################################################

# Load the patient data with text descriptions
# Each patient has a clinical text description in the "Description" field
DATA_PATH = joinpath(@__DIR__, "../..", "lectures", "data_prognostic_text.csv")
patient_data = CSV.read(DATA_PATH, DataFrame)

# Display structure
first(patient_data, 5)

# The data contains:
# - id: Patient identifier
# - time: Observation time
# - yPK, yPD: PK and PD observations
# - Description: Clinical text description of patient wellness
# - Score: Wellness score (0-10 scale)

# Create a Pumas population
pop = read_pumas(
    patient_data; 
    observations = [:yPK, :yPD], 
    covariates = [:Description, :Score]
)

# Split into training and test sets
train_pop = pop[1:100]
test_pop = pop[101:200]

# Visualize a few subjects
plotgrid(train_pop[1:8]; observation = :yPD)

@chain DataFrame(pop[1:8]) begin
    # Get the first row for each patient (as identified by :id)
    @by :id $first
    @select :id :Description
end

############################################################################################
# 2) Define and fit a baseline NLME model (without text embeddings)
############################################################################################

# Define a simple PK/PD indirect response model
model = @model begin
    @param begin
        # PK parameters
        tvKa ∈ RealDomain(; lower = 0, init = 0.5)
        tvCL ∈ RealDomain(; lower = 0, init = 1.0)
        tvVc ∈ RealDomain(; lower = 0, init = 1.0)
        # PD parameters
        tvSmax ∈ RealDomain(; lower = 0, init = 2.5)
        tvn ∈ RealDomain(; lower = 0, init = 1.5)
        tvSC50 ∈ RealDomain(; lower = 0, init = 0.05)
        tvKout ∈ RealDomain(; lower = 0, init = 2.0)
        tvKin ∈ RealDomain(; lower = 0, init = 1.0)
        # Between-subject variability (random effects)
        Ω ∈ PDiagDomain(5)
        # Residual error
        σ ∈ RealDomain(; lower = 0, init = 0.1)
        σ_pk ∈ RealDomain(; lower = 0, init = 0.02)
    end

    @random begin
        # Random effects for inter-individual variability
        η ~ MvNormal(Ω)
    end

    @pre begin
        # Apply random effects to parameters
        Ka = tvKa * exp(η[1])
        CL = tvCL
        Vc = tvVc * exp(η[2])
        Smax = tvSmax * exp(η[3])
        SC50 = tvSC50 * exp(η[4])
        Kout = tvKout * exp(η[5])
        Kin = tvKin
        n = tvn
    end

    @init begin
        # Initial PD response at steady-state
        R = Kin / Kout
    end

    @vars begin
        # Ensure non-negative PK concentration
        _cp = max(Central / Vc, 0.0)
        # Drug effect using Emax model
        EFF = Smax * _cp^n / (SC50^n + _cp^n)
    end

    @dynamics begin
        # PK compartments
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
        # PD compartment (indirect response)
        R' = Kin * (1 + EFF) - Kout * R
    end

    @derived begin
        # Observations with residual error
        yPK ~ @. Normal(Central / Vc, σ_pk)
        yPD ~ @. Normal(R, σ)
    end
end

# Fit the baseline model to training data
# Note: Using a subset of iterations for demonstration (increase for real analysis)
fitted_nlme = fit(
    model,
    train_pop,
    init_params(model),
    MAP(FOCE());
    optim_options = (; iterations = 100, show_trace = true)
)

# Get empirical Bayes estimates (EBEs) - these are the individual η values
η_train = empirical_bayes(fitted_nlme)

# EBEs capture patient-specific deviations from population parameters
# We'll now see if we can predict these from the text descriptions!

############################################################################################
# 3) Create text embeddings using a pre-trained language model
############################################################################################

# Load a pre-trained text embedding model from HuggingFace
# This model converts text into high-dimensional vectors that capture semantic meaning
println("Loading pre-trained embedding model...")
loaded_model = hgf"avsolatorio/NoInstruct-small-Embedding-v0"

# Extract the encoder and language model components
const encoder = loaded_model[1]
const llm = loaded_model[2]

# Define functions to get embeddings from text
function get_embedding(context::String)
    # Encode the text
    enc = encode(encoder, context)
    # Pass through language model
    out = llm(enc)
    # Return the pooled embedding vector
    return out.pooled
end

# For a Pumas Subject, extract the Description covariate
get_embedding(subj::Pumas.Subject) = get_embedding(subj.covariates(0).Description)

# Create embeddings for all patients
# This creates a matrix where each column is one patient's embedding
println("Creating embeddings for training set...")
X_train = mapreduce(get_embedding, hcat, train_pop)

println("Creating embeddings for test set...")
X_test = mapreduce(get_embedding, hcat, test_pop)

# Check embedding dimensions
println("Embedding dimension: $(size(X_train, 1))")
println("Number of training patients: $(size(X_train, 2))")

############################################################################################
# 4) Dimensionality reduction with PCA
############################################################################################

# The original embeddings are high-dimensional (typically 384-768 dimensions)
# Our clinical data likely lives on a lower-dimensional manifold
# Use PCA to find the most important directions

# Fit PCA on training embeddings, keeping 10 principal components
println("Fitting PCA...")
trained_pca = fit(PCA, X_train; maxoutdim = 10)

# Transform embeddings to principal component space
X_train_pc = predict(trained_pca, X_train)
X_test_pc = predict(trained_pca, X_test)

println("Reduced embedding dimension: $(size(X_train_pc, 1))")

# Optional: Visualize relationship between embeddings and wellness scores

# Create dataframe with first 4 principal components
npc = 4
pc_labels = ["PC$i" for i in 1:npc]
df_pc = DataFrame(X_train_pc[1:npc, :]', pc_labels)
df_pc.id = [subj.id for subj in train_pop]

# Get wellness scores from training population
df_scores = @chain DataFrame(train_pop) begin
    @by :id $first
    @select :id :Score
end

# Join PC embeddings with wellness scores
df = innerjoin(df_pc, df_scores; on = :id)

# Convert to long format for faceted plotting
df_long = stack(df, pc_labels; variable_name = :PC, value_name = :PC_value)

# Create faceted plot using AlgebraOfGraphics
plt = data(df_long) * 
      mapping(:Score => "Wellness Score", 
              :PC_value => "Principal Component Value"; 
              layout = :PC) * 
      visual(Scatter)

fig = draw(plt; figure = (; size = (800, 600)))

############################################################################################
# 5) Learn the relationship between embeddings and patient parameters (EBEs)
############################################################################################

# Now we train a neural network to predict EBEs (η) from the reduced embeddings
# This learns which aspects of the text are predictive of pharmacological parameters

# Create a convenience function to get PC embeddings for any subject
pc_embedder(subj::Pumas.Subject) = (; 
    id = subj.id,
    embeddings = vec(predict(trained_pca, get_embedding(subj))),
)

# Create a population with embeddings as covariates
_df = DataFrame(vcat(train_pop, test_pop))
embeddings = pc_embedder.(vcat(train_pop, test_pop))
_df = innerjoin(_df, DataFrame(embeddings); on = :id)

embedding_pop = read_pumas(
    _df; 
    observations = [:yPK, :yPD], 
    covariates = [:embeddings]
)

# Split back into train and test
tpope = embedding_pop[1:length(train_pop)]
vpope = embedding_pop[length(train_pop)+1:end]

# Preprocess to create targets for neural network training
# This extracts the η values we want to predict
target = preprocess(fitted_nlme.model, tpope, coef(fitted_nlme), FOCE(); covs=(:embeddings,))

# Define a neural network architecture
# Input: 10 PCs → Hidden layers: 15, 10 neurons → Output: 5 η values
nn = MLPDomain(
    numinputs(target),    # Number of input features (10 PCs)
    15, 10,               # Hidden layer sizes
    (numoutputs(target), identity);  # Output size and activation
    backend = :staticflux,
    act = tanh,           # Activation function
    reg = L1(1)           # L1 regularization to prevent overfitting
)

# Fit the neural network to predict EBEs from embeddings
println("Training neural network to predict patient parameters from embeddings...")
fnn = fit(
    nn, 
    target; 
    optim_options = (; loss = l2),
    training_fraction = 0.8  # Use 80% for training, 20% for validation
)

# Evaluate predictions on test set
vη = empirical_bayes(model, vpope, coef(fitted_nlme), FOCE())

# Compare predictions to computed, "true", EBEs on the validation data
pred_etas = mapreduce(vcat, vpope) do subj
    predicted_ebe = fnn(subj)[1]
    (; id = subj.id,  unroll(predicted_ebe)...)
end |> DataFrame

computed_etas = mapreduce(vcat, vpope) do subj
    ebe = empirical_bayes(model, subj, coef(fitted_nlme), FOCE())
    (; id = subj.id,  unroll(ebe)...)
end |> DataFrame

stacked_preds = stack(pred_etas, Not(:id), value_name = "prediction")
stacked_ebes = stack(computed_etas, Not(:id), value_name = "EBE")
_df = innerjoin(stacked_ebes, stacked_preds; on = [:id, :variable])

spec = data(_df) * mapping(:EBE, :prediction, layout=:variable)
draw(spec)

############################################################################################
# 6) Augment the NLME model with embedding-based predictions
############################################################################################

# Now we combine the traditional NLME model with our embedding-based predictions
# The augmented model uses: η_augmented = η + η_predicted
# This reduces unexplained variability by incorporating information from text

using Setfield
@set! fitted_nlme.data = tpope

# Create augmented model
@time deep_fpm = augment(fitted_nlme, fnn)

# Refit to estimate residual Ω (now smaller because embeddings explain some variability)
fit_deep = fit(
    deep_fpm.model,
    tpope,
    coef(deep_fpm),
    MAP(FOCE());
    constantcoef = Tuple(setdiff(keys(coef(deep_fpm)), (:Ω,)))
)

# Compare predictions on test set
pred_original = predict(fitted_nlme, vpope)
pred_augmented = predict(deep_fpm, vpope)

# Calculate performance metrics
let
    df_orig = DataFrame(pred_original)
    df_aug = DataFrame(pred_augmented)
    
    # Mean Absolute Error for PD predictions
    mae_orig = mean(abs.(df_orig.yPD .- df_orig.yPD_pred))
    mae_aug = mean(abs.(df_aug.yPD .- df_aug.yPD_pred))
    
    # Correlation for PD predictions
    r2_orig = cor(df_orig.yPD, df_orig.yPD_pred)^2
    r2_aug = cor(df_aug.yPD, df_aug.yPD_pred)^2
    
    println("\n=== Model Performance on Test Set ===")
    println("Original Model:")
    println("  MAE: $(round(mae_orig, digits=3))")
    println("  R²: $(round(r2_orig, digits=3))")
    println("\nAugmented Model (with embeddings):")
    println("  MAE: $(round(mae_aug, digits=3))")
    println("  R²: $(round(r2_aug, digits=3))")
    println("\nImprovement:")
    println("  MAE reduction: $(round(100*(mae_orig - mae_aug)/mae_orig, digits=1))%")
    println("  R² increase: $(round(100*(r2_aug - r2_orig)/r2_orig, digits=1))%")
end

# Visualize predictions for first 12 test subjects
pred_plot = predict(deep_fpm, vpope[1:12]; obstimes = 0:0.1:10)
fig = plotgrid(pred_plot; ipred = false, observation = :yPD)
save(joinpath(ASSETS_DIR, "augmented_predictions.png"), fig)
display(fig)

############################################################################################
# 7) Verify improved population predictions with VPC
############################################################################################

# Visual Predictive Check (VPC) compares observed data to model simulations
# A good model should capture the observed variability

println("\nGenerating VPCs...")

# VPC for original model
vpc_original = vpc(fitted_nlme, vpope; observations = [:yPD])
fig_vpc_original = vpc_plot(vpc_original; figure = (; size = (900, 300)))
save(joinpath(ASSETS_DIR, "vpc_original.png"), fig_vpc_original)

# VPC for augmented model
vpc_augmented = vpc(fit_deep; observations = [:yPD])
fig_vpc_augmented = vpc_plot(vpc_augmented; figure = (; size = (900, 300)))
save(joinpath(ASSETS_DIR, "vpc_augmented.png"), fig_vpc_augmented)

display(fig_vpc_original)
display(fig_vpc_augmented)

println("\nExercise complete! Check the assests/ folder for all generated plots.")

############################################################################################
# Summary
############################################################################################

# What we learned:
# 1. How to create text embeddings from clinical descriptions using pre-trained models
# 2. How to reduce dimensionality with PCA to focus on relevant information
# 3. How to train a neural network to predict patient-specific parameters from embeddings
# 4. How to augment traditional NLME models with embedding-based predictions
# 5. How this approach reduces unexplained variability and improves predictions
#
# Key insight: Complex covariates like text can be converted to embeddings and 
# integrated into NLME modeling to capture information that traditional covariates miss!
