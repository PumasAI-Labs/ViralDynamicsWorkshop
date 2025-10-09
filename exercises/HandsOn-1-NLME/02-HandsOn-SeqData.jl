# =============================================================================
# Workshop: Fitting Dynamical models using Pumas (Satellite Course)
# Hands On-1.2 : Preparation of PKPD dataset for Sequential PKPD modeling (IPP approach)
# Goal: Read final PK model → extract individual params → combine and generate PKPD dataset
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
using Chain                      # for @chain
using LinearAlgebra              # Diagonal()
using Unitful                    # u"d" etc.
using Random                     # reproducibility
using Serialization              # serialize/deserialize fits
# using PharmaDatasets           # optional helper (not required)

# Choose a readable theme; try deep_dark() if presenting on dark slides
set_theme!(deep_light())

# Define artifacts path
ASSESTS_DIR = joinpath(@__DIR__, "assests")


########################################
# 1) Load final PK model and dataset
########################################

# Expect a file `hiv-pkpd-data.csv` in the same folder as this script
DATA_PATH = joinpath(@__DIR__, "hiv-pkpd-data.csv")

# Load PK dataset
df_pk_seq = CSV.read(DATA_PATH, DataFrame; missingstring = "", stringtype = String)

df_pk_seq = @chain df_pk begin
    @rtransform(:rate = :evid == 1 ? -2 : missing)
  end

# Load previously fitted PK model (FOCE method) from artifacts
fit_pkseq_foce = deserialize(joinpath(ASSESTS_DIR, "fit_pkseq_foce.jls"))

# Inspect the fit object (optional)
fit_pkseq_foce

########################################
# 2) Extract individual parameters
########################################

# Inspect the fit to extract individual-level estimates
insp_pkseq = inspect(fit_pkseq_foce)
df_inspect = DataFrame(insp_pkseq)

# Visualize the inspect object
vscodedisplay(df_inspect)

# Select unique individual PK parameters
icoef_dataframe = unique(df_inspect[!, [:id, :Ka, :CL, :Vc, :Q, :Vp, :duration_Depot]], :id)

# Rename columns to indicate individual estimates (prefix 'i')
rename!(icoef_dataframe, 
    :Ka => :iKa,
    :CL => :iCL,
    :Vc => :iVc,
    :Q => :iQ,
    :Vp => :iVp,
    :duration_Depot => :iDur
)

# Preview first 5 rows
first(icoef_dataframe, 5)

# Ensure ID is integer type
icoef_dataframe.id = parse.(Int64, string.(icoef_dataframe.id))

# Merge PK parameters with original dataset by ID
pd_dataframe = outerjoin(df_pk_seq, icoef_dataframe; on = [:id])

# Display merged dataset (optional)
vscodedisplay(pd_dataframe)

########################################
# 3) Save PKPD Dataset
########################################

CSV.write(joinpath(ASSESTS_DIR, "hiv-ipp-data.csv"), pd_dataframe)

# =============================================================================
# End of the Script
# =============================================================================
