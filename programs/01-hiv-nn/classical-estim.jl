using Pumas, Random, DataFrames, CSV, Distributions, DataFramesMeta

cd(@__DIR__)

hivPKmodel=@model begin
    @metadata begin
        desc = "HIV PK"
        timeu = u"d" # day
    end
 @param begin
    tvka ∈ RealDomain(lower=0.0001)
    tvcl ∈ RealDomain(lower=0.0001)
    tvvc ∈ RealDomain(lower=0.001)
    tvq ∈ RealDomain(lower=0.0001)
    tvvp ∈ RealDomain(lower=0.0001) 
    tvd ∈ RealDomain(lower=0.0001)
    
    Ω ∈ PDiagDomain(5)
    
    σ_proppk ∈ RealDomain(lower=0)
    σ_addpk ∈ RealDomain(lower=0)
    end
  @random begin
    η ~ MvNormal(Ω)
 end
 #@covariates N
 @pre begin
    Ka = tvka * 24 * exp(η[1]) #KA_0*24  1/day
    CL =  tvcl * 24 * exp(η[2]) #CL_0*24  L/day
    Vc = tvvc * exp(η[3]) #V2_0
    Q  = tvq * 24 * exp(η[4]) #Q_0*24   L/day 
    Vp = tvvp #V3_0  
    
  end
    
    @dosecontrol begin
        duration = (;
        Depot = (tvd/24) * exp(η[5])
        )
        #sequential zero and first order abs codes
        end

  @init begin
    AUC = 0
    end

  @vars begin
    Conc = (Central/(Vc/1000))      # conc. in ng/mL drives viral inhibition
    end

    @dynamics begin
    Depot'= -Ka*Depot
    Central'= Ka*Depot + (Q/Vp)*Periph - (Q/Vc)*Central - CL/Vc*Central
    Periph'= -(Q/Vp)*Periph + (Q/Vc)*Central
    
    AUC' = (Central/(Vc/1000))
  end
  
  @derived begin

    Concentration ~ @. (Normal(Conc, sqrt(σ_addpk ^ 2 + (abs(Conc) * σ_proppk) ^ 2)))
  end
end

PKparams = (
    tvka= 0.408, 
    tvcl= 1.63,
    tvvc= 74.3,
    tvq = 0.989,
    tvvp= 4.24,
    tvd= 2.83,
    
    Ω = Diagonal([
        0.2798, 0.128, 0.1722, 0.25, 0.1544
        ]),
    σ_proppk = 0.190,
    σ_addpk = 17.3)

pkData = CSV.read(@__DIR__() *"/hiv_datax.csv" , DataFrame; missingstring = "", stringtype = String)

_popPK = read_pumas(
    # @rsubset DataFrame(pkData) :time < 3;
    pkData;
    id = :id,
    time = :time,
    observations = [:Concentration], #
    evid = :evid,
    amt = :amt,
    cmt = :cmt    
)


pk_fit = fit(hivPKmodel, _popPK, PKparams, FOCE();)



mdl_insp = inspect(pk_fit)

# ### shrinkage
# etas_shrink = ηshrinkage(mdl_fit)
# eps_shrink = ϵshrinkage(mdl_fit)



# ### Diagnostic plots
# # -----------------------------------------------------------------------------------------------------------

# ### Overall key GOFs:
# goodness_of_fit(mdl_insp)

# ### GOFs:
# observations_vs_ipredictions(mdl_insp)
# observations_vs_predictions(mdl_insp)
# wresiduals_vs_time(mdl_insp)
# #wresiduals_vs_covariates(mdl_insp)
# wresiduals_vs_predictions(mdl_insp)
# wresiduals_dist(mdl_insp)
# empirical_bayes_dist(mdl_insp)



####

df_inspect = DataFrame(mdl_insp);
vscodedisplay(df_inspect)
# Obtain the individual PK parameters
icoef_dataframe = unique(df_inspect[!, [:id, :time, :Ka, :CL, :Vc, :Q, :Vp, :duration_Depot ]], :id)

rename!(icoef_dataframe, 
                        # :id => :ID, 
                        # :time => :TIME,
                        :Ka => :xKa,
                        :CL => :xCL,
                        :Vc => :xVc,
                        :Q => :xQ,
                        :Vp => :xVp,
                        :duration_Depot => :xDur);
first(icoef_dataframe, 5)

icoef_dataframe.id = parse.(Int64, string.(icoef_dataframe.id))

# Merge the individual PK parameters with the original dataset by ID and TIME
pd_dataframe = outerjoin(pkData, icoef_dataframe; on = [:id, :time])
# Arrange the DataFrame by ID and TIME

# Force :id to Float64


# sort by :id, :time
sort!(pd_dataframe, [:id, :time])

# Return the first 5 rows
first(pd_dataframe, 5)

vscodedisplay(pd_dataframe)

####################################################################################################################
####################################################################################################################

pdOnly = @chain pd_dataframe begin
    @rsubset(:FLG == 2)
end

 # 

function forward_fill_columns!(df::DataFrame, cols::Vector{Symbol})
    for col in cols
        last_val = missing
        for i in 1:nrow(df)
            if !ismissing(df[i, col])
                last_val = df[i, col]
            elseif !ismissing(last_val)
                df[i, col] = last_val
            end
        end
    end
    return df
end

# Example usage
forward_fill_columns!(pdOnly, [:xKa, :xCL, :xVc, :xQ, :xVp, :xDur])


vscodedisplay(pdOnly)

CSV.write("hiv-ipp-datax.csv",pdOnly)

_popPD = read_pumas(
    pdOnly;
    id = :id,
    time = :time,
    observations = [:DV], 
    evid = :evid,
    amt = :amt,
    cmt = :cmt,
    covariates = [:xKa, :xCL, :xVc, :xQ, :xVp, :xDur],    
)



hivPDmodel=@model begin
    @metadata begin
        desc = "HIV PD Only"
        timeu = u"d" # day
    end
 @param begin
    tvpro ∈ RealDomain(lower=1)
    tvdelta ∈ RealDomain(lower=0)
    tvlambda ∈ RealDomain(lower=0)
    tvic50 ∈ RealDomain(lower=0)

    Ω ∈ PDiagDomain(4)
    σ_add ∈ RealDomain(lower=0)

    end
  @random begin
    η ~ MvNormal(Ω)
 end
 @covariates xKa xCL xVc xQ xVp xDur

 @pre begin
    Ka = xKa
    CL = xCL
    Vc = xVc
    Q  = xQ
    Vp = xVp

    RR0= tvpro * exp(η[1]) 		#(if <1, you wouldn't stay infected)
    DELTA = tvdelta * exp(η[2])	
    LAMBDA =tvlambda * exp(η[3])
    IC50 = tvic50 * exp(η[4])
    DU = 0.006			        #uninfected cell death rate (1/d)
    DL = 0.04			        #latently infected cell death rate (1/d)
    AL = 0.036			        #rate of conversion from latently to actively infected (1/d)
    #PROD  = 1240			        ;virus production rate (virions/cell/day)
    POVC= 35.4			        #ratio of production rate to clearance rate of virus (Funk 2001: p=1240, c=35)
    DLL= 0.01			#long-lived infected cell death rate (1/day) ; important for long term
    PLLC= 0.374			      #ratio of birth to death rate of virus for long-lived infected cells
    QLL= 0.001				#fraction of infected cells long-lived infected
    QA = 0.97			        #fraction of infected cells actively infected
    QL = 0.029			        #fraction of infected cells latently infected
  end
    
    @dosecontrol begin
        duration = (;
        Depot = xDur,
        )
        #sequential zero and first order abs codes
        end

  @init begin
    #Depot = 0 # equivalent to A_0(1) # not needed for PK, but needed for cells & virus
    #Central= 0 # equivalent to A_0(2)
    #Periph = 0 # equivalent to A_0(3)
    UNINFECTED = LAMBDA/DU/RR0				 	#uninfected cells # equivalent to A_0(4)
    ACTIVEIC = (QA+QL*AL/(DL+AL)) * LAMBDA/DELTA  * (1-(1/RR0)) #actively infected cells# equivalent to A_0(5)
    LATENT = QL* LAMBDA/(DL+AL)* (1-(1/RR0)) #latently infected cells# equivalent to A_0(6)
    LLIC = QLL* LAMBDA/DLL* (1-(1/RR0)) #long-lived infected cells# equivalent to A_0(7)
    # A_0(8) = POVC*A_0(5) + PLLC*A_0(7) ;virus
    #V = 4.54 #log10

    AUC = 0
    end

  @vars begin
    LFAC= QA+QL*AL/(AL+DL)	    	#fraction (slightly <1) for presence of latently infected cells
    BETA = RR0*DU / LAMBDA / (POVC*LFAC/DELTA + PLLC*QLL/DLL) #set beta (1/(virions/ul)/day) from RR0
    #from $des
    Conc = (Central/(Vc/1000))      # conc. in ng/mL drives viral inhibition
    INH = Conc/(Conc+IC50)
    V= abs(POVC*ACTIVEIC + PLLC*LLIC)   #INR= V*BETA
 end

    @dynamics begin
    Depot'= -Ka*Depot
    Central'= Ka*Depot + (Q/Vp)*Periph - (Q/Vc)*Central - CL/Vc*Central
    Periph'= -(Q/Vp)*Periph + (Q/Vc)*Central
    UNINFECTED'= LAMBDA-BETA*V*(1-INH)*UNINFECTED - DU*UNINFECTED                     # uninfected cells
    ACTIVEIC'= QA *BETA*V *(1-INH)*UNINFECTED - DELTA* ACTIVEIC + AL*LATENT                              # actively infected cells
    LATENT'=  QL *BETA* V * (1-INH)*UNINFECTED - DL *LATENT - AL*LATENT	                             # latently infected cells
    LLIC'=   QLL*BETA * V *(1-INH) *UNINFECTED - DLL* LLIC                                 # long-lived infected cells
    # V = 1240 * ACTIVEIC + 13.09 * LLIC - 35*Virus #virus

    AUC' = (Central/(Vc/1000))
  end
  
  @derived begin

    Concentration = Conc

    Virus = @.log10(2*V)+3
    DV ~ @. Normal(Virus, σ_add)

  end
end

PDparams = (
    
    tvpro = 6.8448,	#(if <1, you wouldn't stay infected)
    tvdelta = 0.63911,
    tvlambda = 0.46236 ,
    tvic50 = 324.59,

    
    Ω = Diagonal([
        0.13873, 0.47798, 0.51177, 0.97883
        ]),
    σ_add = 0.318)


    
pdData = CSV.read(@__DIR__() *"/hiv-ipp-datax.csv" , DataFrame; missingstring = "", stringtype = String)

# vscodedisplay(pdData)

_pop = read_pumas(
    #@rsubset DataFrame(pdData) :time < 28;
    pdData;
    id = :id,
    time = :time,
    observations = [:DV], #, :Concentration
    evid = :evid,
    amt = :amt,
    cmt = :cmt,
    covariates = [:xKa, :xCL, :xVc, :xQ, :xVp, :xDur],     
)

_tpop = _pop[1:40]
_vpop = _pop[41:end]




    pd_fit = fit(hivPDmodel, _tpop, PDparams, FOCE();)




### Check convergence:
convergence_trace(pd_fit) 

### Results and Model parameters:
# -----------------------------------------------------------------------------------------------------------
### Export model results:
serialize(joinpath("pd_fit.jls"),pd_fit)
mdl_fit=deserialize("pd_fit.jls")

mdl_insp = inspect(pd_fit)

### shrinkage
etas_shrink = ηshrinkage(mdl_fit)
eps_shrink = ϵshrinkage(mdl_fit)



### Diagnostic plots
# -----------------------------------------------------------------------------------------------------------

### Overall key GOFs:
goodness_of_fit(mdl_insp)

### GOFs:
observations_vs_ipredictions(mdl_insp)
observations_vs_predictions(mdl_insp)
wresiduals_vs_time(mdl_insp)
#wresiduals_vs_covariates(mdl_insp)
wresiduals_vs_predictions(mdl_insp)
wresiduals_dist(mdl_insp)
empirical_bayes_dist(mdl_insp)






model_pred = predict(pd_fit, _vpop; obstimes=0:1:42)


plotgrid(model_pred[1:8], observation = :DV,
  pred = (; label = "model pred",
    linestyle=:dash),
  ipred = (; label= "model ipred"),
  axis = (; limits = ((0., 45.),nothing))
)

