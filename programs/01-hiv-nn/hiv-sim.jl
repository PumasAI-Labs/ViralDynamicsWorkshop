# Synthetic HIV PK–PD data generator (Pumas v2.6.1 / DeepPumas)
# ------------------------------------------------------------------
# This script simulates population PK and HIV viral dynamics 
# under once-daily oral dosing. Drug effect inhibits infection rate (β)
# via an Emax model so we can later swap it for a neural effect (DeepPumas).
#
# Outputs:
#   - pkpd_hiv_sim_short : subject-level pk/pd/dosing info
#
# Author: Sreenath
# Date: 2025-08-08

using Pumas, Random, DataFrames, CSV, Distributions

# ---------------------------------------
# 1) Structural model (PK + T–I–V HIV PD)
# ---------------------------------------
# States: Depot, Central (PK), CD4 target, infected, virus
# Drug effect INH(C) acts on infection rate β: β_eff = (1 - INH)*β
# INH = (Emax * C) / (EC50 + C); default Emax=1


hivPKPD_model=@model begin
    @metadata begin
        desc = "HIV PKPD"
        timeu = u"d" # day
    end
 @param begin
    tvka ∈ RealDomain(lower=0.0001)
    tvcl ∈ RealDomain(lower=0.0001)
    tvvc ∈ RealDomain(lower=0.001)
    tvq ∈ RealDomain(lower=0.0001)
    tvvp ∈ RealDomain(lower=0.0001) 
    tvd ∈ RealDomain(lower=0.0001)
    tvpro ∈ RealDomain(lower=1)
    tvdelta ∈ RealDomain(lower=0)
    tvlambda ∈ RealDomain(lower=0)
    tvic50 ∈ RealDomain(lower=0)
    #tvid50 ∈ RealDomain(lower=0)
    #tvgsklag ∈ RealDomain(lower=0)
    Ω ∈ PDiagDomain(9)
    σ_add ∈ RealDomain(lower=0)
    σ_proppk ∈ RealDomain(lower=0)
    σ_addpk ∈ RealDomain(lower=0)
    end
  @random begin
    η ~ MvNormal(Ω)
 end
 #@covariates N
 @pre begin
    Ka = tvka * 24 * exp(η[5]) #KA_0*24  1/day
    CL =  tvcl * 24 * exp(η[6]) #CL_0*24  L/day
    Vc = tvvc * exp(η[7]) #V2_0
    Q  = tvq * 24 * exp(η[8]) #Q_0*24   L/day 
    Vp = tvvp #V3_0  
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
        Depot = (tvd/24) * exp(η[9])
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

    Concentration ~ @. (Normal(Conc, sqrt(σ_addpk ^ 2 + (abs(Conc) * σ_proppk) ^ 2)))

    Virus = @.log10(2*V)+3
    DV ~ @. Normal(Virus, σ_add)

  end
end

# ------------------------------------
# 2) Typical values and variability
# ------------------------------------
params = (
    tvka= 0.408, 
    tvcl= 1.63,
    tvvc= 74.3,
    tvq = 0.989,
    tvvp= 4.24,
    tvd= 2.83,
    tvpro = 6.8448,	#(if <1, you wouldn't stay infected)
    tvdelta = 0.63911,
    tvlambda = 0.46236 ,
    tvic50 = 124.59,
    #tvid50 = 1, #fix
    #tvgsklag = 1, #fix
    Ω = Diagonal([
        # 0.13873, 0.47798, 0.51177, 0.97883,
        # 0.2798, 0.128, 0.1722, 0.25, 0.1544
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1
        ]),
    σ_add = 0.318,
    σ_proppk = 0.190,
    σ_addpk = 1.3)

# ------------------------------------
# 3) Build a population and dosing design
# ------------------------------------

# obstimes
dense_hr8 = vcat(0:0.04:0.3, 0.5, 0.6)   
dense_day = vcat(1, 2, 3, 7, )   
day42 = vcat(7:7:43)

obstimes = unique(sort(vcat(dense_hr8,dense_day,day42)))


ev4 = DosageRegimen(240; time = 0, cmt = 1, ii =1, addl= 13, evid=1, rate = -2)
pop_ev4 = map(i -> Subject(id = i, events = ev4), 1:100)


# ------------------------------------
# 4) Simulate
# ------------------------------------
Random.seed!(1234)
pop_sim_ev4 = simobs(hivPKPD_model, pop_ev4, params, obstimes = obstimes)
df_pop_ev4 = DataFrame(pop_sim_ev4)
df_pop_ev4[!, :Dose] .= "240 mg every day"


pkpddata_sim = vcat(df_pop_ev4)

# ------------------------------------
# 5) write CSVs
# ------------------------------------

cd(@__DIR__)

# vscodedisplay(pkpddata_hetero_sim)

CSV.write("pkpd_hiv_sim_short.csv", pkpddata_sim)
