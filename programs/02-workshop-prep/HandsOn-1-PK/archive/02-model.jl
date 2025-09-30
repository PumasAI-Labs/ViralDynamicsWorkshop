include("01-population.jl")

# Model definition
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
    
    Ω ∈ PDiagDomain(4)
    
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
    Q  = tvq * 24  #Q_0*24   L/day 
    Vp = tvvp #V3_0  
    
  end
    
    @dosecontrol begin
        duration = (;
        Depot = (tvd/24) * exp(η[4])
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
