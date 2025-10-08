![Static Badge](https://img.shields.io/badge/Take%20me%20to%20the%20website?link=https%3A%2F%2Fpumasai-labs.github.io%2FViralDynamicsWorkshop%2F)

# DeepPumas for viral dynamics workshop

## Schedule

| Time | Session | Duration |
|------|---------|----------|
| 09:00 - 09:20 | **Welcome and Introduction** | 20 min |
| 09:20 - 10:30 | **NLME modeling in Pumas** (hands-on) | 1h 10min |
| 10:30 - 10:45 | ☕ **Coffee Break** | 15 min |
| 10:45 - 11:15 | **DeepNLME** (lecture) | 30 min |
|  | _Neural networks, SciML, UDEs and NeuralODEs_ | |
| 11:15 - 12:30 | **DeepNLME** (hands-on) | 1h 15min |
| 12:30 - 13:30 | 🍽️ **Lunch Break** | 1h |
| 13:30 - 14:15 | **Random effects, fitting NLME, and Generative AI** | 45 min |
| 14:15 - 15:30 | **DeepNLME with Complex Covariates** (lecture & hands-on) | 1h 15min |
| 15:30 - 15:45 | ☕ **Coffee Break** | 15 min |
| 15:45 - 16:15 | **Epidemiology Demo** | 30 min |
| 16:15 - 17:00 | **Discussions and Conclusions** | 45 min |

**Total Workshop Time:** 09:00 - 17:00 (8 hours including breaks)  
**Total Session Time:** 6h 30min


- Introduction
  - greet
  - table-of-contnents
- NLME modeling in Pumas.
  - JuliaHub - Pumas enterprise
  - Dataset -> wrangling -> read_pumas -> handle the population object
  - sequential PKPD modelling (simple viral dynamics model): 1-cmt PK model -> ebes -> sequential model
  - Fitting (different LL approx)
  - Plotting
- DeepNLME - part 1
  - lecture (30min)
    - NLME modelling
    - SciML and UDEs
      - Neural networks
      - SciML vs UDEs vs Neural ODEs
      - Encoded knowledge
      - States matter! Limitations of the ODE itself.
    - DeepNLME
      - UDEs for longitudinal data (?)
      - Individualizability through covariates and random effects
  - Hands on: 
    - Simple DeepNLME viral dynamics model
- DeepNLME - part 2
  - Lecture (45 min)
      - Marginal likelihood
      - NLME and Generative AI
      - Between subject variability and covariates
  - Hands on:
    - Complex viral dynamics
    - Complex covariates
- Short demos
  - Epidemiology
  - Bridging information (causality or correlation)
- Discussions and conclusion    