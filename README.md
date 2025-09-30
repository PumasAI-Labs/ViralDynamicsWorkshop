# DeepPumas for viral dynamics workshop

## Schedule

| Time | Session | Duration |
|------|---------|----------|
| 10:00 - 10:15 | Introduction | 15 min |
| 10:15 - 11:30 | NLME modeling in Pumas (hands-on) | 1h 15min |
| 11:30 - 12:00 | DeepNLME - Part 1 (lecture) | 30 min |
| 12:00 - 12:30 | DeepNLME - Part 1 (hands-on) | 30 min |
| 12:30 - 13:30 | 🍽️ Lunch Break | 1h |
| 13:30 - 15:30 | DeepNLME with complex covariates (lecture & hands-on) | 2h |
| 15:30 - 15:45 | ☕ Coffee Break | 15 min |
| 15:45 - 16:15 | Epidemiology demo | 30 min |
| 16:15 - 17:00 | Discussions and conclusions | 45 min |

**Total Workshop Time:** 10:00 - 17:00 (7 hours including breaks)  
**Total Session Time:** 5h 45min


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