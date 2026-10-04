import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Produces reproducible results
#10,000 simulated trials
np.random.seed(20) 
N_SIMS = 10000      


TEMP_ANOMALY = 1.50   # +1.5 global warming scenario
MOISTURE_FACTOR = 0.07 # 7% Clausius-Clapeyron moisture increase
RUNOFF_AMP = 1.50      # Soil saturation runoff multiplier

# Global Frequency Shift Multiplier Formula
CLIM_FREQ_MULT = 1 + (TEMP_ANOMALY * MOISTURE_FACTOR * RUNOFF_AMP)

#Data set used for the simulation
states_data = [
    ("Alabama", 1.30, 52000, 1.45), ("Alaska", 0.40, 38000, 1.35),
    ("Arizona", 0.45, 34000, 1.30), ("Arkansas", 0.95, 41000, 1.40),
    ("California", 0.65, 68000, 1.55), ("Colorado", 0.50, 42000, 1.35),
    ("Connecticut", 0.70, 49000, 1.45), ("Delaware", 0.75, 46000, 1.40),
    ("Florida", 2.20, 88000, 1.60), ("Georgia", 1.15, 48000, 1.45),
    ("Hawaii", 0.55, 51000, 1.35), ("Idaho", 0.35, 31000, 1.30),
    ("Illinois", 0.90, 44000, 1.40), ("Indiana", 0.85, 39000, 1.38),
    ("Iowa", 0.85, 43000, 1.42), ("Kansas", 0.65, 36000, 1.35),
    ("Kentucky", 1.10, 47000, 1.50), ("Louisiana", 2.10, 82000, 1.58),
    ("Maine", 0.45, 35000, 1.35), ("Maryland", 0.80, 48000, 1.42),
    ("Massachusetts", 0.75, 52000, 1.45), ("Michigan", 0.60, 37000, 1.35),
    ("Minnesota", 0.60, 39000, 1.35), ("Mississippi", 1.40, 54000, 1.48),
    ("Missouri", 1.05, 46000, 1.45), ("Montana", 0.40, 33000, 1.30),
    ("Nebraska", 0.55, 38000, 1.35), ("Nevada", 0.30, 32000, 1.25),
    ("New Hampshire", 0.45, 36000, 1.35), ("New Jersey", 1.20, 65000, 1.52),
    ("New Mexico", 0.40, 33000, 1.30), ("New York", 1.25, 69000, 1.55),
    ("North Carolina", 1.50, 62000, 1.52), ("North Dakota", 0.50, 41000, 1.35),
    ("Ohio", 0.85, 40000, 1.38), ("Oklahoma", 0.80, 42000, 1.40),
    ("Oregon", 0.60, 44000, 1.45), ("Pennsylvania", 1.15, 53000, 1.48),
    ("Rhode Island", 0.55, 47000, 1.40), ("South Carolina", 1.35, 58000, 1.50),
    ("South Dakota", 0.45, 35000, 1.32), ("Tennessee", 1.05, 46000, 1.45),
    ("Texas", 1.70, 72000, 1.55), ("Utah", 0.30, 30000, 1.25),
    ("Vermont", 0.70, 49000, 1.55), ("Virginia", 1.00, 50000, 1.45),
    ("Washington", 0.70, 46000, 1.50), ("West Virginia", 1.05, 45000, 1.48),
    ("Wisconsin", 0.60, 38000, 1.35), ("Wyoming", 0.30, 29000, 1.25)
]

# Simulation arrays to hold the aggregate annual losses for the entire portfolio
portfolio_base_annual_losses = np.zeros(N_SIMS)
portfolio_clim_annual_losses = np.zeros(N_SIMS)

#Heavy tail losses
SIGMA = 0.65  

state_output_records = []


for state, b_freq, b_sev, s_mult in states_data:
    c_freq = b_freq * CLIM_FREQ_MULT
    c_sev = b_sev * s_mult
    
    # Convert mean claim cost into lognormal mu parameter
    mu_base = np.log(b_sev) - 0.5 * (SIGMA**2)
    mu_clim = np.log(c_sev) - 0.5 * (SIGMA**2)
    
    # 1. Generating number of storms per year for 10,000 years
    n_storms_base = np.random.poisson(b_freq, N_SIMS)
    n_storms_clim = np.random.poisson(c_freq, N_SIMS)
    
    #Storage for state-level annual losses
    state_base_losses = np.zeros(N_SIMS)
    state_clim_losses = np.zeros(N_SIMS)
    
    # 2.Generates claims for each storm and sums them to get annual losses for each simulation
    for sim_idx in range(N_SIMS):
        if n_storms_base[sim_idx] > 0:
            claims = np.random.lognormal(mu_base, SIGMA, n_storms_base[sim_idx])
            state_base_losses[sim_idx] = np.sum(claims)
            
        if n_storms_clim[sim_idx] > 0:
            claims = np.random.lognormal(mu_clim, SIGMA, n_storms_clim[sim_idx])
            state_clim_losses[sim_idx] = np.sum(claims)
            
    # Adding states to the portfolio-level annual losses
    portfolio_base_annual_losses += state_base_losses
    portfolio_clim_annual_losses += state_clim_losses
    
    # Calculating 99th percentile level losses
    pml_99_clim = np.percentile(state_clim_losses, 99)
    tvar_99_clim = np.mean(state_clim_losses[state_clim_losses >= pml_99_clim])
    
    state_output_records.append({
        'State': state,
        'Sim_Baseline_Pure_Prem ($)': round(np.mean(state_base_losses), 2),
        'Sim_Climate_Pure_Prem ($)': round(np.mean(state_clim_losses), 2),
        'Sim_1_in_100_PML ($k)': round(pml_99_clim / 1000, 1),
        'Sim_1_in_100_TVaR ($k)': round(tvar_99_clim / 1000, 1)
    })

#National Metrics
base_mean = np.mean(portfolio_base_annual_losses)
clim_mean = np.mean(portfolio_clim_annual_losses)
base_pml = np.percentile(portfolio_base_annual_losses, 99)
clim_pml = np.percentile(portfolio_clim_annual_losses, 99)
clim_tvar = np.mean(portfolio_clim_annual_losses[portfolio_clim_annual_losses >= clim_pml])

print("\n" + "="*55)
print("     NATIONWIDE MONTE CARLO RESULTS (10,000 YEARS)")
print("="*55)
print(f"Historical Baseline Pure Premium:   ${base_mean:,.0f}")
print(f"Climate-Adjusted Pure Premium:      ${clim_mean:,.0f}")
print(f"Portfolio Rate Inadequacy:          +{((clim_mean - base_mean) / base_mean) * 100:.1f}%")
print(f"Historical 1-in-100 Year PML (VaR): ${base_pml:,.0f}")
print(f"Climate 1-in-100 Year PML (VaR):    ${clim_pml:,.0f}")
print(f"Climate 1-in-100 Year TVaR:         ${clim_tvar:,.0f}")
print("="*55)

# Generate EP Curve Chart
sorted_base = np.sort(portfolio_base_annual_losses)[::-1]
sorted_clim = np.sort(portfolio_clim_annual_losses)[::-1]
probabilities = np.arange(1, N_SIMS + 1) / N_SIMS
return_periods = 1 / probabilities

plt.figure(figsize=(10, 6))
plt.plot(return_periods, sorted_base / 1e6, label='Historical Baseline Portfolio', color='#1F497D', linewidth=2.5)
plt.plot(return_periods, sorted_clim / 1e6, label='Climate-Adjusted Portfolio (+1.5°C)', color='#C00000', linewidth=2.5)
plt.axvline(100, color='gray', linestyle='--', label='1-in-100 Year Benchmark')

plt.xscale('log')
plt.xlim(1, 200)
plt.xlabel('Return Period in Years (Log Scale)', fontsize=12, fontweight='bold')
plt.ylabel('Aggregate Portfolio Loss ($ Millions)', fontsize=12, fontweight='bold')
plt.title('U.S. Property Flood Portfolio: Exceedance Probability (EP) Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('EP_Curve_Shift.png', dpi=200)
plt.show()

# Export state results to CSV
df_summary = pd.DataFrame(state_output_records)
df_summary.to_csv("Monte_Carlo_50_States.csv", index=False)
print("\nExported 'Monte_Carlo_50_States.csv' successfully!")