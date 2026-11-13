"""Create a SYNTHETIC demo dataset with the same columns as cleaned_credit_portfolio.csv.
Only for trying the app without the real data. The numbers are random and meaningless.
Usage: python make_demo_data.py [n_rows]
"""
import sys
import numpy as np
import pandas as pd

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
rng = np.random.default_rng(0)
emp = ["< 1 year"] + [f"{i} years" if i > 1 else "1 year" for i in range(1, 10)] + ["10+ years", "NI"]
purposes = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
            "renewable_energy", "educational"]
states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "IA", "MS"]
df = pd.DataFrame({
    "id": np.arange(n) + 1000,
    "issue_d": "Dec-2015",
    "revenue": rng.lognormal(11.1, 0.5, n).round(0),
    "dti_n": rng.gamma(4, 4.5, n).round(2),
    "loan_amnt": rng.integers(1, 41, n) * 1000,
    "fico_n": rng.choice(np.arange(662, 850, 5.0), n),
    "experience_c": 1,
    "emp_length": rng.choice(emp, n),
    "purpose": rng.choice(purposes, n, p=np.r_[[.55, .2, .06, .06, .03, .02, .02, .02, .01, .01, .005, .005, .0025, .0075]] / 1.0),
    "home_ownership_n": rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], n, p=[.5, .4, .099, .001]),
    "addr_state": rng.choice(states, n),
})
p = 0.08 + 0.004 * df["dti_n"].clip(upper=40) - 0.0004 * (df["fico_n"] - 660)
df["Default"] = (rng.random(n) < p.clip(0.02, 0.6)).astype(int)
df["revenue_log"] = np.log1p(df["revenue"])
df["dti_capped"] = df["dti_n"].clip(upper=65)
df["dti_extreme"] = (df["dti_n"] >= 100).astype("int8")
df.to_csv("demo_credit_portfolio.csv", index=False)
print("Wrote demo_credit_portfolio.csv", df.shape)
