# ParkEase Synthetic Demonstration Dataset

## Scientific Integrity & Transparency Notice

> [!IMPORTANT]
> The dataset in this directory (`parking_synthetic_demo.csv`) is **strictly synthetic demonstration data** created for educational and system-verification purposes.
>
> It exists solely to validate the end-to-end reproducibility of machine learning pipelines, preprocessing transformations (`StandardScaler` and `OneHotEncoder`), experiment tracking (MLflow), and Flask web integration.
>
> **It does not represent observational real-world parking records.** Metrics (such as MAE, RMSE, and $R^2$) evaluated against this synthetic data demonstrate pipeline correctness and software stability, but **cannot and will not establish real-world predictive parking accuracy**.

---

## Dataset Specifications

* **File Name:** `parking_synthetic_demo.csv`
* **Sample Count:** 2,500 records
* **Target Feature:** `Parking Fee (INR)` (continuous, non-negative)
* **Random Seed:** `42` (Fixed deterministic generation)

### Features Schema

| Column Name | Data Type | Permitted Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `Duration (min)` | Float (`float64`) | $\ge 5.0$ minutes | Elapsed parking session duration in minutes |
| `Vehicle Type` | String (`object`) | `['car', '2-wheeler', 'auto', 'suv/van']` | Vehicle classification category |
| `Electric` | String (`object`) | `['electric', 'non-electric']` | Electric vehicle status indicator |
| `Weather` | String (`object`) | `['good', 'average', 'bad']` | Environmental condition tier (Trichy climate model) |
| `Congestion Level` | Float (`float64`) | $[0.0, 1.0]$ | Ratio of occupied parking slots ($0.0 = 0\%$, $1.0 = 100\%$) |
| `Parking Fee (INR)` | Float (`float64`) | $\ge 5.00$ INR | Calculated target parking charge (simulated ground truth) |

---

## Simulated Pricing Domain Logic

The synthetic ground truth is generated via a deterministic economic pricing formulation:

1. **Base Hourly Rates ($R_{\text{vehicle}}$):**
   * `2-wheeler`: ₹15.00 / hour
   * `auto`: ₹25.00 / hour
   * `car`: ₹40.00 / hour
   * `suv/van`: ₹60.00 / hour

2. **Duration Treatment & Long-Stay Discount:**
   * Minimum billable time: 15 minutes ($0.25$ hours).
   * For hours $H \le 4.0$: standard hourly rate.
   * For hours $H > 4.0$: excess hours receive a $15\%$ long-stay discount ($0.85$ multiplier).
   $$\text{Base Fee} = R_{\text{vehicle}} \times (\min(H, 4.0) + 0.85 \times \max(0.0, H - 4.0))$$

3. **EV Incentive Multiplier:**
   * `electric`: $0.85$ (15% green energy incentive discount)
   * `non-electric`: $1.00$

4. **Weather Condition Multiplier:**
   * `bad`: $1.15$ (+15% surge for covered/structured parking demand during storms or extreme heat)
   * `average`: $1.00$
   * `good`: $0.95$ (-5% standard off-peak demand)

5. **Congestion Surcharge:**
   $$\text{Congestion Multiplier} = 1.0 + (0.30 \times \text{Congestion Level})$$
   *(Linearly scales fee up to $+30\%$ at full capacity).*

6. **Noise & Boundary:**
   * Zero-mean Gaussian perturbation $\epsilon \sim \mathcal{N}(0, 1.50)$ added to simulate transaction variations.
   * Hard lower bound enforced: $\text{Parking Fee (INR)} = \max(5.00, \text{Fee}_{\text{raw}})$.

---

## How to Regenerate Deterministically

From the project root:
```bash
python3 ml/generate_synthetic_data.py --output ml/data/parking_synthetic_demo.csv --n-records 2500 --seed 42
```
Generating with the same seed (`42`) will produce an identical CSV file with the exact same SHA-256 hash.
