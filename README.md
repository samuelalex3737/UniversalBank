# Bank Personal Loan Acceptance Dashboard (Streamlit)

## Overview
This project delivers a professional, interactive **Streamlit analytics dashboard** to explore and explain customer behavior related to **Personal Loan acceptance** in a banking dataset.

The dashboard focuses on:
- Customer demographics (Age, Education, Family)
- Financial capacity and behavior (Income, CCAvg, Mortgage)
- Product ownership impact (Securities Account, CD Account, Credit Card)
- Relationships that differentiate customers who **accepted** vs **did not accept** a Personal Loan

---

## Dataset Description
The dataset is a banking customer dataset commonly used for loan acceptance analysis. Typical columns include:

- `Age` – customer age
- `Income` – annual income
- `CCAvg` – average credit card spending
- `Family` – family size
- `Education` – education level (categorical numeric)
- `Mortgage` – mortgage value
- `Personal Loan` – target variable (0/1)
- `Securities Account` – has securities account (0/1)
- `CD Account` – has CD account (0/1)
- `CreditCard` – has credit card (0/1)
- `ZIP Code` – zipcode

> The app supports common column name variations (e.g., `Zipcode`, `Zip Code`, `ZIP Code`).

---

## Dashboard Features & Visualizations (Required)
### Filters (Sidebar)
- Personal Loan status
- Education level (if available)
- Family size (if available)
- Configurable CSV path input (default: `bank.csv`)

### Visualizations
1. **Histograms**
   - Income distribution by Personal Loan status
   - Age distribution by Personal Loan status

2. **Scatter Plot**
   - CCAvg vs Income (color-coded by Personal Loan)

3. **Multivariate Analysis**
   - Zipcode vs Avg Income vs Personal Loan
   - Bubble size = customer count, color = loan acceptance rate

4. **Correlation Heatmap**
   - Correlation among all numeric columns

5. **Family Analysis**
   - Family size vs Income (segmented by Personal Loan)
   - Bubble relationship encoding Mortgage/CCAvg where available

6. **Financial Products Comparison**
   - Securities Account vs Loan acceptance
   - CD Account vs Loan acceptance
   - Credit Card vs Loan acceptance

7. **Box & Whisker Plots**
   - CCAvg distribution by Personal Loan
   - Income distribution by Personal Loan
   - Credit card spending (interpreted as CCAvg) comparison

8. **Education Analysis**
   - Education vs Income segmented by Personal Loan
   - Acceptance rate by Education level

9. **Mortgage Analysis**
   - Mortgage vs Income with Family size encoding
   - Highlight Personal Loan status

---

## Tech Stack
- Python
- Streamlit
- Pandas / NumPy
- Plotly (interactive visualizations)
- Seaborn (imported for compatibility / styling; Plotly is primary)

---

## Run Locally
### 1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate