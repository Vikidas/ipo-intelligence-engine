# 📈 IPO Intelligence Engine

> A machine learning system that predicts whether an upcoming Indian IPO will list at a gain — trained on 559 real IPOs from 2010 to 2025.

**Live App:**(https://vikidas-ipo-intelligence-engine-app-qn89ei.streamlit.app/)← replace with your actual link after deploying

---

## The Problem

Every month, lakhs of retail investors in India apply to IPOs based on WhatsApp forwards and gut feeling. There is no free, data-driven tool that tells you — *"based on historical patterns of 500+ IPOs, here is the probability this IPO lists at a gain, and here is exactly why."*

This project solves that.

---

## Demo

![App Screenshot](screenshot.png)

**How it works in real life:**
1. An IPO closes subscription on Day 3 (around 6pm)
2. You check the final QIB, HNI, Retail numbers on NSE/BSE
3. You enter them into the app along with GMP and issue details
4. The model gives you a probability score + SHAP explanation
5. You make an informed decision

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.834** |
| Accuracy (test set) | **80.4%** |
| Precision | **85.4%** |
| Recall | **89.4%** |
| Naive baseline (always apply) | 69.4% |
| **Improvement over baseline** | **+11%** |

Tested on 112 most recent IPOs (2024–2025) using chronological split — no data leakage.

---

## Features

- **Real dataset** — 559 Indian IPOs scraped from Chittorgarh (2010–2025)
- **Key signals** — QIB, HNI, RII subscription ratios, Nifty 5-day return, issue size
- **Three models compared** — Logistic Regression, Random Forest, XGBoost
- **SHAP explainability** — every prediction explained in plain English
- **Live Nifty data** — market conditions auto-fetched via yfinance
- **GMP input** — grey market premium as an additional signal
- **Deployed app** — anyone can use it via browser, no installation needed

---

## What the App Shows

### Predict Tab
Enter any upcoming IPO's subscription data and get:
- ✅ APPLY or ❌ AVOID recommendation
- Probability of listing gain (0–100%)
- Confidence level (High / Medium / Low)
- SHAP explanation — top reasons for the prediction

### Historical Analysis Tab
- Win rate by year (2010–2025)
- Listing gain distribution
- Recent IPO performance table

### How It Works Tab
- Full methodology explanation
- Feature importance ranking
- How to use for real investing decisions

---

## Key Findings from the Data

- **Overall win rate:** 69.4% of Indian IPOs list at a gain — blindly applying wins 7 out of 10 times
- **Best year:** 2023 had an 85% win rate
- **Worst year:** 2022 had a 61.5% win rate
- **Strongest signal:** Total subscription (0.72 correlation with listing gain)
- **QIB matters most:** Institutional conviction is more predictive than retail enthusiasm
- **Issue size is negative:** Larger IPOs tend to list with smaller gains

---

## Tech Stack

| Component | Technology |
|---|---|
| Data collection | Selenium, BeautifulSoup, requests |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Market data | yfinance |
| Dashboard | Streamlit, Plotly, Matplotlib |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
ipo-intelligence-engine/
├── app.py                  # Streamlit dashboard
├── ipo_model_ready.csv     # Cleaned dataset (559 IPOs)
├── best_ipo_model.pkl      # Trained Random Forest model
├── model_features.pkl      # Feature list
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/ipo-intelligence-engine
cd ipo-intelligence-engine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## How to Use for a Real IPO

**Step 1** — Wait until the IPO subscription closes (Day 3, after 6pm)

**Step 2** — Go to `chittorgarh.com` or `nseindia.com` and find the final subscription data:
- QIB subscription (x times)
- HNI/NII subscription (x times)
- Retail/RII subscription (x times)
- Total subscription (x times)

**Step 3** — Note the offer price, issue size, and GMP (grey market premium) from Chittorgarh

**Step 4** — Enter all values in the app → click Predict

**Step 5** — Use the prediction as one signal alongside your own research

> ⚠️ Always use **Day 3 final numbers** — not Day 1 or Day 2. QIBs and HNIs submit most bids on the last day, so early numbers are misleading.

---

## Model Validation

The model is validated using **chronological split** — trained on older IPOs and tested on recent ones. This mirrors real-world usage and prevents data leakage.

- Training set: 447 IPOs (2010 → Feb 2024)
- Test set: 112 IPOs (Feb 2024 → Aug 2025)

This is the correct way to validate time-series financial data. Random splitting would leak future information into training and produce artificially inflated accuracy.

---

## Limitations

- GMP data is not available historically so it is not a training feature — only an input signal
- Model does not account for company fundamentals (P/E ratio, promoter holding, revenue growth)
- Market crashes on listing day cannot be fully predicted
- SME IPOs behave differently from mainboard IPOs — this model is trained on mainboard data
- Past patterns do not guarantee future results

---

## What I Would Add in v2

- NLP on DRHP prospectus — sentiment analysis of risk factors section
- P/E ratio vs sector average as a valuation feature
- Promoter holding % post-IPO
- Separate model for SME IPOs
- Real-time GMP scraping — auto-fetch instead of manual entry
- Email/WhatsApp alert when a new IPO crosses prediction threshold

---

## Disclaimer

This tool is for **educational and personal research purposes only**. It is not SEBI-registered investment advice. The predictions are based on historical patterns and do not guarantee future results. Always do your own research before investing. Never invest more than you can afford to lose.

---

## About

Built by **Vignesh** — 3rd year undergraduate student passionate about data science and Indian financial markets.

This project was built to solve a real problem I noticed — retail investors in India have no data-driven tool to evaluate IPOs. Everything from data collection to model training to deployment was built from scratch.

**Connect with me:**
- LinkedIn: [your linkedin]
- GitHub: [your github]

---

*If you found this useful, please ⭐ the repo — it helps others find it too.*
