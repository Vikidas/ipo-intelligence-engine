import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO Intelligence Engine",
    page_icon="📈",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('best_ipo_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv('ipo_model_ready.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['nifty_5d_return'] = df['nifty_5d_return'].fillna(
        df['nifty_5d_return'].median()
    )
    return df

@st.cache_data
def get_nifty_return():
    try:
        nifty = yf.download("^NSEI", period="10d", progress=False)
        closes = nifty['Close'].dropna()
        if len(closes) >= 5:
            ret = ((closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5]) * 100
            return float(ret)
    except:
        pass
    return 0.0

model, FEATURES = load_model()
df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 IPO Intelligence Engine")
st.markdown("*Predict IPO listing gains using ML trained on 559 real Indian IPOs (2010–2025)*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict IPO", "📊 Historical Analysis", "🧠 How It Works"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter IPO Details")
    st.info("Fill in the subscription data available on NSE/BSE on the last day of subscription (Day 3).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Subscription Data**")
        qib   = st.number_input("QIB Subscription (x)", min_value=0.0, value=50.0, step=0.1,
                                 help="How many times QIB portion was subscribed")
        hni   = st.number_input("HNI Subscription (x)", min_value=0.0, value=30.0, step=0.1,
                                 help="How many times HNI portion was subscribed")
        rii   = st.number_input("RII/Retail Subscription (x)", min_value=0.0, value=15.0, step=0.1,
                                 help="How many times Retail portion was subscribed")
        total = st.number_input("Total Subscription (x)", min_value=0.0,
                                 value=round((qib+hni+rii)/3, 2), step=0.1,
                                 help="Overall subscription across all categories")

    with col2:
        st.markdown("**IPO Details**")
        offer_price  = st.number_input("Offer Price (₹)", min_value=1.0, value=500.0, step=1.0)
        issue_size   = st.number_input("Issue Size (₹ Crores)", min_value=1.0, value=1000.0, step=10.0)
        listing_month= st.selectbox("Listing Month", list(range(1,13)),
                                     format_func=lambda x: [
                                         'Jan','Feb','Mar','Apr','May','Jun',
                                         'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        listing_year = st.selectbox("Listing Year", [2024, 2025, 2026], index=2)

    with col3:
        st.markdown("**Market Conditions**")
        auto_nifty = get_nifty_return()
        nifty_ret  = st.number_input(
            "Nifty 5-day return % (auto-fetched)",
            value=round(auto_nifty, 2),
            step=0.1,
            help="Nifty 50 return in last 5 days. Auto-fetched but you can override."
        )
        gmp = st.number_input(
            "GMP — Grey Market Premium (₹)",
            min_value=-500.0, value=0.0, step=1.0,
            help="Check on Chittorgarh.com. Optional but improves judgment."
        )
        st.markdown("---")
        st.markdown("**GMP as % of offer price**")
        if offer_price > 0 and gmp != 0:
            gmp_pct = (gmp / offer_price) * 100
            color   = "green" if gmp_pct > 0 else "red"
            st.markdown(f"<h2 style='color:{color}'>{gmp_pct:+.1f}%</h2>",
                        unsafe_allow_html=True)
        else:
            st.markdown("*Enter GMP above*")

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("---")
    predict_btn = st.button("🔮 Predict Listing Gain", type="primary", use_container_width=True)

    if predict_btn:
        # Build feature vector
        eps = 0.01
        qib_rii  = qib  / (rii  + eps)
        hni_rii  = hni  / (rii  + eps)
        inst_dom = qib  / (total + eps)
        log_iss  = np.log1p(issue_size)
        log_qib  = np.log1p(qib)
        log_hni  = np.log1p(hni)
        log_rii  = np.log1p(rii)
        quarter  = (listing_month - 1) // 3 + 1

        feature_values = {
            'QIB':                    qib,
            'HNI':                    hni,
            'RII':                    rii,
            'Total':                  total,
            'Issue_Size(crores)':     issue_size,
            'Offer Price':            offer_price,
            'qib_to_rii_ratio':       qib_rii,
            'hni_to_rii_ratio':       hni_rii,
            'institutional_dominance':inst_dom,
            'log_issue_size':         log_iss,
            'log_qib':                log_qib,
            'log_hni':                log_hni,
            'log_rii':                log_rii,
            'month':                  listing_month,
            'quarter':                quarter,
            'year':                   listing_year,
            'nifty_5d_return':        nifty_ret
        }

        X_input = pd.DataFrame([feature_values])[FEATURES]
        prob    = model.predict_proba(X_input)[0][1]
        pred    = "APPLY ✅" if prob >= 0.5 else "AVOID ❌"

        # GMP adjustment note
        gmp_note = ""
        if gmp > 0:
            gmp_note = f" | GMP suggests +{gmp_pct:.1f}% listing — supports model"
        elif gmp < 0:
            gmp_note = f" | GMP is negative ({gmp_pct:.1f}%) — be cautious"

        # ── Result display ────────────────────────────────────────────────────
        st.markdown("### 📊 Prediction Result")
        r1, r2, r3 = st.columns(3)

        with r1:
            color = "#2ecc71" if prob >= 0.5 else "#e74c3c"
            st.markdown(f"""
                <div style='background:{color}22; border:2px solid {color};
                            border-radius:12px; padding:20px; text-align:center'>
                    <h1 style='color:{color}; margin:0'>{pred}</h1>
                    <p style='margin:0'>Model Recommendation</p>
                </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
                <div style='background:#3498db22; border:2px solid #3498db;
                            border-radius:12px; padding:20px; text-align:center'>
                    <h1 style='color:#3498db; margin:0'>{prob:.1%}</h1>
                    <p style='margin:0'>Probability of Listing Gain</p>
                </div>
            """, unsafe_allow_html=True)

        with r3:
            confidence = "High" if abs(prob-0.5) > 0.3 else "Medium" if abs(prob-0.5) > 0.15 else "Low"
            conf_color = "#2ecc71" if confidence == "High" else "#f39c12" if confidence == "Medium" else "#e74c3c"
            st.markdown(f"""
                <div style='background:{conf_color}22; border:2px solid {conf_color};
                            border-radius:12px; padding:20px; text-align:center'>
                    <h1 style='color:{conf_color}; margin:0'>{confidence}</h1>
                    <p style='margin:0'>Confidence Level</p>
                </div>
            """, unsafe_allow_html=True)

        if gmp_note:
            st.info(f"📌 GMP Signal{gmp_note}")

        # ── SHAP explanation ──────────────────────────────────────────────────
        st.markdown("### 🧠 Why this prediction?")

        try:
            explainer   = shap.TreeExplainer(model)
            sv          = explainer.shap_values(X_input)

            if isinstance(sv, list):
                sv_i = np.array(sv[1][0]).flatten()
            elif np.array(sv).ndim == 3:
                sv_i = np.array(sv)[0, :, 1].flatten()
            else:
                sv_i = np.array(sv[0]).flatten()

            shap_df = pd.DataFrame({
                'Feature':   FEATURES,
                'Impact':    sv_i,
                'Value':     X_input.values[0]
            }).sort_values('Impact', key=abs, ascending=False).head(8)

            for _, row in shap_df.iterrows():
                direction = "🟢 pushes toward GAIN" if row['Impact'] > 0 else "🔴 pushes toward LOSS"
                bar_width = min(int(abs(row['Impact']) * 1000), 100)
                bar_color = "#2ecc71" if row['Impact'] > 0 else "#e74c3c"
                st.markdown(f"""
                    <div style='margin:6px 0; padding:8px 12px; background:var(--background-color);
                                border-left:4px solid {bar_color}; border-radius:4px'>
                        <b>{row['Feature']}</b> = {row['Value']:.2f}
                        &nbsp;&nbsp;→&nbsp;&nbsp;{direction}
                        &nbsp;&nbsp;<span style='color:{bar_color}'>({row['Impact']:+.4f})</span>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("---")
        st.warning("⚠️ This is an ML-based analysis tool for educational purposes. "
                   "Not SEBI-registered investment advice. Always do your own research before investing.")

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Historical IPO Performance (2010–2025)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total IPOs", len(df))
    c2.metric("Overall Win Rate", f"{df['target'].mean()*100:.1f}%")
    c3.metric("Avg Listing Gain (Winners)", f"{df[df['target']==1]['Listing Gain'].mean():.1f}%")
    c4.metric("Avg Listing Loss (Losers)",  f"{df[df['target']==0]['Listing Gain'].mean():.1f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Win Rate by Year**")
        win_rate = df.groupby('year')['target'].mean() * 100
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71' if v >= 70 else '#e74c3c' for v in win_rate.values]
        ax1.bar(win_rate.index, win_rate.values, color=colors, alpha=0.85, edgecolor='white')
        ax1.axhline(70, color='gray', linestyle='--', linewidth=1)
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_xlabel("Year")
        ax1.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.markdown("**Listing Gain Distribution**")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(df['Listing Gain'], bins=40, color='steelblue',
                 edgecolor='white', alpha=0.85)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax2.axvline(df['Listing Gain'].mean(), color='green',
                    linestyle='--', linewidth=1.5,
                    label=f"Mean: {df['Listing Gain'].mean():.1f}%")
        ax2.set_xlabel("Listing Day Gain (%)")
        ax2.set_ylabel("Number of IPOs")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("**Recent IPOs**")
    recent = df.sort_values('Date', ascending=False).head(20)[
        ['IPO_Name', 'Date', 'QIB', 'HNI', 'RII',
         'Offer Price', 'Listing Gain', 'target']
    ].copy()
    recent['Result'] = recent['target'].map({1: '✅ Gain', 0: '❌ Loss'})
    recent['Date']   = recent['Date'].dt.date
    recent = recent.drop('target', axis=1)
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How the IPO Intelligence Engine Works")

    st.markdown("""
    ### The Problem
    Every month, lakhs of retail investors apply to IPOs based on WhatsApp forwards and gut feeling.
    This tool uses machine learning trained on **559 real Indian IPOs** to give a data-driven signal.

    ### The Model
    - **Algorithm**: Random Forest (83.4% ROC-AUC on 2024–2025 test data)
    - **Training data**: 447 IPOs from 2010–2024
    - **Test data**: 112 IPOs from 2024–2025 (chronological split — no data leakage)
    - **Baseline**: Blindly applying to every IPO wins 69.4% of the time
    - **Our model**: 80.4% accuracy on unseen recent IPOs

    ### Key Features (in order of importance)
    | Feature | Why it matters |
    |---|---|
    | Total Subscription | Overall market demand signal |
    | HNI Subscription | Smart money — HNIs do due diligence |
    | QIB Subscription | Institutional conviction — strongest signal |
    | RII/Retail | Retail FOMO indicator |
    | Nifty 5-day return | Market mood on listing week |
    | Issue Size | Larger IPOs have less room to run |

    ### How to use for a real IPO
    1. Wait until **Day 3 of subscription** — final subscription data is published on NSE/BSE
    2. Check **GMP** on Chittorgarh.com
    3. Enter all values in the **Predict IPO** tab
    4. Use the prediction as **one signal** among your own research
    5. Never invest more than you can afford to lose

    ### Disclaimer
    This tool is for educational and personal research purposes only.
    It is not SEBI-registered investment advice.
    Past patterns do not guarantee future results.
    """)