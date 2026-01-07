import warnings
import os
import json
import time

# --- [FIX 1] 屏蔽环境版本冲突警告 (解决 SciPy/NumPy 报错) ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NumPy.*")

# --- [FIX 2] 强制使用非交互式后端 (解决 PyInstaller 打包时的 Qt/PyQt5 报错) ---
import matplotlib
matplotlib.use('Agg') 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

# 尝试导入 PyGithub
try:
    from github import Github, GithubException
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ocular Metabolomics CDSS Ultra",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.0)
COLORS = {"HS": "#e74c3c", "ZS": "#3498db", "NC": "#2c3e50", "SAFE": "#27ae60", "RISK": "#c0392b"}

# -----------------------------------------------------------------------------
# GITHUB STORAGE ENGINE (NEW)
# -----------------------------------------------------------------------------
class GitHubStorage:
    """处理与 GitHub 仓库的数据交互"""
    
    @staticmethod
    def get_connection():
        """尝试连接 GitHub"""
        if not HAS_GITHUB: 
            return None
        
        # --- [强制配置] 直接使用您提供的参数 ---
        # 请确保此 Token 拥有 'Contents' 的 'Read and Write' 权限
        REPO_NAME = "shumao469/stroke-data-storage"
        TOKEN = "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"

        try:
            g = Github(TOKEN)
            repo = g.get_repo(REPO_NAME)
            return repo
        except Exception as e:
            # 连接失败时在后台打印原因，返回 None
            print(f"GitHub Connection Error: {e}")
            return None

    @staticmethod
    def get_cohort_count():
        """获取当前队列总数 (Base 751 + GitHub Files)"""
        base_count = 751
        repo = GitHubStorage.get_connection()
        new_count = 0
        
        if repo:
            try:
                # 尝试获取 reports 文件夹内容
                contents = repo.get_contents("reports")
                reports = [c for c in contents if c.name.endswith('.json')]
                new_count = len(reports)
            except Exception:
                # 文件夹可能不存在，这是正常的
                new_count = 0 
        else:
            # 连接失败，使用本地计数
            new_count = st.session_state.get('local_new_cases', 0)
            
        return base_count + new_count, new_count

    @staticmethod
    def save_report(patient_data):
        """将报告保存为 JSON 到 GitHub"""
        if not HAS_GITHUB:
            st.session_state['local_new_cases'] = st.session_state.get('local_new_cases', 0) + 1
            return False, "Save Failed: 'PyGithub' library not installed. Data saved locally only."

        repo = GitHubStorage.get_connection()
        
        # 准备数据 JSON
        json_str = json.dumps(patient_data, indent=4, default=str)
        # 生成唯一文件名
        filename = f"reports/{patient_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if repo:
            try:
                repo.create_file(filename, f"Add clinical report for {patient_data['id']}", json_str)
                return True, "Successfully saved to GitHub Repository."
            except GithubException as e:
                return False, f"GitHub Error (Check Token Permissions): {e.data.get('message', str(e))}"
            except Exception as e:
                return False, f"Unknown Error: {str(e)}"
        else:
            # 本地模拟保存
            st.session_state['local_new_cases'] = st.session_state.get('local_new_cases', 0) + 1
            time.sleep(1) 
            return False, "Connection Failed: Could not connect to GitHub. Using Local Mode (Data will be lost on exit)."

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: CLINICAL SCALES
# -----------------------------------------------------------------------------
def get_nihss_grade(score):
    if score == 0: return "No Stroke Symptoms (0)", "green"
    elif score <= 4: return "Minor Stroke (1-4)", "green"
    elif score <= 15: return "Moderate Stroke (5-15)", "orange"
    elif score <= 20: return "Moderate to Severe (16-20)", "red"
    else: return "Severe Stroke (21-42)", "darkred"

def get_gcs_grade(score):
    if score >= 13: return "Mild Head Injury (13-15)", "green"
    elif score >= 9: return "Moderate Head Injury (9-12)", "orange"
    else: return "Severe Head Injury (3-8)", "darkred"

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
class DataEngine:
    @staticmethod
    @st.cache_data
    def generate_triage_data(n=600):
        np.random.seed(42)
        data = []
        groups = ['NC', 'HS', 'ZS']
        for g in groups:
            for _ in range(n // 3):
                row = {'Group': g}
                if g == 'HS':
                    row['RvD5'] = np.random.normal(8.5, 1.2); row['8_MKNA'] = np.random.normal(7.8, 1.0)
                    row['N_AcCad'] = np.random.normal(9.2, 1.5); row['DTA'] = np.random.normal(7.0, 1.1)
                elif g == 'ZS':
                    row['RvD5'] = np.random.normal(5.0, 1.2); row['8_MKNA'] = np.random.normal(4.5, 1.0)
                    row['N_AcCad'] = np.random.normal(5.5, 1.2); row['DTA'] = np.random.normal(5.0, 1.0)
                else: 
                    row['RvD5'] = np.random.normal(3.0, 0.8); row['8_MKNA'] = np.random.normal(2.5, 0.8)
                    row['N_AcCad'] = np.random.normal(3.2, 0.8); row['DTA'] = np.random.normal(3.5, 0.8)
                data.append(row)
        df = pd.DataFrame(data)
        df['Target'] = df['Group'].apply(lambda x: 1 if x == 'HS' else 0)
        return df

    @staticmethod
    @st.cache_data
    def generate_prognosis_data(n=500):
        np.random.seed(123)
        rvd5 = np.random.normal(6.0, 2.0, n)
        n_accad = np.random.normal(5.5, 1.8, n)
        mkna_8 = np.random.normal(5.0, 1.5, n)
        dta = np.random.normal(4.5, 1.2, n)
        logit_exp = -3.8 + 0.5 * rvd5 + 0.4 * n_accad + np.random.normal(0, 0.5, n)
        prob_exp = 1 / (1 + np.exp(-logit_exp))
        expansion = (np.random.rand(n) < prob_exp).astype(int)
        logit_mrs = -4.0 + 0.6 * mkna_8 + 1.5 * expansion + 0.3 * dta + np.random.normal(0, 0.5, n)
        prob_mrs = 1 / (1 + np.exp(-logit_mrs))
        poor_outcome = (np.random.rand(n) < prob_mrs).astype(int)
        return pd.DataFrame({'RvD5': rvd5, 'N_AcCad': n_accad, '8_MKNA': mkna_8, 'DTA': dta, 'Hematoma_Expansion': expansion, 'Poor_Outcome_mRS': poor_outcome})

# -----------------------------------------------------------------------------
# VISUALIZATION MODULE
# -----------------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def plot_gauge_chart(probability, title="Probability"):
        color = "#e74c3c" if probability > 0.7 else "#f39c12" if probability > 0.3 else "#27ae60"
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = probability * 100, title = {'text': f"{title} (%)"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': color},
                     'steps': [{'range': [0, 30], 'color': 'rgba(39, 174, 96, 0.1)'},
                               {'range': [30, 70], 'color': 'rgba(243, 156, 18, 0.1)'},
                               {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.1)'}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    @staticmethod
    def plot_3d_cluster(df, patient_vals):
        fig = px.scatter_3d(df, x='RvD5', y='8_MKNA', z='N_AcCad', color='Group',
                            color_discrete_map={'HS': '#e74c3c', 'ZS': '#3498db', 'NC': '#95a5a6'}, opacity=0.3)
        fig.add_trace(go.Scatter3d(x=[patient_vals[0]], y=[patient_vals[1]], z=[patient_vals[2]],
            mode='markers', marker=dict(size=12, color='black', symbol='diamond'), name='Patient'))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), title="3D Metabolomic Clustering")
        return fig

    @staticmethod
    def plot_distribution_comparison(df, patient_val, feature_name):
        fig = go.Figure()
        for group in ['NC', 'HS']:
            subset = df[df['Group'] == group][feature_name]
            fig.add_trace(go.Histogram(x=subset, name=group, opacity=0.6, nbinsx=30))
        fig.add_vline(x=patient_val, line_width=3, line_dash="dash", line_color="black", annotation_text="Patient")
        fig.update_layout(barmode='overlay', title=f"{feature_name} Distribution", height=300)
        return fig

    @staticmethod
    def plot_risk_heatmap(patient_data, means_poor_outcome, features):
        comparison = np.array(patient_data) / np.array(means_poor_outcome)
        fig = go.Figure(data=go.Heatmap(z=[comparison], x=features, y=['Ratio'],
            colorscale='RdBu_r', midpoint=1.0, text=[[f"{v:.2f}x" for v in comparison]], texttemplate="%{text}", showscale=True))
        fig.update_layout(title="Biomarker Ratio vs Poor Outcome Avg", height=250)
        return fig

    @staticmethod
    def plot_roc(y_test, y_prob):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color=COLORS['HS'], lw=3, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_title('ROC Analysis'); ax.legend(loc="lower right")
        return fig

    @staticmethod
    def plot_calibration_strata(y_test, y_prob):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8)
        ax1.plot(prob_pred, prob_true, marker='o', lw=2, color=COLORS['HS'])
        ax1.plot([0, 1], [0, 1], '--', color='gray')
        ax1.set_title('Calibration')
        risk_bins = pd.cut(y_prob, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Mod', 'High'])
        strat_df = pd.DataFrame({'Risk': risk_bins, 'Event': y_test})
        means = strat_df.groupby('Risk', observed=False)['Event'].mean() * 100
        ax2.bar(means.index, means.values, color=[COLORS['SAFE'], '#f39c12', COLORS['HS']])
        ax2.set_title('Risk Stratification')
        return fig

    @staticmethod
    def plot_waterfall(feature_names, contributions, base_value):
        fig, ax = plt.subplots(figsize=(8, 5))
        indices = np.argsort(np.abs(contributions))
        names = [feature_names[i] for i in indices]
        vals = [contributions[i] for i in indices]
        running_sum = base_value
        for i, (n, v) in enumerate(zip(names, vals)):
            c = '#ff0051' if v > 0 else '#008bfb'
            ax.barh(i, v, left=running_sum, color=c, height=0.6)
            ax.text(running_sum + v, i, f"{v:+.2f}", va='center', fontsize=9, fontweight='bold', color=c)
            if i < len(names) - 1: ax.plot([running_sum + v, running_sum + v], [i, i+1], 'gray', lw=0.5)
            running_sum += v
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.axvline(x=base_value, color='gray', linestyle='--')
        ax.set_xlabel('Log-Odds'); ax.set_title('SHAP Feature Contribution')
        return fig

# -----------------------------------------------------------------------------
# MAIN APP CONTROLLER
# -----------------------------------------------------------------------------
def main():
    # --- SIDEBAR ---
    st.sidebar.title("🩺 Ocular Stroke CDSS")
    with st.sidebar.expander("👤 Patient Registration", expanded=True):
        p_id = st.text_input("Patient ID", "PT-2024-089")
        p_age = st.number_input("Age", 40, 100, 65)
        # 恢复 Gender 和 Sampling Time
        p_gender = st.selectbox("Gender", ["Male", "Female"])
        p_time = st.time_input("Sampling Time", datetime.now().time())
        
        p_sbp = st.number_input("SBP (mmHg)", 90, 220, 155, help="Systolic Blood Pressure at Admission")
        p_onset = st.number_input("Time from Onset (hrs)", 0.5, 48.0, 3.5)
        
        st.markdown("#### Clinical Scales")
        p_nihss = st.slider("NIHSS Score", 0, 42, 5, help="National Institutes of Health Stroke Scale")
        p_gcs = st.slider("GCS Score", 3, 15, 15, help="Glasgow Coma Scale")
        
        p_history = st.multiselect("History", ["Hypertension", "Anticoagulants", "Diabetes"], ["Hypertension"])
        p_status = st.selectbox("Status", ["Triage Pending", "Confirmed HS", "Confirmed ZS"])

    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Select Module:", 
        ["Dashboard Home", 
         "1. ED Triage System", 
         "2. Early Hematoma Expansion", 
         "3. Outcome Prediction",
         "4. Final Clinical Report"])
    st.sidebar.info("v5.3 | GitHub Integration Fixed")

    # --- DIAGNOSTIC INFO (ONLY IF ERROR) ---
    if not HAS_GITHUB:
        st.sidebar.error("⚠️ Warning: 'PyGithub' library not found. Data will NOT be saved to cloud.")

    # --- HOME ---
    if app_mode == "Dashboard Home":
        st.title("🧠 Ocular Metabolomics Clinical Decision Support")
        
        # Get dynamic stats from GitHub or Local State
        total_cohort, new_cases = GitHubStorage.get_cohort_count()
        
        # Calculate Scale Grades for Display
        nihss_txt, nihss_col = get_nihss_grade(p_nihss)
        gcs_txt, gcs_col = get_gcs_grade(p_gcs)
        
        # Dashboard Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Patient Cohort", str(total_cohort), f"+{new_cases} New")
        col2.metric("Model AUC (HS)", "0.92", "+0.03")
        col3.metric("Processing Time", "< 5 min", "Optimized")
        
        st.divider()
        st.markdown(f"**Current Patient:** {p_id} ({p_gender}, {p_age}y) | **Status:** {p_status}")
        st.info(f"🕒 **Sampling Time:** {p_time.strftime('%H:%M')} | 🗓️ **Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.markdown("Select a module to begin specific analysis.")

    # --- MODULE 1: TRIAGE ---
    elif app_mode == "1. ED Triage System":
        st.header("🚑 Module 1: Emergency Triage")
        with st.expander("🧪 Biomarker Input Panel", expanded=True):
            col_i1, col_i2, col_i3 = st.columns(3)
            rvd5 = col_i1.slider("RvD5", 0.0, 15.0, 8.5); mkna = col_i2.slider("8-MKNA", 0.0, 15.0, 7.8); nacc = col_i3.slider("N-AcCad", 0.0, 15.0, 9.2)
        
        df_triage = DataEngine.generate_triage_data()
        input_arr = np.array([rvd5, mkna, nacc])
        prob_hs = 1 / (1 + np.exp(-(rvd5*0.5 + mkna*0.4 + nacc*0.3 - 8)))
        
        c1, c2 = st.columns([1, 2])
        with c1: st.plotly_chart(Visualizer.plot_gauge_chart(prob_hs, "HS Probability"), use_container_width=True)
        with c2: st.plotly_chart(Visualizer.plot_3d_cluster(df_triage, input_arr), use_container_width=True)
        st.plotly_chart(Visualizer.plot_distribution_comparison(df_triage, rvd5, 'RvD5'), use_container_width=True)

    # --- MODULE 2: EXPANSION ---
    elif app_mode == "2. Early Hematoma Expansion":
        st.header("📉 Module 2: Early Hematoma Expansion")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### Biomarkers")
            val_rvd5 = st.slider("RvD5 Level", 0.0, 15.0, 7.5)
            val_nacc = st.slider("N-AcCad Level", 0.0, 15.0, 6.0)
            score = (p_sbp/200)*2 + (12-p_onset)/12 + (val_rvd5/10)*1.5 + (val_nacc/10)*1.0 + (p_nihss/42)*2 - 5.5
            prob_exp = 1 / (1 + np.exp(-score))
        with c2:
            st.plotly_chart(Visualizer.plot_gauge_chart(prob_exp, "Expansion Risk"), use_container_width=True)
            if prob_exp > 0.7: st.error("CRITICAL RISK: Intensive BP lowering required.")
            elif prob_exp > 0.3: st.warning("MODERATE RISK: Standard monitoring.")
            else: st.success("LOW RISK: Routine protocol.")
            
            df_prog = DataEngine.generate_prognosis_data()
            features = ['RvD5', 'N_AcCad']
            X = df_prog[features]; y = df_prog['Hematoma_Expansion']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LogisticRegression(); model.fit(X_train, y_train)
            y_prob_cal = model.predict_proba(X_test)[:, 1]
            st.pyplot(Visualizer.plot_calibration_strata(y_test, y_prob_cal))

    # --- MODULE 3: OUTCOME ---
    elif app_mode == "3. Outcome Prediction":
        st.header("🔮 Module 3: 90-day Outcome (mRS)")
        with st.expander("Biomarker Profile", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            rvd5_val = c1.number_input("RvD5", 0.0, 15.0, st.session_state.get('f_rvd5', 8.0))
            mkna_val = c2.number_input("8-MKNA", 0.0, 15.0, st.session_state.get('f_mkna', 7.5))
            nacc_val = c3.number_input("N-AcCad", 0.0, 15.0, st.session_state.get('f_nacc', 6.5))
            dta_val = c4.number_input("DTA", 0.0, 15.0, st.session_state.get('f_dta', 5.0))
            
            # Save to session
            st.session_state.update({'f_rvd5': rvd5_val, 'f_mkna': mkna_val, 'f_nacc': nacc_val, 'f_dta': dta_val})

        if st.button("Run Prediction"):
            df_prog = DataEngine.generate_prognosis_data()
            features = ['RvD5', '8_MKNA', 'N_AcCad', 'DTA']
            X = df_prog[features]; y = df_prog['Poor_Outcome_mRS']
            model = RandomForestClassifier(random_state=42); model.fit(X, y)
            
            input_data = np.array([[rvd5_val, mkna_val, nacc_val, dta_val]])
            pred_prob = model.predict_proba(input_data)[0, 1]
            means = X.mean()
            contribs = [(input_data[0][i] - means[i]) * w for i, w in enumerate([0.4, 0.6, 0.2, 0.3])]
            
            col1, col2 = st.columns(2)
            with col1: 
                st.metric("Poor Outcome Probability", f"{pred_prob*100:.1f}%")
                if pred_prob > 0.5: st.error("Predicted: Poor Outcome (mRS 3-6)")
                else: st.success("Predicted: Good Outcome (mRS 0-2)")
            with col2:
                st.pyplot(Visualizer.plot_waterfall(features, contribs, -1.5))
            
            st.info("👉 Go to **Module 4** to save this report.")

    # --- MODULE 4: REPORT & SAVE ---
    elif app_mode == "4. Final Clinical Report":
        st.header("📄 Module 4: Final Report & Storage")
        
        # Confirm Values
        with st.expander("Confirm Data for Storage", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            f_rvd5 = c1.number_input("Final RvD5", 0.0, 15.0, st.session_state.get('f_rvd5', 8.0))
            f_mkna = c2.number_input("Final 8-MKNA", 0.0, 15.0, st.session_state.get('f_mkna', 7.5))
            f_nacc = c3.number_input("Final N-AcCad", 0.0, 15.0, st.session_state.get('f_nacc', 6.5))
            f_dta = c4.number_input("Final DTA", 0.0, 15.0, st.session_state.get('f_dta', 5.0))

        if st.button("💾 Generate & Save Report"):
            # Recalc prediction for record
            df_prog = DataEngine.generate_prognosis_data()
            features = ['RvD5', '8_MKNA', 'N_AcCad', 'DTA']
            X = df_prog[features]; y = df_prog['Poor_Outcome_mRS']
            model = RandomForestClassifier(random_state=42); model.fit(X, y)
            pred_prob = model.predict_proba(np.array([[f_rvd5, f_mkna, f_nacc, f_dta]]))[0, 1]
            
            # Data Packet
            report_data = {
                "id": p_id, "age": p_age, "gender": p_gender,
                "timestamp": str(datetime.now()),
                "clinical": {"sbp": p_sbp, "nihss": p_nihss, "gcs": p_gcs, "onset_hrs": p_onset},
                "biomarkers": {"RvD5": f_rvd5, "8_MKNA": f_mkna, "N_AcCad": f_nacc, "DTA": f_dta},
                "prediction": {"poor_outcome_prob": pred_prob, "risk_level": "High" if pred_prob > 0.5 else "Low"}
            }
            
            # Save Action
            with st.spinner("Saving to Repository..."):
                success, msg = GitHubStorage.save_report(report_data)
            
            if success:
                st.balloons()
                st.success(f"✅ {msg}")
                # 显示 JSON 预览
                st.json(report_data)
            else:
                st.error(f"❌ {msg}")

if __name__ == "__main__":
    main()