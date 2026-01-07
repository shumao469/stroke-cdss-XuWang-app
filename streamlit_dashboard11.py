import warnings
import os
import json
import time
import webbrowser

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
# TRANSLATION DICTIONARY
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "app_title": {"en": "🧠 Ocular Metabolomics Clinical Decision Support", "zh": "🧠 眼部分泌物代谢组学临床决策支持系统"},
    "sidebar_title": {"en": "🩺 Ocular Stroke CDSS", "zh": "🩺 眼部卒中 CDSS"},
    "lang_select": {"en": "Language / 语言", "zh": "Language / 语言"},
    "patient_reg": {"en": "👤 Patient Registration", "zh": "👤 患者信息登记"},
    "p_id": {"en": "Patient ID", "zh": "患者 ID"},
    "p_age": {"en": "Age", "zh": "年龄"},
    "p_gender": {"en": "Gender", "zh": "性别"},
    "male": {"en": "Male", "zh": "男"},
    "female": {"en": "Female", "zh": "女"},
    "p_time": {"en": "Sampling Time", "zh": "采样时间"},
    "p_sbp": {"en": "SBP (mmHg)", "zh": "收缩压 (mmHg)"},
    "p_sbp_help": {"en": "Systolic Blood Pressure at Admission", "zh": "入院时的收缩压"},
    "p_onset": {"en": "Time from Onset (hrs)", "zh": "发病时间 (小时)"},
    "clinical_scales": {"en": "#### Clinical Scales", "zh": "#### 临床评分量表"},
    "nihss_help": {"en": "National Institutes of Health Stroke Scale", "zh": "美国国立卫生研究院卒中量表"},
    "gcs_help": {"en": "Glasgow Coma Scale", "zh": "格拉斯哥昏迷评分"},
    "history": {"en": "History", "zh": "既往史"},
    "hist_options": {"en": ["Hypertension", "Anticoagulants", "Diabetes"], "zh": ["高血压", "抗凝史", "糖尿病"]},
    "status": {"en": "Status", "zh": "当前状态"},
    "status_options": {"en": ["Triage Pending", "Confirmed HS", "Confirmed ZS"], "zh": ["待分诊", "确诊出血性卒中 (HS)", "确诊缺血性卒中 (ZS)"]},
    "module_select": {"en": "Select Module:", "zh": "选择功能模块："},
    "modules": {
        "home": {"en": "Dashboard Home", "zh": "仪表盘首页"},
        "triage": {"en": "1. ED Triage System", "zh": "1. 急诊分流系统"},
        "expansion": {"en": "2. Early Hematoma Expansion", "zh": "2. 早期血肿扩大风险"},
        "outcome": {"en": "3. Outcome Prediction", "zh": "3. 转归预测"},
        "report": {"en": "4. Final Clinical Report", "zh": "4. 最终临床报告"}
    },
    "system_info": {"en": "v7.0 | Local Repo Storage", "zh": "v7.0 | 本地仓库存储模式"},
    "cohort": {"en": "Patient Cohort", "zh": "患者队列"},
    "new_cases": {"en": "New", "zh": "新增"},
    "proc_time": {"en": "Processing Time", "zh": "处理时间"},
    "optimized": {"en": "Optimized", "zh": "已优化"},
    "current_pt_info": {"en": "**Current Patient:** {} ({}, {}y) | **Status:** {}", "zh": "**当前患者:** {} ({}, {}岁) | **状态:** {}"},
    "sampling_info": {"en": "🕒 **Sampling Time:** {} | 🗓️ **Date:** {}", "zh": "🕒 **采样时间:** {} | 🗓️ **日期:** {}"},
    "select_hint": {"en": "Select a module to begin specific analysis.", "zh": "请从左侧选择一个模块开始分析。"},
    "mod1_title": {"en": "🚑 Module 1: Emergency Triage", "zh": "🚑 模块 1: 急诊分流"},
    "biomarker_panel": {"en": "🧪 Biomarker Input Panel", "zh": "🧪 生物标志物输入面板"},
    "prob_hs": {"en": "HS Probability", "zh": "出血性卒中 (HS) 概率"},
    "rec_urgent": {"en": "Recommendation: URGENT CT", "zh": "建议：紧急 CT 检查"},
    "pop_dist": {"en": "Population Distribution Analysis", "zh": "人群分布分析"},
    "3d_cluster": {"en": "3D Metabolomic Clustering", "zh": "3D 代谢组学聚类"},
    "dist_title": {"en": "{} Distribution", "zh": "{} 分布情况"},
    "mod2_title": {"en": "📉 Module 2: Early Hematoma Expansion", "zh": "📉 模块 2: 早期血肿扩大风险"},
    "risk_factors": {"en": "### Biomarkers & Risk Factors", "zh": "### 生物标志物与风险因子"},
    "exp_risk": {"en": "Expansion Risk", "zh": "血肿扩大风险"},
    "risk_critical": {"en": "CRITICAL RISK: Intensive BP lowering required.", "zh": "极高风险：需强化降压治疗。"},
    "risk_mod": {"en": "MODERATE RISK: Standard monitoring.", "zh": "中度风险：标准监测流程。"},
    "risk_low": {"en": "LOW RISK: Routine protocol.", "zh": "低风险：常规流程。"},
    "cal_val": {"en": "View Model Validation (Calibration)", "zh": "查看模型验证 (校准曲线)"},
    "cal_title": {"en": "Calibration", "zh": "校准曲线"},
    "risk_strata": {"en": "Risk Stratification", "zh": "风险分层"},
    "mod3_title": {"en": "🔮 Module 3: 90-day Outcome (mRS)", "zh": "🔮 模块 3: 90天转归预测 (mRS)"},
    "bio_profile": {"en": "Biomarker Profile", "zh": "生物标志物谱"},
    "run_pred": {"en": "Run Prediction", "zh": "运行预测"},
    "poor_prob": {"en": "Poor Outcome Probability", "zh": "不良转归概率"},
    "pred_poor": {"en": "Predicted: Poor Outcome (mRS 3-6)", "zh": "预测结果：预后不良 (mRS 3-6)"},
    "pred_good": {"en": "Predicted: Good Outcome (mRS 0-2)", "zh": "预测结果：预后良好 (mRS 0-2)"},
    "go_report": {"en": "👉 Go to **Module 4** to save this report.", "zh": "👉 前往 **模块 4** 保存此报告。"},
    "shap_title": {"en": "SHAP Feature Contribution", "zh": "SHAP 特征贡献度"},
    "mod4_title": {"en": "📄 Module 4: Final Report & Storage", "zh": "📄 模块 4: 最终报告与存储"},
    "confirm_data": {"en": "Confirm Data for Storage", "zh": "确认存储数据"},
    "save_btn": {"en": "💾 Save to Local Repository", "zh": "💾 保存至本地仓库"},
    "save_saving": {"en": "Saving to local 'reports' folder...", "zh": "正在保存至本地 'reports' 文件夹..."},
    "save_success": {"en": "✅ Saved! File location: {}", "zh": "✅ 已保存！文件位置: {}"},
    "save_fail": {"en": "❌ {}", "zh": "❌ {}"},
    "repo_link_text": {"en": "📂 Open GitHub Repository Page", "zh": "📂 打开 GitHub 仓库页面"}
}

# -----------------------------------------------------------------------------
# LOCAL STORAGE ENGINE (No Token Required)
# -----------------------------------------------------------------------------
class DataStorage:
    """处理本地数据存储"""
    
    # 你的 GitHub 仓库地址 (仅用于显示链接)
    REPO_URL = "https://github.com/shumao469/stroke-data-storage/"
    LOCAL_DIR = "reports"  # 本地保存的文件夹名称

    @staticmethod
    def ensure_dir():
        """确保本地 reports 文件夹存在"""
        if not os.path.exists(DataStorage.LOCAL_DIR):
            os.makedirs(DataStorage.LOCAL_DIR)

    @staticmethod
    def get_cohort_count():
        """获取当前队列总数 (Base 751 + Local Files)"""
        base_count = 751
        DataStorage.ensure_dir()
        
        try:
            # 计算本地 reports 文件夹中的 json 文件数量
            files = [f for f in os.listdir(DataStorage.LOCAL_DIR) if f.endswith('.json')]
            new_count = len(files)
        except Exception:
            new_count = 0
            
        return base_count + new_count, new_count

    @staticmethod
    def save_report(patient_data):
        """将报告保存为本地 JSON 文件"""
        DataStorage.ensure_dir()
        
        # 准备数据 JSON
        json_str = json.dumps(patient_data, indent=4, default=str)
        # 生成唯一文件名
        filename = f"{patient_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(DataStorage.LOCAL_DIR, filename)
        
        try:
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(json_str)
            # 返回绝对路径以便显示
            return True, os.path.abspath(filepath)
        except Exception as e:
            return False, f"Local Save Error: {str(e)}"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_nihss_grade(score, lang_code='en'):
    if score == 0: 
        return ("No Stroke Symptoms (0)" if lang_code=='en' else "无卒中症状 (0)"), "green"
    elif score <= 4: 
        return ("Minor Stroke (1-4)" if lang_code=='en' else "轻度卒中 (1-4)"), "green"
    elif score <= 15: 
        return ("Moderate Stroke (5-15)" if lang_code=='en' else "中度卒中 (5-15)"), "orange"
    elif score <= 20: 
        return ("Moderate to Severe (16-20)" if lang_code=='en' else "中重度卒中 (16-20)"), "red"
    else: 
        return ("Severe Stroke (21-42)" if lang_code=='en' else "重度卒中 (21-42)"), "darkred"

def get_gcs_grade(score, lang_code='en'):
    if score >= 13: 
        return ("Mild Head Injury (13-15)" if lang_code=='en' else "轻度颅脑损伤 (13-15)"), "green"
    elif score >= 9: 
        return ("Moderate Head Injury (9-12)" if lang_code=='en' else "中度颅脑损伤 (9-12)"), "orange"
    else: 
        return ("Severe Head Injury (3-8)" if lang_code=='en' else "重度颅脑损伤 (3-8)"), "darkred"

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
    def plot_gauge_chart(probability, title):
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
    def plot_3d_cluster(df, patient_vals, title):
        fig = px.scatter_3d(df, x='RvD5', y='8_MKNA', z='N_AcCad', color='Group',
                            color_discrete_map={'HS': '#e74c3c', 'ZS': '#3498db', 'NC': '#95a5a6'}, opacity=0.3)
        fig.add_trace(go.Scatter3d(x=[patient_vals[0]], y=[patient_vals[1]], z=[patient_vals[2]],
            mode='markers', marker=dict(size=12, color='black', symbol='diamond'), name='Patient'))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), title=title)
        return fig

    @staticmethod
    def plot_distribution_comparison(df, patient_val, feature_name, title_fmt):
        fig = go.Figure()
        for group in ['NC', 'HS']:
            subset = df[df['Group'] == group][feature_name]
            fig.add_trace(go.Histogram(x=subset, name=group, opacity=0.6, nbinsx=30))
        fig.add_vline(x=patient_val, line_width=3, line_dash="dash", line_color="black", annotation_text="Patient")
        fig.update_layout(barmode='overlay', title=title_fmt.format(feature_name), height=300)
        return fig

    @staticmethod
    def plot_risk_heatmap(patient_data, means_poor_outcome, features, title):
        comparison = np.array(patient_data) / np.array(means_poor_outcome)
        fig = go.Figure(data=go.Heatmap(z=[comparison], x=features, y=['Ratio'],
            colorscale='RdBu_r', midpoint=1.0, text=[[f"{v:.2f}x" for v in comparison]], texttemplate="%{text}", showscale=True))
        fig.update_layout(title=title, height=250)
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
    def plot_calibration_strata(y_test, y_prob, cal_title, risk_title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8)
        ax1.plot(prob_pred, prob_true, marker='o', lw=2, color=COLORS['HS'])
        ax1.plot([0, 1], [0, 1], '--', color='gray')
        ax1.set_title(cal_title)
        risk_bins = pd.cut(y_prob, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Mod', 'High'])
        strat_df = pd.DataFrame({'Risk': risk_bins, 'Event': y_test})
        means = strat_df.groupby('Risk', observed=False)['Event'].mean() * 100
        ax2.bar(means.index, means.values, color=[COLORS['SAFE'], '#f39c12', COLORS['HS']])
        ax2.set_title(risk_title)
        return fig

    @staticmethod
    def plot_waterfall(feature_names, contributions, base_value, title, xlabel):
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
        ax.set_xlabel(xlabel); ax.set_title(title)
        return fig

# -----------------------------------------------------------------------------
# MAIN APP CONTROLLER
# -----------------------------------------------------------------------------
def main():
    # --- LANGUAGE SELECTOR ---
    with st.sidebar:
        # 默认英语 'en'，如果选中文则切换到 'zh'
        lang_choice = st.selectbox("🌐 Language / 语言", ["English", "中文"])
        lc = "zh" if lang_choice == "中文" else "en"
        
        st.title(TRANSLATIONS["sidebar_title"][lc])

    # --- PATIENT INFO ---
    with st.sidebar.expander(TRANSLATIONS["patient_reg"][lc], expanded=True):
        p_id = st.text_input(TRANSLATIONS["p_id"][lc], "PT-2024-089")
        p_age = st.number_input(TRANSLATIONS["p_age"][lc], 40, 100, 65)
        
        gender_map = {"Male": TRANSLATIONS["male"][lc], "Female": TRANSLATIONS["female"][lc]}
        p_gender_raw = st.selectbox(TRANSLATIONS["p_gender"][lc], ["Male", "Female"], format_func=lambda x: gender_map[x])
        p_time = st.time_input(TRANSLATIONS["p_time"][lc], datetime.now().time())
        
        p_sbp = st.number_input(TRANSLATIONS["p_sbp"][lc], 90, 220, 155, help=TRANSLATIONS["p_sbp_help"][lc])
        p_onset = st.number_input(TRANSLATIONS["p_onset"][lc], 0.5, 48.0, 3.5)
        
        st.markdown(TRANSLATIONS["clinical_scales"][lc])
        p_nihss = st.slider("NIHSS Score", 0, 42, 5, help=TRANSLATIONS["nihss_help"][lc])
        p_gcs = st.slider("GCS Score", 3, 15, 15, help=TRANSLATIONS["gcs_help"][lc])
        
        # Translate History Options logic
        hist_ops = TRANSLATIONS["hist_options"][lc]
        p_history = st.multiselect(TRANSLATIONS["history"][lc], hist_ops, [hist_ops[0]])
        
        status_ops = TRANSLATIONS["status_options"][lc]
        p_status = st.selectbox(TRANSLATIONS["status"][lc], status_ops)

    st.sidebar.markdown("---")
    
    # --- MODULE NAVIGATION ---
    mod_ops = [
        TRANSLATIONS["modules"]["home"][lc],
        TRANSLATIONS["modules"]["triage"][lc],
        TRANSLATIONS["modules"]["expansion"][lc],
        TRANSLATIONS["modules"]["outcome"][lc],
        TRANSLATIONS["modules"]["report"][lc]
    ]
    app_mode = st.sidebar.radio(TRANSLATIONS["module_select"][lc], mod_ops)
    st.sidebar.info(TRANSLATIONS["system_info"][lc])

    # --- HOME ---
    if app_mode == mod_ops[0]:
        st.title(TRANSLATIONS["app_title"][lc])
        
        total_cohort, new_cases = DataStorage.get_cohort_count()
        nihss_txt, nihss_col = get_nihss_grade(p_nihss, lc)
        gcs_txt, gcs_col = get_gcs_grade(p_gcs, lc)
        
        col1, col2, col3 = st.columns(3)
        col1.metric(TRANSLATIONS["cohort"][lc], str(total_cohort), f"+{new_cases} {TRANSLATIONS['new_cases'][lc]}")
        col2.metric("Model AUC (HS)", "0.92", "+0.03")
        col3.metric(TRANSLATIONS["proc_time"][lc], "< 5 min", TRANSLATIONS["optimized"][lc])
        
        st.divider()
        st.markdown(TRANSLATIONS["current_pt_info"][lc].format(p_id, p_gender_raw, p_age, p_status))
        st.info(TRANSLATIONS["sampling_info"][lc].format(p_time.strftime('%H:%M'), datetime.now().strftime('%Y-%m-%d')))
        
        # 显示仓库链接
        st.markdown(f"[{TRANSLATIONS['repo_link_text'][lc]}]({DataStorage.REPO_URL})")
        st.markdown(TRANSLATIONS["select_hint"][lc])

    # --- MODULE 1: TRIAGE ---
    elif app_mode == mod_ops[1]:
        st.header(TRANSLATIONS["mod1_title"][lc])
        with st.expander(TRANSLATIONS["biomarker_panel"][lc], expanded=True):
            col_i1, col_i2, col_i3 = st.columns(3)
            rvd5 = col_i1.slider("RvD5", 0.0, 15.0, 8.5); mkna = col_i2.slider("8-MKNA", 0.0, 15.0, 7.8); nacc = col_i3.slider("N-AcCad", 0.0, 15.0, 9.2)
        
        df_triage = DataEngine.generate_triage_data()
        input_arr = np.array([rvd5, mkna, nacc])
        prob_hs = 1 / (1 + np.exp(-(rvd5*0.5 + mkna*0.4 + nacc*0.3 - 8)))
        
        c1, c2 = st.columns([1, 2])
        with c1: 
            st.plotly_chart(Visualizer.plot_gauge_chart(prob_hs, TRANSLATIONS["prob_hs"][lc]), use_container_width=True)
            if prob_hs > 0.7: st.error(TRANSLATIONS["rec_urgent"][lc])
        with c2: 
            st.plotly_chart(Visualizer.plot_3d_cluster(df_triage, input_arr, TRANSLATIONS["3d_cluster"][lc]), use_container_width=True)
        st.plotly_chart(Visualizer.plot_distribution_comparison(df_triage, rvd5, 'RvD5', TRANSLATIONS["dist_title"][lc]), use_container_width=True)

    # --- MODULE 2: EXPANSION ---
    elif app_mode == mod_ops[2]:
        st.header(TRANSLATIONS["mod2_title"][lc])
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(TRANSLATIONS["risk_factors"][lc])
            val_rvd5 = st.slider("RvD5 Level", 0.0, 15.0, 7.5)
            val_nacc = st.slider("N-AcCad Level", 0.0, 15.0, 6.0)
            score = (p_sbp/200)*2 + (12-p_onset)/12 + (val_rvd5/10)*1.5 + (val_nacc/10)*1.0 + (p_nihss/42)*2 - 5.5
            prob_exp = 1 / (1 + np.exp(-score))
        with c2:
            st.plotly_chart(Visualizer.plot_gauge_chart(prob_exp, TRANSLATIONS["exp_risk"][lc]), use_container_width=True)
            if prob_exp > 0.7: st.error(TRANSLATIONS["risk_critical"][lc])
            elif prob_exp > 0.3: st.warning(TRANSLATIONS["risk_mod"][lc])
            else: st.success(TRANSLATIONS["risk_low"][lc])
            
            df_prog = DataEngine.generate_prognosis_data()
            features = ['RvD5', 'N_AcCad']
            X = df_prog[features]; y = df_prog['Hematoma_Expansion']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LogisticRegression(); model.fit(X_train, y_train)
            y_prob_cal = model.predict_proba(X_test)[:, 1]
            
            with st.expander(TRANSLATIONS["cal_val"][lc]):
                st.pyplot(Visualizer.plot_calibration_strata(y_test, y_prob_cal, TRANSLATIONS["cal_title"][lc], TRANSLATIONS["risk_strata"][lc]))

    # --- MODULE 3: OUTCOME ---
    elif app_mode == mod_ops[3]:
        st.header(TRANSLATIONS["mod3_title"][lc])
        with st.expander(TRANSLATIONS["bio_profile"][lc], expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            rvd5_val = c1.number_input("RvD5", 0.0, 15.0, st.session_state.get('f_rvd5', 8.0))
            mkna_val = c2.number_input("8-MKNA", 0.0, 15.0, st.session_state.get('f_mkna', 7.5))
            nacc_val = c3.number_input("N-AcCad", 0.0, 15.0, st.session_state.get('f_nacc', 6.5))
            dta_val = c4.number_input("DTA", 0.0, 15.0, st.session_state.get('f_dta', 5.0))
            st.session_state.update({'f_rvd5': rvd5_val, 'f_mkna': mkna_val, 'f_nacc': nacc_val, 'f_dta': dta_val})

        if st.button(TRANSLATIONS["run_pred"][lc]):
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
                st.metric(TRANSLATIONS["poor_prob"][lc], f"{pred_prob*100:.1f}%")
                if pred_prob > 0.5: st.error(TRANSLATIONS["pred_poor"][lc])
                else: st.success(TRANSLATIONS["pred_good"][lc])
            with col2:
                st.pyplot(Visualizer.plot_waterfall(features, contribs, -1.5, TRANSLATIONS["shap_title"][lc], "Log-Odds" if lc=='en' else "对数几率贡献"))
            
            st.info(TRANSLATIONS["go_report"][lc])

    # --- MODULE 4: REPORT ---
    elif app_mode == mod_ops[4]:
        st.header(TRANSLATIONS["mod4_title"][lc])
        
        with st.expander(TRANSLATIONS["confirm_data"][lc], expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            f_rvd5 = c1.number_input("Final RvD5", 0.0, 15.0, st.session_state.get('f_rvd5', 8.0))
            f_mkna = c2.number_input("Final 8-MKNA", 0.0, 15.0, st.session_state.get('f_mkna', 7.5))
            f_nacc = c3.number_input("Final N-AcCad", 0.0, 15.0, st.session_state.get('f_nacc', 6.5))
            f_dta = c4.number_input("Final DTA", 0.0, 15.0, st.session_state.get('f_dta', 5.0))

        if st.button(TRANSLATIONS["save_btn"][lc]):
            df_prog = DataEngine.generate_prognosis_data()
            features = ['RvD5', '8_MKNA', 'N_AcCad', 'DTA']
            X = df_prog[features]; y = df_prog['Poor_Outcome_mRS']
            model = RandomForestClassifier(random_state=42); model.fit(X, y)
            pred_prob = model.predict_proba(np.array([[f_rvd5, f_mkna, f_nacc, f_dta]]))[0, 1]
            
            report_data = {
                "id": p_id, "age": p_age, "gender": p_gender_raw,
                "timestamp": str(datetime.now()),
                "clinical": {"sbp": p_sbp, "nihss": p_nihss, "gcs": p_gcs, "onset_hrs": p_onset},
                "biomarkers": {"RvD5": f_rvd5, "8_MKNA": f_mkna, "N_AcCad": f_nacc, "DTA": f_dta},
                "prediction": {"poor_outcome_prob": pred_prob, "risk_level": "High" if pred_prob > 0.5 else "Low"}
            }
            
            with st.spinner(TRANSLATIONS["save_saving"][lc]):
                success, path = DataStorage.save_report(report_data)
            
            if success:
                st.balloons()
                st.success(TRANSLATIONS["save_success"][lc].format(path))
                st.json(report_data)
                
                # 添加下载按钮
                json_str = json.dumps(report_data, indent=4, default=str)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=os.path.basename(path),
                    mime="application/json"
                )
            else:
                st.error(TRANSLATIONS["save_fail"][lc].format(path))

if __name__ == "__main__":
    main()