import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phát hiện Giao dịch Bất thường",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để tạo giao diện cao cấp (Premium Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient Header */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #b0bec5;
        font-weight: 300;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(0, 114, 255, 0.3);
    }
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8892b0;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-value.anomaly {
        color: #ff4a5a;
    }
    .metric-value.safe {
        color: #00e676;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 0.75rem 1.5rem;
        color: #8892b0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 114, 255, 0.05);
        color: #00c6ff;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 114, 255, 0.1) !important;
        border-bottom: 2px solid #0072ff !important;
        color: #00c6ff !important;
    }
    
    /* Alert Boxes */
    .alert-success {
        padding: 1rem;
        background-color: rgba(0, 230, 118, 0.1);
        border-left: 5px solid #00e676;
        color: #b9f6ca;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .alert-danger {
        padding: 1rem;
        background-color: rgba(255, 74, 90, 0.1);
        border-left: 5px solid #ff4a5a;
        color: #ffcdd2;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG
# ----------------------------------------------------

@st.cache_data
def load_data(filepath):
    """Đọc dữ liệu thô từ file CSV"""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Xử lý và tạo đặc trưng (Feature Engineering)"""
    data = df.copy()
    
    # 1. Định dạng lại ngày tháng và tạo các đặc trưng thời gian
    data['transaction_date'] = pd.to_datetime(data['transaction_date'], format='%d/%m/%Y %H:%M')
    data['hour'] = data['transaction_date'].dt.hour
    data['day_of_week'] = data['transaction_date'].dt.dayofweek
    
    # 2. Xử lý cột nhân viên
    if 'is_employee' in data.columns:
        data['is_employee'] = data['is_employee'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
        data['is_employee'] = data['is_employee'].fillna(0).astype(int)
        
    # 3. Tạo nhãn default (1: Bất thường nếu ID chứa ANOM, 0: Bình thường)
    data['default'] = data['transaction_id'].str.contains('ANOM').astype(int)
    
    # 4. Ép kiểu dữ liệu amount
    data['amount'] = pd.to_numeric(data['amount'], errors='coerce').fillna(0)
    
    # 5. Đặc trưng thời gian đặc biệt (Giao dịch đêm từ 0h-5h)
    data['is_nighttime'] = ((data['hour'] >= 0) & (data['hour'] < 5)).astype(int)
    
    # 6. Các đặc trưng gom nhóm theo khách hàng (customer)
    cust_avg_amt = data.groupby('customer_id_hash')['amount'].transform('mean')
    cust_count = data.groupby('customer_id_hash')['amount'].transform('count')
    
    data['cust_avg_amount'] = cust_avg_amt
    data['cust_tx_count'] = cust_count
    
    # Chỉ số lệch so với trung bình của chính khách hàng đó
    cust_std_amt = data.groupby('customer_id_hash')['amount'].transform('std').fillna(0)
    data['cust_amt_zscore'] = (data['amount'] - data['cust_avg_amount']) / (cust_std_amt + 1e-5)
    
    return data

def build_features(data, is_training=True):
    """Mã hóa One-hot cho các biến danh mục và chuẩn bị tập X, y"""
    cat_features = ['transaction_type', 'channel', 'location']
    
    # Thực hiện Get Dummies
    df_encoded = pd.get_dummies(data, columns=cat_features, drop_first=True)
    
    # Loại bỏ các cột không dùng làm đặc trưng huấn luyện
    cols_to_drop = [
        'transaction_id', 'transaction_date', 'customer_id_hash', 
        'account_no_hash', 'counterparty_bank', 'status', 'default'
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df_encoded.columns]
    
    X = df_encoded.drop(columns=cols_to_drop)
    y = df_encoded['default'] if 'default' in df_encoded.columns else None
    
    # Đảm bảo toàn bộ là dữ liệu số
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
    return X, y

# ----------------------------------------------------
# 2. HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
# ----------------------------------------------------

@st.cache_resource
def train_and_evaluate(X, y, test_size, random_state, rf_estimators):
    """Huấn luyện 3 mô hình và trả về kết quả chi tiết"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 1. Logistic Regression
    model_lr = LogisticRegression(max_iter=1000, random_state=random_state)
    model_lr.fit(X_train, y_train)
    
    # 2. Decision Tree
    model_dt = DecisionTreeClassifier(random_state=random_state)
    model_dt.fit(X_train, y_train)
    
    # 3. Random Forest
    model_rf = RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state)
    model_rf.fit(X_train, y_train)
    
    models = {
        'Logistic Regression': model_lr,
        'Decision Tree': model_dt,
        'Random Forest': model_rf
    }
    
    eval_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        eval_results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    return eval_results, X_train.columns.tolist()

# ----------------------------------------------------
# 3. GIAO DIỆN CHÍNH STREAMLIT
# ----------------------------------------------------

# Header ứng dụng
st.markdown("""
<div class="main-header">
    <h1>🛡️ ANTIGRAVITY TRANSACTION ANOMALY RADAR</h1>
    <p>Hệ thống giám sát và phát hiện giao dịch bất thường trong thời gian thực sử dụng trí tuệ nhân tạo</p>
</div>
""", unsafe_allow_html=True)

# Khởi tạo đường dẫn dữ liệu mặc định
DEFAULT_DATA_PATH = "transactions_Q1_demo.csv"

# SIDEBAR: Cấu hình dữ liệu và tham số mô hình
st.sidebar.markdown("### ⚙️ Cấu hình dữ liệu & Mô hình")

# Upload file dữ liệu
uploaded_file = st.sidebar.file_uploader(
    "Tải lên tệp dữ liệu giao dịch (CSV)", 
    type=["csv"]
)

# Đọc dữ liệu
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Tải dữ liệu thành công!")
else:
    raw_df = load_data(DEFAULT_DATA_PATH)
    if raw_df is not None:
        st.sidebar.info("Đang sử dụng dữ liệu mẫu: `transactions_Q1_demo.csv`")
    else:
        st.sidebar.error("Không tìm thấy dữ liệu mẫu. Hãy tải lên file CSV để bắt đầu.")

if raw_df is not None:
    # Tiền xử lý dữ liệu nền
    processed_df = preprocess_data(raw_df)
    X, y = build_features(processed_df)
    
    # Tham số huấn luyện mô hình ở Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Tham số Huấn luyện")
    test_ratio = st.sidebar.slider("Tỷ lệ tập Test (%)", 10, 50, 20, step=5) / 100.0
    random_seed = st.sidebar.number_input("Random Seed (Trạng thái ngẫu nhiên)", value=32)
    rf_trees = st.sidebar.slider("Số lượng cây trong Random Forest (n_estimators)", 10, 200, 100, step=10)
    
    # Huấn luyện mô hình tự động
    results, feature_cols = train_and_evaluate(X, y, test_ratio, random_seed, rf_trees)
    
    # Lấy ra mô hình tốt nhất dựa trên F1-score
    best_model_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model_info = results[best_model_name]
    
    # Khai báo các tab giao diện chính
    tab_overview, tab_model, tab_detection = st.tabs([
        "📊 Dashboard Tổng quan", 
        "🧠 Huấn luyện & Đánh giá", 
        "🔍 Phát hiện Giao dịch"
    ])
    
    # ==========================================
    # TAB 1: DASHBOARD TỔNG QUAN
    # ==========================================
    with tab_overview:
        st.markdown("### 📈 Chỉ số giao dịch toàn hệ thống")
        
        # Tính toán các chỉ số KPIs
        total_tx = len(processed_df)
        total_anoms = processed_df['default'].sum()
        anom_rate = (total_anoms / total_tx) * 100
        total_anom_amount = processed_df[processed_df['default'] == 1]['amount'].sum()
        
        # Giao diện KPI Grid
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tổng số giao dịch</div>
                <div class="metric-value">{total_tx:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Giao dịch bất thường</div>
                <div class="metric-value anomaly">{total_anoms:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tỷ lệ bất thường</div>
                <div class="metric-value anomaly">{anom_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tổng giá trị nghi ngờ</div>
                <div class="metric-value" style="color:#ffb74d;">{total_anom_amount:,.0f} VND</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### 📊 Trực quan hóa dữ liệu giao dịch")
        
        # Chia cột vẽ biểu đồ
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 1. Biểu đồ đường: Số lượng giao dịch và số giao dịch bất thường theo thời gian (Hàng ngày)
            daily_stats = processed_df.groupby(processed_df['transaction_date'].dt.date).agg(
                total_tx=('transaction_id', 'count'),
                anom_tx=('default', 'sum')
            ).reset_index()
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=daily_stats['transaction_date'], y=daily_stats['total_tx'],
                mode='lines', name='Tổng số giao dịch',
                line=dict(color='#0072ff', width=3)
            ))
            fig_time.add_trace(go.Scatter(
                x=daily_stats['transaction_date'], y=daily_stats['anom_tx'],
                mode='lines', name='Giao dịch bất thường',
                line=dict(color='#ff4a5a', width=2, dash='dot')
            ))
            fig_time.update_layout(
                title="Tần suất Giao dịch Hàng ngày (Q1/2026)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Ngày",
                yaxis_title="Số giao dịch"
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        with col_chart2:
            # 2. Biểu đồ Donut: Phân bố giao dịch theo Kênh (Channel)
            channel_counts = processed_df['channel'].value_counts().reset_index()
            fig_channel = px.pie(
                channel_counts, names='channel', values='count',
                hole=0.4, title="Phân bố Giao dịch theo Kênh",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_channel.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_channel, use_container_width=True)
            
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            # 3. Biểu đồ cột: Tỷ lệ bất thường theo Chi nhánh (Location)
            loc_stats = processed_df.groupby('location').agg(
                total=('transaction_id', 'count'),
                anom=('default', 'sum')
            ).reset_index()
            loc_stats['anom_rate'] = (loc_stats['anom'] / loc_stats['total']) * 100
            loc_stats = loc_stats.sort_values(by='anom_rate', ascending=False)
            
            fig_loc = px.bar(
                loc_stats, x='location', y='anom_rate',
                title="Tỷ lệ Giao dịch Bất thường (%) theo Chi nhánh",
                labels={'anom_rate': 'Tỷ lệ bất thường (%)', 'location': 'Chi nhánh'},
                color='anom_rate',
                color_continuous_scale='Reds'
            )
            fig_loc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_loc, use_container_width=True)
            
        with col_chart4:
            # 4. Phân bố giao dịch theo giờ trong ngày
            hour_stats = processed_df.groupby(['hour', 'default']).size().reset_index(name='count')
            hour_stats['Trạng thái'] = hour_stats['default'].map({0: 'Bình thường', 1: 'Bất thường'})
            
            fig_hour = px.bar(
                hour_stats, x='hour', y='count', color='Trạng thái',
                title="Số lượng giao dịch theo Giờ trong ngày",
                labels={'hour': 'Giờ trong ngày (0h-23h)', 'count': 'Số lượng giao dịch'},
                color_discrete_map={'Bình thường': '#0072ff', 'Bất thường': '#ff4a5a'}
            )
            fig_hour.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode='stack'
            )
            st.plotly_chart(fig_hour, use_container_width=True)
            
    # ==========================================
    # TAB 2: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
    # ==========================================
    with tab_model:
        st.markdown("### 🧠 Kết quả so sánh 3 Mô hình")
        st.markdown(f"Đang sử dụng Random Seed = `{random_seed}` và tỷ lệ Train/Test = `{1-test_ratio:.0%}/{test_ratio:.0%}`.")
        
        # Xây dựng bảng so sánh kết quả
        models_data = []
        for name, metrics in results.items():
            models_data.append({
                'Mô hình': name,
                'Độ chính xác (Accuracy)': f"{metrics['accuracy']:.4f}",
                'Độ chuẩn xác (Precision)': f"{metrics['precision']:.4f}",
                'Độ nhạy (Recall)': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        comparison_df = pd.DataFrame(models_data)
        st.table(comparison_df)
        
        # Biểu đồ so sánh F1-score
        f1_df = pd.DataFrame({
            'Mô hình': list(results.keys()),
            'F1-Score': [results[name]['f1_score'] for name in results.keys()]
        })
        fig_f1 = px.bar(
            f1_df, x='Mô hình', y='F1-Score',
            title="So sánh chỉ số F1-Score giữa các mô hình",
            color='F1-Score',
            color_continuous_scale='Blues',
            text_auto='.3f'
        )
        fig_f1.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 1.1]
        )
        st.plotly_chart(fig_f1, use_container_width=True)
        
        st.markdown("---")
        
        # Khám phá sâu chi tiết từng mô hình
        st.markdown("### 🔍 Phân tích chi tiết mô hình")
        selected_model_name = st.selectbox("Chọn mô hình để phân tích sâu:", list(results.keys()), index=2)
        
        model_details = results[selected_model_name]
        col_det1, col_det2 = st.columns(2)
        
        with col_det1:
            # Vẽ Confusion Matrix
            cm = model_details['confusion_matrix']
            # Labels
            labels = ["Bình thường", "Bất thường"]
            
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=labels, y=labels,
                color_continuous_scale='YlOrRd',
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                title=f"Ma trận Nhầm lẫn (Confusion Matrix) - {selected_model_name}"
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_det2:
            # Vẽ độ quan trọng đặc trưng (Feature Importance) nếu mô hình hỗ trợ
            fitted_model = model_details['model']
            if hasattr(fitted_model, 'feature_importances_'):
                importances = fitted_model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    'Đặc trưng': feature_cols,
                    'Độ quan trọng': importances
                }).sort_values(by='Độ quan trọng', ascending=True).tail(10) # Lấy top 10 đặc trưng
                
                fig_imp = px.bar(
                    feat_imp_df, x='Độ quan trọng', y='Đặc trưng',
                    orientation='h',
                    title=f"Top 10 Đặc trưng Quan trọng nhất - {selected_model_name}",
                    color='Độ quan trọng',
                    color_continuous_scale='Purples'
                )
                fig_imp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                # Đối với Logistic Regression hiển thị Hệ số trọng số (Coefficients)
                coefs = fitted_model.coef_[0]
                coef_df = pd.DataFrame({
                    'Đặc trưng': feature_cols,
                    'Trọng số Hệ số': coefs
                })
                # Sắp xếp theo trị tuyệt đối
                coef_df['abs_weight'] = coef_df['Trọng số Hệ số'].abs()
                coef_df = coef_df.sort_values(by='abs_weight', ascending=True).tail(10)
                
                fig_imp = px.bar(
                    coef_df, x='Trọng số Hệ số', y='Đặc trưng',
                    orientation='h',
                    title=f"Top 10 Hệ số Trọng số lớn nhất - {selected_model_name}",
                    color='Trọng số Hệ số',
                    color_continuous_scale='RdBu_r'
                )
                fig_imp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # ==========================================
    # TAB 3: PHÁT HIỆN GIAO DỊCH
    # ==========================================
    with tab_detection:
        st.markdown(f"**Mô hình đang được áp dụng:** `{best_model_name}` (Là mô hình có chỉ số F1-Score tốt nhất: `{best_model_info['f1_score']:.4f}`)")
        
        mode_detection = st.radio(
            "Chọn hình thức kiểm tra:", 
            ["Kiểm tra thủ công một giao dịch (Ad-hoc check)", "Quét giao dịch hàng loạt từ file mới (Batch Scan)"]
        )
        
        if mode_detection == "Kiểm tra thủ công một giao dịch (Ad-hoc check)":
            st.markdown("### 🔍 Điền thông tin giao dịch cần quét")
            
            # Form nhập liệu giao dịch
            with st.form("manual_tx_form"):
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    input_amount = st.number_input("Số tiền giao dịch (VND)", min_value=0, value=1500000)
                    input_type = st.selectbox("Loại giao dịch", processed_df['transaction_type'].unique())
                    input_channel = st.selectbox("Kênh giao dịch", processed_df['channel'].unique())
                with col_f2:
                    input_date = st.date_input("Ngày giao dịch")
                    input_time = st.time_input("Giờ giao dịch")
                    input_location = st.selectbox("Chi nhánh thực hiện", processed_df['location'].unique())
                with col_f3:
                    input_customer_id = st.text_input("Mã băm khách hàng (customer_id_hash)", "5627107b9a5a354e")
                    input_acc = st.text_input("Mã băm tài khoản (account_no_hash)", "c94df1f2bb40c7aa")
                    input_is_employee = st.checkbox("Khách hàng là Nhân viên ngân hàng", value=False)
                    
                submit_button = st.form_submit_button("🛡️ Quét giao dịch")
                
            if submit_button:
                # Tổ hợp thông tin
                txn_datetime_str = f"{input_date.strftime('%d/%m/%Y')} {input_time.strftime('%H:%M')}"
                single_txn = {
                    'transaction_id': 'TXN_MANUAL_CHECK',
                    'transaction_date': txn_datetime_str,
                    'customer_id_hash': input_customer_id,
                    'account_no_hash': input_acc,
                    'amount': input_amount,
                    'transaction_type': input_type,
                    'channel': input_channel,
                    'counterparty_bank': 'INTERNAL',
                    'status': 'COMPLETED',
                    'location': input_location,
                    'is_employee': 'TRUE' if input_is_employee else 'FALSE'
                }
                
                # Thực hiện dự đoán
                # Chuyển đổi single txn thành DataFrame và ghép với dữ liệu mẫu để căn chỉnh dummies cột
                temp_df = pd.concat([raw_df, pd.DataFrame([single_txn])], ignore_index=True)
                temp_processed = preprocess_data(temp_df)
                X_temp, _ = build_features(temp_processed)
                
                # Căn chỉnh để khớp các cột thuộc tính của mô hình
                for col in feature_cols:
                    if col not in X_temp.columns:
                        X_temp[col] = 0
                X_single = X_temp[feature_cols].iloc[[-1]]
                
                # Dự báo xác suất
                pred_label = best_model_info['model'].predict(X_single)[0]
                pred_prob = best_model_info['model'].predict_proba(X_single)[0][1]
                
                # Hiển thị kết quả cảnh báo
                st.markdown("---")
                st.markdown("### 🔔 Kết quả kiểm tra:")
                if pred_label == 1:
                    st.markdown(f"""
                    <div class="alert-danger">
                        <h4>⚠️ PHÁT HIỆN RỦI RO BẤT THƯỜNG (ANOMALY DETECTED)</h4>
                        <p>Giao dịch này có xác suất gian lận/bất thường là <b>{pred_prob*100:.2f}%</b>.</p>
                        <ul>
                            <li><b>Lý do nghi ngờ chính:</b> Số tiền lớn giao dịch bất thường hoặc giao dịch ngoài giờ hành chính qua kênh rủi ro.</li>
                            <li><b>Khuyến nghị hành động:</b> Tạm giữ giao dịch và chuyển trạng thái sang xác thực KYC nâng cao hoặc liên hệ khách hàng xác minh.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-success">
                        <h4>✅ GIAO DỊCH AN TOÀN (NORMAL TRANSACTION)</h4>
                        <p>Giao dịch này được đánh giá là Bình thường với độ tin cậy an toàn là <b>{(1 - pred_prob)*100:.2f}%</b> (xác suất rủi ro chỉ {pred_prob*100:.2f}%).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        elif mode_detection == "Quét giao dịch hàng loạt từ file mới (Batch Scan)":
            st.markdown("### 📂 Quét bất thường hàng loạt từ file dữ liệu mới")
            st.markdown("File tải lên cần có các cột giống cấu trúc: `transaction_id`, `transaction_date`, `customer_id_hash`, `account_no_hash`, `amount`, `transaction_type`, `channel`, `location`, `is_employee`.")
            
            uploaded_batch = st.file_uploader("Chọn file giao dịch mới (CSV hoặc Excel)", type=["csv", "xlsx"])
            
            if uploaded_batch is not None:
                if uploaded_batch.name.endswith('.csv'):
                    batch_df = pd.read_csv(uploaded_batch)
                else:
                    batch_df = pd.read_excel(uploaded_batch)
                
                st.write(f"Tải lên thành công: `{len(batch_df)}` giao dịch cần xử lý.")
                
                if st.button("🚀 Bắt đầu Quét Hệ thống"):
                    with st.spinner("Hệ thống đang tiền xử lý dữ liệu và áp dụng mô hình AI..."):
                        # Trộn với dữ liệu gốc để giữ vững khớp Dummy
                        combined_batch = pd.concat([raw_df, batch_df], ignore_index=True)
                        combined_processed = preprocess_data(combined_batch)
                        X_comb, _ = build_features(combined_processed)
                        
                        # Căn chỉnh thuộc tính
                        for col in feature_cols:
                            if col not in X_comb.columns:
                                X_comb[col] = 0
                                
                        # Lấy lại các hàng của file batch mới tải lên (ở cuối danh sách)
                        X_batch = X_comb[feature_cols].iloc[-len(batch_df):]
                        
                        # Dự báo
                        preds = best_model_info['model'].predict(X_batch)
                        probs = best_model_info['model'].predict_proba(X_batch)[:, 1]
                        
                        # Ghép lại vào dataframe kết quả
                        result_batch = batch_df.copy()
                        result_batch['Dự đoán'] = preds
                        result_batch['Xác suất bất thường (%)'] = np.round(probs * 100, 2)
                        result_batch['Trạng thái'] = result_batch['Dự đoán'].map({0: 'Bình thường', 1: 'CẢNH BÁO: Bất thường'})
                        
                        # Lọc ra các giao dịch bất thường phát hiện được
                        anomalies_detected = result_batch[result_batch['Dự đoán'] == 1]
                        
                        st.markdown("---")
                        st.markdown("### 🔔 Kết quả quét hàng loạt")
                        
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            st.metric("Tổng số giao dịch đã quét", f"{len(result_batch)}")
                        with col_r2:
                            st.metric("Giao dịch bất thường phát hiện", f"{len(anomalies_detected)}", delta=f"{len(anomalies_detected)}", delta_color="inverse")
                        with col_r3:
                            st.metric("Tỷ lệ bất thường phát hiện", f"{(len(anomalies_detected)/len(result_batch))*100:.2f}%")
                            
                        # Hiển thị bảng giao dịch bất thường
                        st.markdown("#### 🚨 Danh sách các giao dịch bất thường phát hiện được")
                        if len(anomalies_detected) > 0:
                            st.dataframe(anomalies_detected[[
                                'transaction_id', 'transaction_date', 'customer_id_hash', 
                                'amount', 'transaction_type', 'channel', 'location', 'Xác suất bất thường (%)'
                            ]].sort_values(by='Xác suất bất thường (%)', ascending=False))
                            
                            # Nút tải báo cáo bất thường
                            csv_data = anomalies_detected.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Tải xuống danh sách giao dịch bất thường (CSV)",
                                data=csv_data,
                                file_name="detected_anomalous_transactions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.success("🎉 Tuyệt vời! Hệ thống không phát hiện bất kỳ giao dịch bất thường nào trong tệp này.")
else:
    st.warning("⚠️ Vui lòng tải dữ liệu mẫu hoặc file giao dịch lên để ứng dụng hoạt động.")
