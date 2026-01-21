import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity # تم إضافتها للصفحة السادسة

# --- 1. إعدادات الصفحة واللغة ---
st.set_page_config(page_title="Rahma AI Platform", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'ar'

def toggle_lang():
    st.session_state.lang = 'en' if st.session_state.lang == 'ar' else 'ar'

translations = {
    'ar': {
        'nav': ["الرئيسية", "التحليلات والرسومات", "تنبؤ الأرباح (AI)", "تنبؤ أولوية الطلب (AI)", "مجموعات العملاء (RFM)", "نظام الترشيحات (AI)", "التقرير الختامي (Summary)"],
        'lang_btn': "English Version",
        'welcome': "مرحباً بك في منصة المتجر العالمي الضخم",
        'sub_title': "إعداد: رحمة عبد العزيز",
        'custom_draw': "🛠️ ارسم علاقتك الخاصة (الرسم الحر)",
        'predict_header': "💰 نموذج التنبؤ بالأرباح الهجين (Sequential)",
        'predict_btn': "احسب التنبؤ الآن",
        'res_profit': "الربح المتوقع",
        'res_loss': "الخسارة المتوقعة"
    },
    'en': {
        'nav': ["Home", "Analytics & Charts", "Profit Prediction (AI)", "Order Priority (AI)", "Customer Segments (RFM)", "Recommendations (AI)", "Executive Report"],
        'lang_btn': "النسخة العربية",
        'welcome': "Welcome to Global Super Store Platform",
        'sub_title': "Prepared by: Rahma Abd El-Aziz",
        'custom_draw': "🛠️ Custom Chart Builder",
        'predict_header': "💰 Hybrid Profit Prediction (Sequential)",
        'predict_btn': "Calculate Prediction",
        'res_profit': "Expected Profit",
        'res_loss': "Expected Loss"
    }
}
t = translations[st.session_state.lang]

# --- 2. محرك البيانات (تحميل الأنظمة) ---
@st.cache_resource
def load_rahma_system():
    df = pd.read_csv('Global_Superstore2_cleaned.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    
    # ميزات نظام الأرباح
    df['Profit_Transformed'] = np.sign(df['Profit']) * np.power(np.abs(df['Profit']), 1/3)
    df['Sales_to_Shipping'] = df['Sales'] / (df['Shipping Cost'] + 1)
    df['Discount_Impact'] = df['Sales'] * df['Discount']
    df['Unit_Price'] = df['Sales'] / df['Quantity']
    
    # ميزات نظام الأولوية
    df['Days_to_Ship'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Shipping_Cost_Ratio'] = df['Shipping Cost'] / (df['Sales'] + 1)
    df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 1)
    
    # --- تدريب نظام الأرباح ---
    cat_cols = ['Market', 'Category', 'Segment', 'Ship Mode']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(df[cat_cols])
    num_feats = ['Sales', 'Quantity', 'Discount', 'Shipping Cost', 'Sales_to_Shipping', 'Discount_Impact', 'Unit_Price']
    X = pd.concat([df[num_feats].reset_index(drop=True), pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(cat_cols))], axis=1)
    
    y_binary = (df['Profit_Transformed'] > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=100).fit(SMOTE(random_state=42).fit_resample(X, y_binary)[0], SMOTE(random_state=42).fit_resample(X, y_binary)[1])
    rf_pos = RandomForestRegressor(n_estimators=100).fit(X.loc[df[df['Profit_Transformed'] > 0].index], df.loc[df['Profit_Transformed'] > 0, 'Profit_Transformed'])
    rf_neg = RandomForestRegressor(n_estimators=100).fit(X.loc[df[df['Profit_Transformed'] <= 0].index], df.loc[df['Profit_Transformed'] <= 0, 'Profit_Transformed'])
    
    # --- تدريب نظام الأولوية الهجين ---
    p_feats = ['Sales', 'Quantity', 'Discount', 'Days_to_Ship', 'Shipping_Cost_Ratio', 'Profit_Margin', 'Ship Mode', 'Segment', 'Market']
    X_p = pd.get_dummies(df[p_feats])
    y_r = df['Order Priority'].map({'Critical': 1, 'High': 1, 'Medium': 0, 'Low': 0})
    rf_router = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_p, y_r)
    
    m_urg = df['Order Priority'].isin(['Critical', 'High'])
    rf_urg = RandomForestClassifier(n_estimators=100).fit(X_p[m_urg], df['Order Priority'][m_urg])
    
    m_norm = df['Order Priority'].isin(['Medium', 'Low'])
    rf_norm = RandomForestClassifier(n_estimators=100).fit(X_p[m_norm], df['Order Priority'][m_norm])
    
    return df, clf, rf_pos, rf_neg, encoder, num_feats, cat_cols, rf_router, rf_urg, rf_norm, X_p.columns

df, clf_model, rf_pos, rf_neg, encoder, num_feats, cat_cols, rf_router, rf_urg, rf_norm, p_cols = load_rahma_system()

# --- 3. السايدبار ---
with st.sidebar:
    st.button(t['lang_btn'], on_click=toggle_lang, key="lang_btn_sidebar")
    st.divider()
    choice = st.radio("القائمة" if st.session_state.lang == 'ar' else "Menu", t['nav'], key="rahma_nav")

# --- 4. محتوى الصفحات ---

# صفحة 1
if choice == t['nav'][0]:
    st.markdown(f'<div style="background-color: #0E1117; padding: 40px; border-radius: 15px; border: 2px solid #00d4ff; text-align: center;"><h1 style="color: white;">{t["welcome"]}</h1><h3 style="color: #00d4ff;">{t["sub_title"]}</h3></div>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1200")

# --- استبدال كود الصفحة الثانية (choice == t['nav'][1]) بهذا الكود المنظم ---

elif choice == t['nav'][1]:
    st.header(t['nav'][1])
    
    # تعريف أسماء الأزرار (Tabs) بناءً على اللغة
    if st.session_state.lang == 'ar':
        tab_names = ["📄 جدول البيانات", "📊 مبيعات الفئات", "📈 تحليل القطاعات", "👤 أهم العملاء", "🌍 خريطة الأرباح", "📅 اتجاهات النمو", "🛠️ الرسم الحر"]
    else:
        tab_names = ["📄 Data Table", "📊 Category Sales", "📈 Segment Analysis", "👤 Top Customers", "🌍 Profit Map", "📅 Growth Trends", "🛠️ Custom Builder"]
    
    # إنشاء الألسنة (Tabs)
    tabs = st.tabs(tab_names)
    
    # --- Tab 1: الجدول ---
    with tabs[0]:
        st.subheader(tab_names[0])
        st.dataframe(df.head(10), use_container_width=True)
    
    # --- Tab 2: مبيعات الفئات ---
    with tabs[1]:
        st.subheader(tab_names[1])
        cat_data = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=cat_data, x='Category', y='Sales', ax=ax1, palette='Blues_d')
        ax2 = ax1.twinx()
        sns.lineplot(data=cat_data, x='Category', y='Profit', ax=ax2, color='red', marker='o')
        st.pyplot(fig1)
    
    # --- Tab 3: تحليل القطاعات ---
    with tabs[2]:
        st.subheader(tab_names[2])
        fig2, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df, x='Category', y='Sales', hue='Segment', estimator=sum, errorbar=None)
        st.pyplot(fig2)
    
    # --- Tab 4: أهم العملاء والدول ---
    with tabs[3]:
        st.subheader(tab_names[3])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 Customers**")
            fig3, ax = plt.subplots()
            df.groupby('Customer Name')['Sales'].sum().sort_values().tail(10).plot(kind='barh', ax=ax, color='teal')
            st.pyplot(fig3)
        with c2:
            st.markdown("**Top 10 Countries**")
            fig4, ax = plt.subplots()
            top_c = df.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(10)
            sns.barplot(x=top_c.values, y=top_c.index, palette='viridis', ax=ax)
            st.pyplot(fig4)
    
    # --- Tab 5: خريطة الأرباح ---
    with tabs[4]:
        st.subheader(tab_names[4])
        st.plotly_chart(px.choropleth(df.groupby('Country')['Profit'].sum().reset_index(), 
                                      locations="Country", locationmode='country names', color="Profit"), 
                        use_container_width=True)
    
    # --- Tab 6: اتجاهات النمو ---
    with tabs[5]:
        st.subheader(tab_names[5])
        trend = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        st.plotly_chart(px.line(trend, x='Month', y='Sales', color=trend['Year'].astype(str)), 
                        use_container_width=True)
    
    # --- Tab 7: الرسم الحر ---
    with tabs[6]:
        st.subheader(t['custom_draw'])
        cx, cy = st.columns(2)
        x_sel = cx.selectbox("X-Axis", df.columns, key="x_free")
        y_sel = cy.selectbox("Y-Axis", ['Sales', 'Profit', 'Shipping Cost'], key="y_free")
        fig_f, ax_f = plt.subplots()
        sns.scatterplot(data=df, x=x_sel, y=y_sel, hue='Category', ax=ax_f)
        st.pyplot(fig_f)
# صفحة 3
elif choice == t['nav'][2]:
    st.header(t['predict_header'])
    with st.form("detailed_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1: s_val = st.number_input("Sales", value=200.0); m_val = st.selectbox("Market", ['Africa', 'APAC', 'Canada', 'EMEA', 'EU', 'LATAM', 'US'])
        with col2: q_val = st.number_input("Quantity", value=2); c_val = st.selectbox("Category", ['Furniture', 'Office Supplies', 'Technology'])
        with col3: d_val = st.number_input("Discount", value=0.0); sg_val = st.selectbox("Segment", ['Consumer', 'Corporate', 'Home Office'])
        with col4: sh_val = st.number_input("Shipping", value=20.0); sm_val = st.selectbox("Ship Mode", ['First Class', 'Same Day', 'Second Class', 'Standard Class'])
        if st.form_submit_button(t['predict_btn']):
            s_to_s = s_val / (sh_val + 1); d_imp = s_val * d_val; u_pri = s_val / q_val
            in_n = pd.DataFrame([[s_val, q_val, d_val, sh_val, s_to_s, d_imp, u_pri]], columns=num_feats)
            in_c = pd.DataFrame([[m_val, c_val, sg_val, sm_val]], columns=cat_cols)
            X_f = pd.concat([in_n, pd.DataFrame(encoder.transform(in_c), columns=encoder.get_feature_names_out(cat_cols))], axis=1)
            if clf_model.predict(X_f)[0] == 1: res = np.power(rf_pos.predict(X_f)[0], 3); st.success(f"### {t['res_profit']}: ${res:,.2f}")
            else: raw = rf_neg.predict(X_f)[0]; res = np.sign(raw) * np.power(np.abs(raw), 3); st.error(f"### {t['res_loss']}: ${res:,.2f}")

# صفحة 4
elif choice == t['nav'][3]:
    st.header("📦 Order Priority Prediction (Hybrid Model)")
    with st.form("priority_form"):
        c1, c2, c3 = st.columns(3)
        with c1: p_s = st.number_input("Sales", value=150.0); p_q = st.number_input("Quantity", value=2); p_m = st.selectbox("Market", ['Africa', 'APAC', 'Canada', 'EMEA', 'EU', 'LATAM', 'US'])
        with c2: p_d = st.number_input("Discount", value=0.0); p_days = st.slider("Days to Ship", 0, 7, 3); p_seg = st.selectbox("Segment", ['Consumer', 'Corporate', 'Home Office'])
        with c3: p_prof = st.number_input("Profit", value=20.0); p_sh = st.number_input("Shipping Cost", value=15.0); p_sm = st.selectbox("Ship Mode", ['First Class', 'Same Day', 'Second Class', 'Standard Class'])
        if st.form_submit_button("Predict Priority"):
            s_ratio = p_sh / (p_s + 1); p_margin = p_prof / (p_s + 1)
            in_p = pd.DataFrame([[p_s, p_q, p_d, p_days, s_ratio, p_margin, p_sm, p_seg, p_m]], columns=['Sales', 'Quantity', 'Discount', 'Days_to_Ship', 'Shipping_Cost_Ratio', 'Profit_Margin', 'Ship Mode', 'Segment', 'Market'])
            X_p_in = pd.get_dummies(in_p).reindex(columns=p_cols, fill_value=0)
            is_urgent = rf_router.predict(X_p_in)[0]
            final_p = rf_urg.predict(X_p_in)[0] if is_urgent == 1 else rf_norm.predict(X_p_in)[0]
            color = "#FF4B4B" if is_urgent == 1 else "#28A745"
            st.markdown(f'<div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 10px solid {color}; text-align: center;"><h2 style="color: white;">Priority: <span style="color: {color};">{final_p}</span></h2></div>', unsafe_allow_html=True)

# الصفحة 5: مجموعات العملاء (RFM Analysis)
elif choice == t['nav'][4]:
    st.header("👥 Customer Segmentation Analysis (RFM)")
    
    # 1. تجميع بيانات RFM
    snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)
    rfm_data = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (snapshot_date - x.max()).days,
        'Order ID': 'count',
        'Sales': 'sum'
    }).rename(columns={'Order Date': 'Recency', 'Order ID': 'Frequency', 'Sales': 'Monetary'})

    # 2. Scaling و KMeans (3 مجموعات)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 3. تعريف الأسماء والدمج
    cluster_names = {1: "Champions", 2: "Potential", 0: "At Risk"}
    rfm_data['Cluster_Name'] = rfm_data['Cluster'].map(cluster_names)
    df_merged = df.merge(rfm_data[['Cluster_Name']], on='Customer ID', how='left')

    # --- الرسومات الاحترافية للصفحة الخامسة ---
    
    # الرسم 1: Recency vs Monetary
    st.subheader("📍 Separation Check: Recency vs Monetary")
    fig_rfm, ax_rfm = plt.subplots(figsize=(12, 7))
    sns.scatterplot(data=rfm_data, x='Recency', y='Monetary', hue='Cluster_Name', palette='viridis', alpha=0.6, ax=ax_rfm)
    ax_rfm.set_yscale('log')
    ax_rfm.axvline(x=100, color='r', linestyle='--', label='Critical Threshold')
    ax_rfm.legend(); st.pyplot(fig_rfm)

    # الرسم 2: Dashboard المقارنات المالية والكميات
    st.subheader("📊 Comprehensive Cluster Comparison")
    cluster_comparison = df_merged.groupby('Cluster_Name')[['Sales', 'Profit', 'Quantity']].mean().reset_index()
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_s, ax_s = plt.subplots(); sns.barplot(x='Cluster_Name', y='Sales', data=cluster_comparison, palette='viridis', ax=ax_s)
        ax_s.set_title('Avg Sales ($)'); st.pyplot(fig_s)
    with c2:
        fig_p, ax_p = plt.subplots(); sns.barplot(x='Cluster_Name', y='Profit', data=cluster_comparison, palette='viridis', ax=ax_p)
        ax_p.set_title('Avg Profit ($)'); st.pyplot(fig_p)
    with c3:
        fig_q, ax_q = plt.subplots(); sns.barplot(x='Cluster_Name', y='Quantity', data=cluster_comparison, palette='viridis', ax=ax_q)
        ax_q.set_title('Avg Quantity'); st.pyplot(fig_q)

    # الرسم 3: المنتجات حسب المجموعة
    st.subheader("📦 Category Distribution per Cluster")
    fig_prod, ax_prod = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Cluster_Name', hue='Category', data=df_merged, palette='magma', ax=ax_prod)
    st.pyplot(fig_prod)

    # الرسم 4: توزيع الدول (Top 10)
    st.subheader("🌍 Top 10 Countries by Customer Group")
    top_10_countries = df_merged['Country'].value_counts().head(10).index
    df_top = df_merged[df_merged['Country'].isin(top_10_countries)]
    fig_geo, ax_geo = plt.subplots(figsize=(12, 8))
    sns.countplot(y='Country', hue='Cluster_Name', data=df_top, palette='viridis', ax=ax_geo)
    st.pyplot(fig_geo)

# الصفحة 6: نظام الترشيحات (Collaborative Filtering)
elif choice == t['nav'][5]:
    st.header("🎯 AI-Based Product Recommendations")
    
    # 1. إنشاء مصفوفة (العملاء مقابل المنتجات) باستخدام كودك الدقيق
    user_item_matrix = df.pivot_table(index='Customer Name', columns='Product Name', values='Quantity').fillna(0)

    # 2. حساب التشابه (Cosine Similarity)
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    # واجهة اختيار العميل
    target_user = st.selectbox("Select Customer Name", user_item_matrix.index)

    if st.button("Generate Recommendations"):
        if target_user not in user_item_matrix.index:
            st.error("العميل غير موجود")
        else:
            # أ. إيجاد أكثر العملاء تشابهاً (أفضل 50 شبيه)
            similar_users = user_sim_df[target_user].sort_values(ascending=False).iloc[1:51]
            
            # ب. جلب مشتريات هؤلاء الأشباه مع حساب الوزن (Weighted Score)
            sim_user_indices = similar_users.index
            similar_users_products = user_item_matrix.loc[sim_user_indices]
            weights = similar_users.values.reshape(-1, 1)
            
            # ج. حساب نقاط الترشيح
            recommendation_scores = (similar_users_products * weights).sum(axis=0)
            
            # د. استبعاد المنتجات المشتراة مسبقاً
            already_bought = user_item_matrix.loc[target_user]
            final_recs = recommendation_scores[already_bought == 0].sort_values(ascending=False).head(10)
            
            # عرض النتائج
            st.success(f"✅ Target User: {target_user}")
            st.write(f"👥 Similar users found with similarity up to: {similar_users.max():.2%}")
            
            st.markdown("### 🚀 Recommended Products for You:")
            for i, (name, score) in enumerate(final_recs.items(), 1):
                st.markdown(f"**{i}.** {name}")

# --- استبدال كود الصفحة السابعة (choice == t['nav'][6]) بهذا الكود الاحترافي ---

elif choice == t['nav'][6]:
    st.header("📊 " + ("التقرير الختامي وتحليل الرؤى الاستراتيجية" if st.session_state.lang == 'ar' else "Final Strategic Report & Insights"))
    
    # 1. تحليل المبيعات والنمو (بناءً على صورة Monthly Sales Trend)
    st.subheader("📈 " + ("أولاً: تحليل الاتجاهات والنمو" if st.session_state.lang == 'ar' else "I. Trend & Growth Analysis"))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # إنشاء Heatmap احترافي بدلاً من الدائرة
        st.markdown("**" + ("توزيع صافي الأرباح حسب السوق والفئة (Heatmap)" if st.session_state.lang == 'ar' else "Net Profit Distribution by Market & Category") + "**")
        pivot_profit = df.pivot_table(index='Market', columns='Category', values='Profit', aggfunc='sum')
        fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_profit, annot=True, fmt=".0f", cmap="RdYlGn", cbar_kws={'label': 'Profit ($)'}, ax=ax_heat)
        st.pyplot(fig_heat)
    
    with col2:
        if st.session_state.lang == 'ar':
            st.info("""
            **رؤية النمو:**
            * شهدت المبيعات قفزة نوعية بنسبة **90.31%** من بداية عام 2011 حتى نهاية 2014.
            * يظهر التحليل الموسمي ذروة مبيعات ثابتة في شهر **نوفمبر وديسمبر** من كل عام، مما يستدعي تكثيف المخزون في الربع الأخير.
            * السوق الأمريكي (US) يتصدر قائمة المبيعات والأرباح، بينما تحتاج أسواق مثل (Turkey & Nigeria) لإعادة هيكلة بسبب صافي الخسائر.
            """)
        else:
            st.info("""
            **Growth Insight:**
            * Sales recorded a significant **90.31%** increase from 2011 to 2014.
            * Seasonal trends show consistent peaks in **November and December**, suggesting inventory buildup for Q4.
            * The US market dominates both sales and profits, while markets like Turkey and Nigeria require restructuring due to net losses.
            """)

    st.divider()

    # 2. تحليل كفاءة التشغيل (بناءً على صور Shipping & Discounts)
    st.subheader("⚙️ " + ("ثانياً: الكفاءة التشغيلية والخصومات" if st.session_state.lang == 'ar' else "II. Operational Efficiency & Discounts"))
    
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.lang == 'ar':
            st.success("""
            **تأثير الخصومات (Profitability Impact):**
            * هناك علاقة عكسية حادة؛ الخصومات التي تتجاوز **20%** تؤدي مباشرة إلى تحول الربح لنتائج سلبية.
            * فئة **Technology** هي الأكثر مرونة مع تكاليف الشحن، حيث تحافظ على هوامش ربح عالية حتى مع ارتفاع تكلفة الشحن.
            """)
        else:
            st.success("""
            **Discount Impact:**
            * A sharp inverse correlation exists: discounts exceeding **20%** directly push profits into negative territory.
            * **Technology** remains the most resilient category, maintaining high margins even with rising shipping costs.
            """)
            
    with c2:
        if st.session_state.lang == 'ar':
            st.warning("""
            **تحليل الشحن:**
            * متوسط مدة الشحن العام هو **4 أيام**.
            * وضع شحن **Standard Class** يستهلك الوقت الأكبر (5 أيام)، مما قد يؤثر على تقييم العملاء (Customer Satisfaction) في الفئات الحساسة.
            """)
        else:
            st.warning("""
            **Shipping Analysis:**
            * The average shipping time is **4 days**.
            * **Standard Class** shipping takes the longest (5 days), which may impact customer satisfaction in time-sensitive segments.
            """)

    st.divider()

    # 3. تحليل العملاء (بناءً على صور RFM & Top Customers)
    st.subheader("👤 " + ("ثالثاً: تحليل محفظة العملاء" if st.session_state.lang == 'ar' else "III. Customer Portfolio Analysis"))
    
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        # عرض إحصائية سريعة من الـ RFM
        if st.session_state.lang == 'ar':
            st.write("**توزيع القوة الشرائية:**")
            st.write("- فئة **Consumer** تساهم بـ **51.48%** من الدخل.")
            st.write("- أفضل عميل: **Tom Ashbrook** بمشتريات تخطت 40 ألف دولار.")
        else:
            st.write("**Purchasing Power Distribution:**")
            st.write("- **Consumer** segment contributes **51.48%** of total revenue.")
            st.write("- Top Customer: **Tom Ashbrook** with over $40k in purchases.")

    with cc2:
        if st.session_state.lang == 'ar':
            st.markdown("""
            **التوصيات الاستراتيجية النهائية:**
            1. **الولاء:** تفعيل برنامج مكافآت مخصص لعملاء (Champions) الذين يمثلون 50% من قاعدة العملاء لضمان الاستمرارية.
            2. **التحوط:** وقف الخصومات التلقائية في أسواق الخسائر (مثل تركيا) واستبدالها بعروض حزم (Bundling).
            3. **اللوجستيات:** تحويل جزء من شحنات Standard Class إلى Second Class في فئة الأثاث لتقليل زمن الانتظار وزيادة تدوير رأس المال.
            """)
        else:
            st.markdown("""
            **Final Strategic Recommendations:**
            1. **Retention:** Launch a loyalty program for 'Champions' (50% of our base) to secure long-term revenue.
            2. **Mitigation:** Halt automatic discounts in loss-making markets (e.g., Turkey) and implement product bundling instead.
            3. **Logistics:** Shift Furniture shipments from Standard to Second Class to reduce lead times and improve cash flow.
            """)

    st.markdown(f"<br><hr><center><small>{('إعداد: رحمة عبد العزيز' if st.session_state.lang == 'ar' else 'Prepared by: Rahma Abd El-Aziz')}</small></center>", unsafe_allow_html=True)