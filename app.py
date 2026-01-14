import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="نظام تحليل رحلة العميل", layout="wide")
st.title("🚀 نظام تحليل رحلة العميل الذكي")


# 1. تنظيف وتحميل البيانات
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('data_all.xltx')
        # ترتيب البيانات زمنياً لبناء المسارات بشكل صحيح
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        df = df.sort_values(by=['who_id', 'activity_date'])

        # تنظيف النصوص
        for col in ['types', 'Country', 'solution', 'opportunity_stage']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # تحديد الفوز (Win) بناءً على مرحلة الفرصة
        if 'opportunity_stage' in df.columns:
            df['is_won'] = df['opportunity_stage'].apply(lambda x: 1 if str(x).lower() == 'won' else 0)
        else:
            df['is_won'] = 0

        return df
    except Exception as e:
        st.error(f"خطأ في تحميل الملف: {e}")
        return None


df = load_data()

if df is not None:
    # --- القائمة الجانبية لإدخال معلومات الحساب  ---
    st.sidebar.header("📝 بيانات الحساب الجديد")
    countries = sorted(df['Country'].unique()) if 'Country' in df.columns else []
    solutions = sorted(df['solution'].unique()) if 'solution' in df.columns else []

    country_in = st.sidebar.selectbox("اختر الدولة", options=countries)
    solution_in = st.sidebar.selectbox("اختر نوع الحل", options=solutions)

    # --- 2. إيجاد أفضل 5 مسارات لكل فئة  ---
    st.header("🛣️ تحليل أفضل 5 مسارات (Top 5 Paths)")


    def get_paths(data, country, solution):
        filtered = data[(data['Country'] == country) & (data['solution'] == solution)]
        if filtered.empty:
            return None

        # تجميع الأنشطة لكل عميل كمسار متسلسل
        paths = filtered.groupby('who_id')['types'].apply(lambda x: " ➔ ".join(x)).reset_index()
        top_paths = paths['types'].value_counts().head(5).reset_index()
        top_paths.columns = ['المسار (Path)', 'التكرار (Frequency)']
        return top_paths


    top_5_df = get_paths(df, country_in, solution_in)
    if top_5_df is not None:
        st.table(top_5_df)
    else:
        st.warning("لا توجد مسارات مسجلة لهذه الفئة حالياً.")

    # --- 3. عرض المخرجات الأربعة المطلوبة  ---
    st.divider()
    st.header("🎯 التوصيات (Top 4 Actions)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 حسب الدولة")
        st.write(df[df['Country'] == country_in]['types'].value_counts().head(4))  # [cite: 7]

    with col2:
        st.subheader("💡 حسب الحل")
        st.write(df[df['solution'] == solution_in]['types'].value_counts().head(4))  # [cite: 8]

    with col3:
        st.subheader("🌍 الدولة والحل معاً")
        combined_df = df[(df['Country'] == country_in) & (df['solution'] == solution_in)]
        current_top_4 = combined_df['types'].value_counts().head(4)  # [cite: 9]
        st.write(current_top_4)

    # --- 4. خوارزمية ضبط الوزن الديناميكي عند إضافة إجراء  ---
    st.divider()
    st.header("⚖️ ضبط الوزن الديناميكي")

    all_types = sorted(df['types'].unique())
    action_to_add = st.selectbox("أضف إجراءً جديداً:", options=all_types)
    is_first_touch = st.radio("هل هذه اللمسة الأولى؟", ["نعم", "لا"])

    if is_first_touch == "نعم":
        st.info("تم تطبيق الوزن الأولي للإجراء الجديد.")
    else:
        last_touch_val = st.slider("حدد وزن اللمسة الأخيرة (Last Touch Weight):", 0.0, 1.0, 0.2)
        # تطبيق المعادلة المطلوبة: New Weight = Base Weight * (1 - Last Touch Weight)
        if not current_top_4.empty:
            new_weights = current_top_4 * (1 - last_touch_val)
            st.write("📊 الأوزان الجديدة المعاد حسابها:")
            st.bar_chart(new_weights)

    # --- 5. شجرة القرار وتحديد الرحلة الأفضل ---
    st.divider()
    st.header("🧠 تحليل شجرة القرار")

    try:
        le = LabelEncoder()
        df_dt = df.dropna(subset=['Country', 'solution', 'types']).copy()
        df_dt['c_enc'] = le.fit_transform(df_dt['Country'])
        df_dt['s_enc'] = le.fit_transform(df_dt['solution'])
        df_dt['t_enc'] = le.fit_transform(df_dt['types'])

        # استخدام DT لتحديد أهمية الميزات
        dt = DecisionTreeClassifier(max_depth=4)
        dt.fit(df_dt[['c_enc', 's_enc', 't_enc']], df_dt['is_won'])

        importances = pd.Series(dt.feature_importances_, index=['الدولة', 'الحل', 'نوع الإجراء'])
        st.write("أهمية الميزات في تحديد النجاح:")
        st.bar_chart(importances)

        # استخراج أقصر وأفضل رحلة تؤدي للفوز
        st.subheader("🏆 المسار الأفضل والأقصر للنجاح")
        won_journeys = df[df['is_won'] == 1].groupby('who_id')['types'].apply(list).reset_index()
        if not won_journeys.empty:
            won_journeys['length'] = won_journeys['types'].apply(len)
            shortest_win = won_journeys.sort_values(by='length').iloc[0]['types']  # [cite: 2]
            st.success(f"الرحلة الأقصر المقترحة للفوز: {' ➔ '.join(shortest_win)}")
    except Exception as e:
        st.warning(f"تعذر عرض شجرة القرار: {e}")

else:
    st.error("يرجى التأكد من وجود الملف 'data_all.xltx' وتشغيل التطبيق عبر Terminal.")



    # streamlit run app.py