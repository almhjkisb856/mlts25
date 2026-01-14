import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. تحميل البيانات
# تأكد من وضع ملف data_all.xltx في نفس مجلد المشروع
file_path = 'data_all.xltx'
df = pd.read_excel(file_path)


def clean_and_prepare_data(data):
    # أ. تنظيف البيانات الأساسية
    # إزالة المسافات الزائدة من النصوص وتوحيد حالة الأحرف
    text_columns = ['types', 'Country', 'solution', 'opportunity_stage']
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # ب. معالجة التواريخ
    data['activity_date'] = pd.to_datetime(data['activity_date'])

    # ج. ترتيب البيانات حسب العميل والزمن لبناء الرحلة
    data = data.sort_values(by=['who_id', 'activity_date'])

    # د. استخراج النتيجة (Win/Loss)
    # نعتبر الرحلة ناجحة إذا انتهت بـ Won
    data['is_won'] = data['opportunity_stage'].apply(lambda x: 1 if x == 'Won' else 0)

    return data


# تنفيذ التنظيف
df_cleaned = clean_and_prepare_data(df)


# 2. تجميع الحسابات حسب الدولة والحل
def get_top_paths(data):
    # تجميع الـ types في قائمة واحدة (مسار) لكل عميل
    paths = data.groupby(['who_id', 'Country', 'solution'])['types'].apply(list).reset_index()

    # تحويل القائمة إلى نص لسهولة الحساب (مثال: "Email -> Call")
    paths['path_string'] = paths['types'].apply(lambda x: " -> ".join(x))

    # إيجاد أعلى 5 مسارات لكل (Country + Solution)
    top_five_results = paths.groupby(['Country', 'solution'])['path_string'].value_counts().groupby(level=[0, 1]).head(
        5)

    return paths, top_five_results


# استخراج المسارات
df_paths, top_5_paths = get_top_paths(df_cleaned)

# طباعة النتائج الأولية في PyCharm
print("--- تنظيف البيانات تم بنجاح ---")
print(df_cleaned.head())
print("\n--- أعلى 5 مسارات لكل دولة وحل  ---")
print(top_5_paths)


# تجهيز البيانات لشجرة القرار
def build_decision_tree(data):
    # معالجة القيم المفقودة (Cleaning) [cite: 3]
    dt_data = data.dropna(subset=['Country', 'solution', 'types']).copy()

    # تحويل النصوص لأرقام (Encoding)
    le_country = LabelEncoder()
    le_sol = LabelEncoder()
    le_types = LabelEncoder()

    dt_data['country_enc'] = le_country.fit_transform(dt_data['Country'])
    dt_data['solution_enc'] = le_sol.fit_transform(dt_data['solution'])
    dt_data['type_enc'] = le_types.fit_transform(dt_data['types'])

    # تحديد الهدف: هل انتهت المسار بـ Win؟ [cite: 11]
    X = dt_data[['country_enc', 'solution_enc', 'type_enc']]
    y = dt_data['is_won']

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)

    return model, le_country, le_sol, le_types


model, le_country, le_sol, le_types = build_decision_tree(df_cleaned)
print("\n--- أهمية الخصائص (Feature Importance) ---")
print(dict(zip(['Country', 'Solution', 'Action Type'], model.feature_importances_)))


def get_system_output(country, solution, last_touch_weight=0.1):
    # تصفية البيانات حسب الدولة والحل
    mask = (df_cleaned['Country'] == country) & (df_cleaned['solution'] == solution)
    filtered = df_cleaned[mask]

    # 1. أفضل 4 إجراءات حسب الدولة
    top_country = df_cleaned[df_cleaned['Country'] == country]['types'].value_counts().head(4)

    # 2. أفضل 4 إجراءات حسب الحل
    top_solution = df_cleaned[df_cleaned['solution'] == solution]['types'].value_counts().head(4)

    # 3. أفضل 4 إجراءات حسب الاثنين معاً
    top_combined = filtered['types'].value_counts().head(4)

    # 4. إعادة حساب الأوزان (Dynamic Weight Adjustment)
    # نفترض أن التكرار الحالي هو الـ Base Weight
    base_counts = top_combined.to_dict()
    new_weights = {k: v * (1 - last_touch_weight) for k, v in base_counts.items()}[cite: 10]

    return top_country, top_solution, top_combined, new_weights


# تجربة تشغيل النظام
t_country, t_sol, t_comb, updated_w = get_system_output('US', 'MRS')

print("\n--- Top 4 Actions by Country ---")
print(t_country)
print("\n--- Top 4 Actions by Solution ---")
print(t_sol)
print("\n--- Top 4 Actions Combined ---")
print(t_comb)
print("\n--- Recalculated Weights (New Action Added) ---")
print(updated_w)

