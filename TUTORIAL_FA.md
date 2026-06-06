# آموزش کامل کتابخانه DataCleaner

**DataCleaner** یک کتابخانه خودکار براي تميزسازي، استانداردسازي و آماده‌سازي داده براي پروژه‌هاي يادگيري ماشين است.

## فهرست مطالب

1. [نصب](#نصب)
2. [شروع سريع](#شروع-سريع)
3. [بارگذاري داده](#بارگذاري-داده)
4. [پيكربندي](#پيكربندي)
5. [دستور prepare() و همه پارامترها](#دستور-prepare-و-همه-پارامترها)
6. [مديريت مقادير گمشده](#مديريت-مقادير-گمشده)
7. [مديريت داده‌هاي پرت (Outlier)](#مديريت-دادههاي-پرت-outlier)
8. [حذف ستون‌هاي بي‌فايده](#حذف-ستونهاي-بي‌فايده)
9. [انتخاب خودكار مقياس‌كننده (Scaler)](#انتخاب-خودكار-مقياسكننده-scaler)
10. [رمزگذاري خودكار (Encoding)](#رمزگذاري-خودكار-encoding)
11. [مهندسي ويژگي (Feature Engineering)](#مهندسي-ويژگي-feature-engineering)
12. [استخراج ويژگي از تاريخ](#استخراج-ويژگي-از-تاريخ)
13. [شاخص‌هاي مقادير گمشده (Missing Indicators)](#شاخصهاي-مقادير-گمشده-missing-indicators)
14. [انتخاب ويژگي (Feature Selection)](#انتخاب-ويژگي-feature-selection)
15. [رمزگذاري و مقياس‌گذاري سفارشي](#رمزگذاري-و-مقياسگذاري-سفارشي)
16. [مديريت داده‌هاي نامتوازن (SMOTE)](#مديريت-دادههاي-نامتوازن-smote)
17. [ذخيره و بازيابي Pipeline](#ذخيره-و-بازيابي-pipeline)
18. [پيش‌بيني روي داده‌هاي جديد (Inference)](#پيشبيني-روي-دادههاي-جديد-inference)
19. [گزارش پروفايل داده (Data Profiling)](#گزارش-پروفايل-داده-data-profiling)
20. [اعتبارسنجي شمّا (Schema Validation)](#اعتبارسنجي-شما-schema-validation)
21. [تبديل خودكار نوع داده](#تبديل-خودكار-نوع-داده)
22. [حذف سطرهاي تكراري](#حذف-سطرهاي-تكراري)
23. [خروجي گرفتن از داده‌هاي تميز](#خروجي-گرفتن-از-دادههاي-تميز)
24. [آزمون‌هاي آماري (Stats Module)](#آزمونهاي-آماري-stats-module)
25. [م‌اژول تصويرسازي (Plotting)](#ماژول-تصويرسازي-plotting)
26. [مثال كامل سرتاسري](#مثال-كامل-سرتاسري)
27. [نكات و رويه‌هاي پيشنهادي](#نكات-وريـههاي-پيشنهادي)

---

## نصب

### نصب از PyPI

```bash
pip install clean-data-ml
```

### نصب با قابليت‌هاي اضافي

```bash
pip install clean-data-ml[plot]       # و matplotlib, seaborn براي تصويرسازي
pip install clean-data-ml[imbalance]   # SMOTE براي
pip install clean-data-ml[all]         # همه قابليت‌ها
```

### نصب از سورس

```bash
git clone https://github.com/MohammadvHossein/clean-data-ml.git
cd clean_data_ml
pip install -e .
pip install -e .[all]
```

---

## شروع سريع

```python
from clean_data_ml import DataCleaner
from sklearn.svm import SVC

dc = DataCleaner()
dc.load("data.csv")
dc.set_target("purchased")
dc.drop_columns(["ID", "timestamp"])

X_train, X_test, y_train, y_test = dc.prepare(test_size=0.2)

model = SVC()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

---

## بارگذاري داده

### از فايل CSV يا Excel

```python
dc = DataCleaner()
dc.load("data.csv")              # فايل CSV
dc.load("data.xlsx")             # فايل Excel
```

### از DataFrame موجود در حافظه

```python
import pandas as pd
df = pd.DataFrame({
    "age": [25, 30, None, 40],
    "salary": [50000, 60000, 80000, 90000],
    "city": ["Tehran", "Shiraz", "Isfahan", "Tehran"],
    "purchased": [1, 0, 1, 0],
})
dc.load_df(df)
```

---

## پيكربندي

### تعيين ستون هدف

```python
dc.set_target("purchased")
```

### حذف ستون‌هاي ناخواسته

متد `drop_columns()` امن است و مي‌توان چند بار آن را صدا زد — ستون‌ها جمع مي‌شوند، جايگزين نمي‌شوند.

```python
dc.drop_columns(["ID", "timestamp"])
dc.drop_columns(["unnecessary_feature"])   # ستون اضافه مي‌شود، جايگزين نمي‌شود
```

---

## دستور prepare() و همه پارامترها

متد `prepare()` هسته اصلي كتابخانه است. همه مراحل تميزسازي را به ترتيب اجرا مي‌كند:

1. حذف ستون‌هاي مشخص‌شده
2. تشخيص خودكار نوع مسأله (طبقه‌بندي / رگرسيون)
3. استخراج ويژگي از تاريخ (اگر فعال باشد)
4. حذف خودكار ستون‌هاي بي‌فايده
5. تشخيص نوع ستون‌ها (عددي، اسمي، دو‌مقداري)
6. مديريت مقادير گمشده
7. اضافه كردن شاخص‌هاي مقادير گمشده (اگر فعال باشد)
8. مديريت داده‌هاي پرت
9. مهندسي ويژگي (اگر فعال باشد)
10. رمزگذاري
11. مقياس‌گذاري
12. انتخاب ويژگي (اگر فعال باشد)
13. تقسيم داده به Train/Test
14. SMOTE (اگر فعال باشد)

```python
X_train, X_test, y_train, y_test = dc.prepare(
    test_size=0.2,                # نسبت داده‌هاي تست
    val_size=None,                # اگر مقدار بگيرد، يك مجموعه اعتبارسنجي هم مي‌سازد
    handle_nulls=True,            # مديريت خودكار مقادير گمشده
    auto_scale=True,              # انتخاب خودكار مقياس‌كننده بهينه
    auto_encode=True,             # رمزگذاري خودكار ستون‌ها
    null_drop_ratio=None,         # override آستانه حذف مقادير گمشده
    auto_drop_useless=True,       # حذف خودكار ستون‌هاي بي‌فايده
    handle_outliers=None,         # "clip" يا "remove" براي داده‌هاي پرت
    feature_engineering=False,    # ايجاد ويژگي‌هاي چندجمله‌اي
    handle_imbalance=False,       # SMOTE براي داده‌هاي نامتوازن
    n_jobs=1,                     # تعداد پردازش‌هاي موازي
    extract_date_features=False,  # استخراج ويژگي از ستون‌هاي تاريخي
    add_missing_indicators=False, # اضافه كردن شاخص گمشدگي
    feature_selection=None,       # "auto" يا آستانه عددي
    custom_encoders=None,         # رمزگذارهاي سفارشي {نام_ستون: رمزگذار}
    custom_scalers=None,          # مقياس‌كننده‌هاي سفارشي {نام_ستون: مقياس‌كننده}
)
```

### شكل برگشتي

```python
# حالت عادي
X_train, X_test, y_train, y_test = dc.prepare(test_size=0.2)

# با مجموعه اعتبارسنجي
X_train, X_val, X_test, y_train, y_val, y_test = dc.prepare(val_size=0.15)
```

---

## مديريت مقادير گمشده

### آستانه پويا

آستانه حذف سطرهاي داراي مقدار گمشده به اندازه ديتاست وابسته است:

| تعداد سطرها | آستانه حذف | رفتار |
|------------|-----------|--------|
| 100 سطر | 25% | محافظه‌كارانه — ترجيح KNN |
| 1,000 سطر | 5% | متعادل |
| 10,000+ سطر | 1% | حذف جسورانه‌تر |

مي‌توانيد آستانه را بازنويسي كنيد:

```python
X_train, X_test, y_train, y_test = dc.prepare(null_drop_ratio=0.1)
```

### روش كار

- **ستون‌هاي عددي با مقادير گمشده زياد**: `KNNImputer` با 5 همسايه
- **ستون‌هاي اسمي با مقادير گمشده زياد**: پر كردن با مد (مقدار بيشتر)
- **ستون‌هايي با مقادير گمشده كم**: حذف سطرهاي حاوي مقدار گمشده

### مقدار گمشده در هدف

اگر ستون هدف كمتر از 5% مقدار گمشده داشته باشد، آن سطرها حذف مي‌شوند. اگر بيشتر از 5% باشد، خطا مي‌دهد:

```python
# اين خطا مي‌دهد اگر هدف بيش از 5% مقدار گمشده داشته باشد
# ValueError: Target column 'price' has 12.5% null values. Please clean it manually.
```

---

## مديريت داده‌هاي پرت (Outlier)

از روش IQR (ميان چاركي) استفاده مي‌كند:

- **كران پايين**: Q1 - 1.5 x IQR
- **كران بالا**: Q3 + 1.5 x IQR

دو حالت:

```python
# حالت 1: برش (clip) — مقادير پرت را به كران‌ها محدود مي‌كند، تعداد سطرها حفظ مي‌شود
X_train, X_test, y_train, y_test = dc.prepare(handle_outliers="clip")

# حالت 2: حذف (remove) — سطرهاي داراي داده پرت حذف مي‌شوند
X_train, X_test, y_train, y_test = dc.prepare(handle_outliers="remove")

# غيرفعال
X_train, X_test, y_train, y_test = dc.prepare(handle_outliers=None)
```

داده‌هاي پرت بعد از مديريت مقادير گمشده و قبل از مهندسي ويژگي پردازش مي‌شوند.

---

## حذف ستون‌هاي بي‌فايده

كتابخانه به طور خودكار ستون‌هاي زير را تشخيص و حذف مي‌كند:

- **ستون‌هاي با واريانس صفر**: فقط يك مقدار منحصربه‌فرد دارند
- **ستون‌هاي با كارديـناليتي بالا**: ستون‌هاي غيرعددي كه بيش از 90% مقادير آنها منحصربه‌فرد است (مثل متن‌هاي آزاد)

```python
# غيرفعال
X_train, X_test, y_train, y_test = dc.prepare(auto_drop_useless=False)
```

ستون‌هاي حذف‌شده در `dc.pipeline.dropped_useless_cols` ذخيره مي‌شوند.

---

## انتخاب خودكار مقياس‌كننده (Scaler)

براي هر ستون عددي، كتابخانه آزمايش‌هاي زير را انجام مي‌دهد:

1. **نرمال بودن** — آزمون Shapiro-Wilk (p > 0.05 يعني نرمال)
2. **داده پرت** — روش IQR
3. **كران‌ها** — حداقل >= 0 و حداكثر <= 1
4. **تخلخل (Sparsity)** — بيش از 40% صفر

سپس بهترين مقياس‌كننده را انتخاب مي‌كند:

| شرط | مقياس‌كننده |
|-----|------------|
| نرمال + بدون داده پرت | `StandardScaler` |
| داراي داده پرت | `RobustScaler` |
| كراندار [0, 1] | `MinMaxScaler` |
| خلوت (Sparse) | `MaxAbsScaler` |
| پيش‌فرض | `StandardScaler` |

**نكته**: مدل‌هاي درختي (Random Forest, XGBoost, LightGBM و...) به Scaling نياز ندارند. مي‌توانيد Scaling را با `auto_scale=False` غيرفعال كنيد.

```python
X_train, X_test, y_train, y_test = dc.prepare(auto_scale=False)
```

مقياس‌كننده‌ها در `dc.pipeline.scalers` قابل مشاهده هستند:

```python
# مشاهده مقياس‌كننده انتخاب‌شده براي هر ستون
for col, scaler in dc.pipeline.scalers.items():
    print(f"{col}: {type(scaler).__name__}")
```

---

## رمزگذاري خودكار (Encoding)

كتابخانه به طور خودكار تشخيص مي‌دهد:

- **ستون‌هاي دو مقداري (Binary)** — فقط 2 مقدار منحصربه‌فرد → `LabelEncoder`
- **ستون‌هاي اسمي (Categorical)** — بيش از 2 مقدار → `OneHotEncoder`

```python
# غيرفعال كردن رمزگذاري خودكار
X_train, X_test, y_train, y_test = dc.prepare(auto_encode=False)
```

اگر رمزگذاري خودكار غيرفعال شود، همه ستون‌ها با `LabelEncoder` كدگذاري مي‌شوند.

---

## مهندسي ويژگي (Feature Engineering)

با فعال كردن `feature_engineering=True`، ويژگي‌هاي چندجمله‌اي (درجه 2) براي ستون‌هاي عددي با بيش از 2 مقدار منحصربه‌فرد ساخته مي‌شود:

```python
X_train, X_test, y_train, y_test = dc.prepare(feature_engineering=True)
```

اين كار ويژگي‌هاي تعاملي (ضرب دو ستون) و مربع (توان دوم هر ستون) ايجاد مي‌كند. اين ويژگي‌ها در `transform()` نيز به طور خودكار ساخته مي‌شوند.

---

## استخراج ويژگي از تاريخ

با فعال كردن `extract_date_features=True`، ستون‌هاي تاريخي به مؤلفه‌هاي عددي زير تبديل مي‌شوند:

- `{col}_year` — سال
- `{col}_month` — ماه
- `{col}_day` — روز
- `{col}_dayofweek` — روز هفته (0=دوشنبه)
- `{col}_weekend` — آيا آخر هفته است؟ (0 يا 1)

ستون اصلي تاريخ پس از استخراج حذف مي‌شود.

```python
X_train, X_test, y_train, y_test = dc.prepare(extract_date_features=True)
```

اين مرحله **پيش از** حذف ستون‌هاي بي‌فايده انجام مي‌شود تا ستون‌هاي جديد حذف نشوند، و **پيش از** تشخيص نوع داده تا ستون‌هاي عددي جديد به درستي شناسايي شوند.

---

## شاخص‌هاي مقادير گمشده (Missing Indicators)

با فعال كردن `add_missing_indicators=True`، براي هر ستوني كه با KNN مقدارگيري مي‌شود، يك ستون باينري `{col}_missing` اضافه مي‌شود كه مشخص مي‌كند كدام سطرها در اصل مقدار گمشده داشتند:

```python
X_train, X_test, y_train, y_test = dc.prepare(add_missing_indicators=True)
```

اين كار به مدل اجازه مي‌دهد الگوهاي گمشدگي را ياد بگيرد.

---

## انتخاب ويژگي (Feature Selection)

پس از همه تبديل‌ها، اطلاعات متقابل (Mutual Information) بين هر ويژگي و هدف محاسبه مي‌شود. ويژگي‌هاي ضعيف حذف مي‌شوند:

```python
# حالت خودكار — ويژگي‌هاي زير ميانه اطلاعات متقابل حذف مي‌شوند
X_train, X_test, y_train, y_test = dc.prepare(feature_selection="auto")

# با آستانه مشخص — ويژگي‌هاي با MI كمتر از 0.01 حذف مي‌شوند
X_train, X_test, y_train, y_test = dc.prepare(feature_selection=0.01)

# غيرفعال
X_train, X_test, y_train, y_test = dc.prepare(feature_selection=None)
```

اطلاعات تشخيص:

```python
print("آستانه انتخاب ويژگي:", dc.pipeline.feature_selection_threshold)
print("ستون‌هاي حذف‌شده:", dc.pipeline.feature_selection_removed)
```

---

## رمزگذاري و مقياس‌گذاري سفارشي

مي‌توانيد رمزگذارها و مقياس‌كننده‌هاي sklearn خود را به `prepare()` بدهيد:

### رمزگذار سفارشي

```python
from sklearn.preprocessing import OrdinalEncoder

X_train, X_test, y_train, y_test = dc.prepare(
    custom_encoders={"city": OrdinalEncoder()},
)
```

### مقياس‌كننده سفارشي

```python
from sklearn.preprocessing import KBinsDiscretizer

X_train, X_test, y_train, y_test = dc.prepare(
    custom_scalers={"salary": KBinsDiscretizer(n_bins=5, encode="ordinal")},
)
```

### تركيب هر دو

```python
X_train, X_test, y_train, y_test = dc.prepare(
    custom_encoders={"city": OrdinalEncoder(), "gender": LabelEncoder()},
    custom_scalers={"age": RobustScaler(), "salary": StandardScaler()},
)
```

اين تبديل‌گرها در `dc.pipeline.custom_encoders` و `dc.pipeline.custom_scalers` ذخيره مي‌شوند و در `transform()` براي داده‌هاي جديد نيز اعمال مي‌شوند.

---

## مديريت داده‌هاي نامتوازن (SMOTE)

براي مسائل طبقه‌بندي با داده‌هاي نامتوازن:

```python
X_train, X_test, y_train, y_test = dc.prepare(handle_imbalance=True)
```

SMOTE پس از تقسيم داده به Train/Test روي مجموعه Train اعمال مي‌شود. نياز به نصب `imbalanced-learn` دارد:

```bash
pip install imbalanced-learn
```

---

## ذخيره و بازيابي Pipeline

### ذخيره

```python
dc.save_pipeline("my_pipeline.pkl")
```

Pipeline همه چيز را ذخيره مي‌كند: مقياس‌كننده‌ها، رمزگذارها، imputer، ستون‌ها، آستانه‌ها و...

### بازيابي

```python
dc = DataCleaner.load_pipeline("my_pipeline.pkl")
```

### متدهاي كمكي

```python
# دريافت خود pipeline
pipeline = dc.get_pipeline()

# ذخيره و بازيابي مستقيم توسط CleanPipeline
pipeline.save("pipe.pkl")
loaded = CleanPipeline.load("pipe.pkl")
```

---

## پيش‌بيني روي داده‌هاي جديد (Inference)

دو راه داريد:

### روش 1: استفاده از DataCleaner

```python
dc = DataCleaner.load_pipeline("my_pipeline.pkl")
new_data = pd.DataFrame({
    "age": [28, 42],
    "salary": [65000, 95000],
    "city": ["Tehran", "Isfahan"],
    "gender": ["F", "M"],
})
processed = dc.transform(new_data)
predictions = model.predict(processed)
```

### روش 2: استفاده مستقيم از CleanPipeline

```python
pipeline = CleanPipeline.load("my_pipeline.pkl")
processed = pipeline.transform(new_data)
```

**نكته مهم**: اگر ستوني در داده جديد وجود نداشته باشد، با 0 پر مي‌شود. ترتيب ستون‌هاي خروجي دقيقاً مطابق مجموعه آموزش است.

---

## گزارش پروفايل داده (Data Profiling)

اين متد يك گزارش HTML تعاملي از داده توليد مي‌كند:

```python
dc.summary()         # خروجي ديكشنري
dc.profile_report()  # خروجي HTML (نمايش در مرورگر)
dc.profile_report("report.html")  # ذخيره در فايل
```

گزارش شامل:
- كارتهاي نماي كلي (تعداد سطر، ستون، مقادير گمشده، تكراري، حافظه)
- هشدارهاي كيفيت داده
- جدول جزئيات ستون‌ها (نوع، گمشده‌ها، چندك‌ها، چولگي، داده‌هاي پرت)
- هيستوگرام توزيع ستون‌هاي عددي (نياز به matplotlib)
- نقشه حرارتي همبستگي
- نمودار ميلي براي ستون‌هاي اسمي

*براي تصاوير نياز به `matplotlib` دارد، اما بدون آن هم HTML با اطلاعات متني كار مي‌كند.*

---

## اعتبارسنجي شمّا (Schema Validation)

بررسي مي‌كند كه ستون‌هاي مورد نياز وجود دارند و نوع داده آنها صحيح است:

```python
issues = dc.validate_schema(
    expected_schema={
        "age": "numeric",
        "date": "datetime",
        "name": "string",
    },
    required_cols=["age", "salary", "city"],
)

if issues:
    print("مشكلات يافت شده:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("اعتبارسنجي با موفقيت گذشت")
```

---

## تبديل خودكار نوع داده

با `auto_fix_dtypes()`، ستون‌هاي object به طور خودكار به datetime يا عددي تبديل مي‌شوند:

```python
fixes = dc.auto_fix_dtypes()
# نتيجه: ["date_col: object -> datetime", "price_col: object -> numeric"]
```

روش كار:
1. اول امتحان مي‌كند آيا مي‌توان به datetime تبديل كرد (بيش از 70% موفق)
2. اگر نه، امتحان مي‌كند آيا مي‌توان به عدد تبديل كرد (پشتيباني از جداكننده هزارگان)

---

## حذف سطرهاي تكراري

```python
# حذف سطرهاي تكراري بر اساس همه ستون‌ها
dc.drop_duplicates()

# حذف بر اساس ستون‌هاي مشخص
dc.drop_duplicates(subset=["age", "city"])

# نگه‌داشتن آخرين تكرار
dc.drop_duplicates(keep="last")
```

---

## خروجي گرفتن از داده‌هاي تميز

```python
# فقط ويژگي‌ها
dc.export_cleaned("cleaned_data.csv")

# همراه با ستون هدف
dc.export_cleaned("cleaned_with_target.xlsx", include_target=True)
```

فرمت‌هاي پشتيباني‌شده: CSV و Excel (.xlsx)

---

## آزمون‌هاي آماري (Stats Module)

ماژول `clean_data_ml.stats` بيش از 20 تابع آزمون آماري و يك كلاس `StatisticalTestSuite` ارائه مي‌دهد.

### توابع مستقل

```python
from clean_data_ml import stats
import pandas as pd

# آزمون نرمال بودن
result = stats.normality_test(series, method="shapiro")
# خروجي: {"statistic": ..., "p_value": ..., "is_normal": True/False}

# آزمون همبستگي
result = stats.correlation_test(x, y, method="pearson")

# آزمون كايدو (Chi-square)
result = stats.chi_square_test(col_a, col_b)

# آزمون ANOVA يك‌طرفه
result = stats.anova_one_way(group1, group2, group3)

# آزمون t تك نمونه‌اي
result = stats.t_test_one_sample(series, pop_mean=0)

# آزمون t مستقل
result = stats.t_test_independent(a, b)

# آزمون t جفت‌شده
result = stats.t_test_paired(before, after)

# آزمون Z تك نمونه‌اي
result = stats.z_test_one_sample(series, pop_mean=100)

# آزمون Z دو نمونه‌اي
result = stats.z_test_two_sample(a, b)

# آزمون Z نسبت
result = stats.z_test_proportion(successes=45, n=100, p_pop=0.5)

# آزمون Z دو نسبتي
result = stats.z_test_two_proportion(s1=30, n1=100, s2=50, n2=100)

# آزمون برابري واريانس
result = stats.variance_test(a, b, method="levene")

# آزمون اسميرنوف-كولموگروف (KS)
result = stats.ks_test(series_a, series_b)

# اطلاعات متقابل (Mutual Information)
result = stats.mutual_information(X, y)

# آزمون AB (ميانگين)
result = stats.ab_test_mean(control_series, treatment_series)

# آزمون AB (نسبت)
result = stats.ab_test_proportion(control_series, treatment_series)
```

### StatisticalTestSuite

اين كلاس با `DataCleaner` تركيب مي‌شود و آزمون‌ها را روي داده بارگذاري‌شده اجرا مي‌كند:

```python
from clean_data_ml import DataCleaner, stats

dc = DataCleaner()
dc.load_df(data).set_target("purchased")

suite = stats.StatisticalTestSuite(dc)

# آزمون نرمال بودن همه ستون‌هاي عددي
suite.test_normality()

# همبستگي با هدف
suite.test_correlations(target_col="purchased")

# آزمون كايدو بين دو ستون اسمي
suite.test_chi_square("gender", "city")

# آزمون ANOVA
suite.test_anova("age", "city")

# آزمون t مستقل
suite.test_t_independent("age", "score")

# آزمون AB بر اساس گروه
suite.test_ab_by_group(
    metric_col="converted",
    group_col="group",
    control_value="A",
    treatment_value="B",
    metric_type="proportion",
)

# نمايش همه نتايج
print(suite.summary())
```

خروجي `summary()`:

```
=== Normality Tests ===
  age: NORMAL (p=0.2341)
  salary: NOT NORMAL (p=0.0012)

=== Mutual Information ===
  age: 0.1234
  salary: 0.0891
  city: 0.0456

=== AB Test ===
  Significant: YES (p=0.0031)
  Lift: 12.5%
  CI: (0.0234, 0.0891)
  Control: 0.32 (n=500)
  Treatment: 0.45 (n=480)
```

---

## ماژول تصويرسازي (Plotting)

ماژول اختياري `clean_data_ml.plotting` نياز به `matplotlib` و `seaborn` دارد:

```bash
pip install clean-data-ml[plot]
```

```python
from clean_data_ml import plotting

# گزارش مقادير گمشده (نمودار ميلي)
plotting.plot_null_report(dc)

# توزيع ستون‌هاي عددي (هيستوگرام + باكس پلات)
plotting.plot_distributions(dc)
plotting.plot_distributions(dc, cols=["age", "salary"])

# نقشه حرارتي همبستگي
plotting.plot_correlation(dc)
plotting.plot_correlation(dc, figsize=(12, 10))

# مقايسه توزيع Before/After
plotting.plot_before_after(dc)  # نياز به prepare() دارد
```

---

## مثال كامل سرتاسري

### مرحله 1: آموزش و ذخيره Pipeline

```python
from clean_data_ml import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

# ايجاد داده نمونه
data = pd.DataFrame({
    "ID": range(100),
    "age": [25, 30, 35, None, 40, 45, 50, 55, 60, 65] * 10,
    "salary": [50000, 60000, None, 80000, 90000, 100000, 110000, 120000, None, 140000] * 10,
    "city": ["Tehran", "Shiraz", "Tehran", "Isfahan", None, "Tehran", "Shiraz", "Isfahan", "Tehran", "Shiraz"] * 10,
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"] * 10,
    "purchased": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 10,
    "register_date": pd.date_range("2024-01-01", periods=100, freq="D"),
})

# بارگذاري و پيكربندي
dc = DataCleaner(random_state=42)
dc.load_df(data)
dc.set_target("purchased")
dc.drop_columns(["ID"])

# نماي كلي
info = dc.summary()
print(f"Shape: {info['shape']}")
print(f"Nulls: {info['null_counts']}")

# اجراي Pipeline كامل با همه قابليت‌ها
X_train, X_test, y_train, y_test = dc.prepare(
    test_size=0.2,
    handle_nulls=True,
    auto_scale=True,
    auto_encode=True,
    auto_drop_useless=True,
    handle_outliers="clip",
    feature_engineering=True,
    extract_date_features=True,
    add_missing_indicators=True,
    feature_selection="auto",
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"Scalers: {list(dc.pipeline.scalers.keys())}")
print(f"Problem type: {dc.pipeline.problem_type}")
print(f"Useless dropped: {dc.pipeline.dropped_useless_cols}")
print(f"Feature selection dropped: {dc.pipeline.feature_selection_removed}")

# آموزش مدل
model = SVC(probability=True)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# آزمايش Pipeline روي داده جديد
new_data = pd.DataFrame({
    "age": [28, 42, 35],
    "salary": [65000, 95000, 78000],
    "city": ["Tehran", "Isfahan", "Shiraz"],
    "gender": ["F", "M", "F"],
    "register_date": pd.to_datetime(["2024-06-15", "2024-07-20", "2024-08-10"]),
})
processed = dc.transform(new_data)
predictions = model.predict(processed)
probabilities = model.predict_proba(processed)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Purchased" if pred == 1 else "Not Purchased"
    print(f"Customer {i+1}: {status} (confidence: {max(prob):.2%})")

# خروجي گرفتن از داده تميز
dc.export_cleaned("cleaned_data.csv")
dc.export_cleaned("cleaned_with_target.xlsx", include_target=True)

# ذخيره همه چيز
joblib.dump(model, "model.pkl")
dc.save_pipeline("full_pipeline.pkl")
```

### مرحله 2: Inference روي داده جديد

```python
from clean_data_ml import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

# بارگذاري Pipeline و مدل
dc = DataCleaner.load_pipeline("full_pipeline.pkl")
model = joblib.load("model.pkl")

# داده جديد
new_customers = pd.DataFrame({
    "age": [28, 42, 35],
    "salary": [65000, 95000, 78000],
    "city": ["Tehran", "Isfahan", "Shiraz"],
    "gender": ["F", "M", "F"],
    "register_date": pd.to_datetime(["2024-06-15", "2024-07-20", "2024-08-10"]),
})

# تبديل و پيش‌بيني
processed = dc.transform(new_customers)
predictions = model.predict(processed)
probabilities = model.predict_proba(processed)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Purchased" if pred == 1 else "Not Purchased"
    print(f"Customer {i+1}: {status} (confidence: {max(prob):.2%})")
```

### مرحله 3: تحليل آماري

```python
from clean_data_ml import DataCleaner, stats

dc = DataCleaner()
dc.load_df(data)
dc.set_target("purchased")

suite = stats.StatisticalTestSuite(dc)

# آزمون‌هاي مختلف
normality = suite.test_normality()
correlations = suite.test_correlations(target_col="purchased")
chi2 = suite.test_chi_square("gender", "city")
anova = suite.test_anova("age", "city")
ztest = suite.test_z_one_sample("age", pop_mean=35)
ab_result = suite.test_ab_by_group(
    "purchased", "gender", "M", "F", metric_type="proportion"
)
mutual_info = suite.test_mutual_info()

# نمايش خلاصه
print(suite.summary())
```

---

## نكات و رويه‌هاي پيشنهادي

### 1. مدل‌هاي درختي نياز به Scaling ندارند

مدل‌هاي مبتني بر درخت (Random Forest, XGBoost, LightGBM, Decision Tree, Gradient Boosting) به نرمال‌سازي و Scaling نياز ندارند. از `auto_scale=False` استفاده كنيد:

```python
X_train, X_test, y_train, y_test = dc.prepare(auto_scale=False)
```

### 2. ترتيب زنجيره متدها

همه متدهاي پيكربندي `self` برمي‌گردانند، پس مي‌توان زنجيره‌اي نوشت:

```python
dc = DataCleaner()
dc.load_df(data).set_target("purchased").drop_columns(["ID"])
```

### 3. پيش از prepare() حتماً set_target() را صدا بزنيد

اگر هدف را تعيين نكنيد، `prepare()` خطا مي‌دهد:

```
ValueError: Target column not set. Use .set_target() first.
```

### 4. Pipeline را ذخيره كنيد نه DataCleaner را

از `dc.save_pipeline()` استفاده كنيد، نه `pickle.dump(dc)`. Pipeline سبك‌تر است و فقط اجزاي لازم براي Inference را دارد.

### 5. تست روي داده‌هاي جديد بدون هدف

داده‌هاي جديد براي Inference نبايد ستون هدف داشته باشند. اگر داشته باشند، `transform()` آن را ناديده مي‌گيرد (چون در `columns_to_drop` نيست و در `feature_cols` هست، اگر باشد نگه داشته مي‌شود).

### 6. استفاده از n_jobs براي سرعت

براي ديتاست‌هاي بزرگ با ستون‌هاي عددي زياد:

```python
X_train, X_test, y_train, y_test = dc.prepare(n_jobs=-1)  # استفاده از همه هسته‌ها
```

### 7. مهندسي ويژگي + انتخاب ويژگي

تركيب `feature_engineering=True` و `feature_selection="auto"` مي‌تواند بهترين ويژگي‌ها را نگه دارد و ويژگي‌هاي چندجمله‌اي كم‌اهميت را حذف كند.

### 8. شاخص‌هاي گمشدگي + يادگيري الگو

فعال كردن `add_missing_indicators=True` به مدل‌هايي مثل XGBoost و Random Forest كمك مي‌كند از الگوي گمشدگي داده ياد بگيرند.

### 9. استخراج تاريخ براي داده‌هاي زماني

براي ديتاست‌هاي حاوي ستون تاريخ، `extract_date_features=True` را فعال كنيد تا مؤلفه‌هاي زماني (سال، ماه، روز، روز هفته) استخراج شوند.

### 10. خروجي تميز براي تحليل

از `export_cleaned()` براي خروجي گرفتن از داده تميزشده استفاده كنيد تا در ابزارهاي ديگر (Excel, Tableau, Power BI) تحليل شود.

---

## منابع

- **GitHub**: https://github.com/MohammadvHossein/clean-data-ml
- **PyPI**: https://pypi.org/project/clean-data-ml/
- **نويسنده**: Mohammad Hossein Habibpour
- **ايميل**: habibpour.programming@gmail.com
- **مجوز**: MIT
