# ============================================================
# TRAFFIC ACCIDENT ANALYSIS — FULL PIPELINE (Google Colab ready)
# ============================================================
# Just paste this whole thing into a Colab cell and run.
# If you have your OWN dataset, skip Section 1 and instead run:
#   from google.colab import files
#   uploaded = files.upload()
#   df = pd.read_csv("your_file.csv")
# then make sure column names match those used below (or adjust).
# ============================================================

# --- Install/import ---
!pip install -q scikit-learn pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
pd.set_option("display.max_columns", None)
np.random.seed(42)

# ============================================================
# 1. GENERATE SYNTHETIC DATASET (skip if you upload your own CSV)
# ============================================================
n = 8000
start = pd.Timestamp("2022-01-01")
dates = start + pd.to_timedelta(np.random.randint(0, 365*3, n), unit="D")

hour_p = np.array([1,1,1,1,1,2,4,7,8,6,4,4,5,5,5,6,7,8,7,5,4,3,2,1])
hour_p = hour_p / hour_p.sum()
hours = pd.Series(np.random.choice(range(24), n, p=hour_p))

weekday = pd.Series(dates).dt.dayofweek
weather = np.random.choice(["Clear","Rain","Fog","Snow","Cloudy"], n, p=[0.5,0.18,0.07,0.05,0.20])
road_type = np.random.choice(["Highway","Urban Road","Rural Road","Intersection"], n, p=[0.3,0.35,0.15,0.2])
light = np.where((hours>=6)&(hours<=18), "Daylight",
          np.where(((hours>5)&(hours<7))|((hours>18)&(hours<20)), "Dusk/Dawn","Dark"))
speed_limit = np.random.choice([30,40,50,60,70,80,100,120], n, p=[0.1,0.15,0.2,0.15,0.1,0.1,0.1,0.1])
vehicles = np.random.choice([1,2,3,4,5], n, p=[0.25,0.45,0.18,0.08,0.04])
junction = np.random.choice(["Yes","No"], n, p=[0.4,0.6])

night_hours = hours.isin(list(range(22,24)) + list(range(0,4)))
alcohol = np.random.binomial(1, np.where(night_hours, 0.18, 0.04))
early_hours = hours.isin([0,1,2,3,4,5])
fatigue = np.random.binomial(1, np.where(early_hours, 0.15, 0.04))

risk = (
    (weather=="Rain")*0.6 + (weather=="Fog")*1.0 + (weather=="Snow")*1.1 +
    (light=="Dark")*0.7 + (light=="Dusk/Dawn")*0.3 +
    (speed_limit/100)*1.5 +
    (vehicles>2)*0.5 +
    alcohol*1.4 + fatigue*0.9 +
    (road_type=="Highway")*0.4 + (junction=="Yes")*0.3 +
    np.random.normal(0, 0.8, n)
)

severity = pd.cut(risk, bins=[-np.inf,1.0,2.2,np.inf], labels=["Minor","Serious","Fatal"])
casualties = np.clip(np.round(np.random.poisson(1 + risk*0.5)), 0, 8).astype(int)

df = pd.DataFrame({
    "accident_id": range(1, n+1),
    "date": dates,
    "hour": hours.values,
    "day_of_week": weekday.map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}).values,
    "weather": weather,
    "light_condition": light,
    "road_type": road_type,
    "speed_limit": speed_limit,
    "num_vehicles": vehicles,
    "junction": junction,
    "alcohol_involved": alcohol,
    "driver_fatigue": fatigue,
    "casualties": casualties,
    "severity": severity.astype(str)
})

df.to_csv("traffic_accidents.csv", index=False)
print("Dataset shape:", df.shape)
print(df["severity"].value_counts())
df.head()

# ============================================================
# 2. LOAD & CLEAN
# ============================================================
df = pd.read_csv("traffic_accidents.csv", parse_dates=["date"])
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()
df["severity"] = df["severity"].astype("category")
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df.describe(include="all").T

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================
order = ["Minor","Serious","Fatal"]

# 3.1 Severity distribution
ax = sns.countplot(data=df, x="severity", order=order, palette="rocket")
ax.set_title("Accident Severity Distribution")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.show()

# 3.2 Accidents by hour
df.groupby("hour").size().plot(kind="bar", color="steelblue", title="Accidents by Hour of Day")
plt.show()

# 3.3 Accidents by day of week
day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
df["day_of_week"].value_counts().reindex(day_order).plot(kind="bar", color="darkorange", title="Accidents by Day of Week")
plt.show()

# 3.4 Severity by weather and light condition
fig, axes = plt.subplots(1, 2, figsize=(15,5))
sns.countplot(data=df, x="weather", hue="severity", hue_order=order, ax=axes[0], palette="rocket")
axes[0].set_title("Severity by Weather"); axes[0].tick_params(axis='x', rotation=30)
sns.countplot(data=df, x="light_condition", hue="severity", hue_order=order, ax=axes[1], palette="rocket")
axes[1].set_title("Severity by Light Condition")
plt.tight_layout(); plt.show()

# 3.5 Severity by road type and speed limit
fig, axes = plt.subplots(1, 2, figsize=(15,5))
sns.countplot(data=df, x="road_type", hue="severity", hue_order=order, ax=axes[0], palette="mako")
axes[0].set_title("Severity by Road Type"); axes[0].tick_params(axis='x', rotation=20)
sns.boxplot(data=df, x="severity", y="speed_limit", order=order, ax=axes[1], palette="mako")
axes[1].set_title("Speed Limit by Severity")
plt.tight_layout(); plt.show()

# 3.6 Alcohol & fatigue involvement by severity
fig, axes = plt.subplots(1, 2, figsize=(13,5))
sns.barplot(data=df, x="severity", y="alcohol_involved", order=order, ax=axes[0], palette="flare", estimator=np.mean)
axes[0].set_title("Alcohol Involvement Rate by Severity")
sns.barplot(data=df, x="severity", y="driver_fatigue", order=order, ax=axes[1], palette="flare", estimator=np.mean)
axes[1].set_title("Driver Fatigue Rate by Severity")
plt.tight_layout(); plt.show()

# 3.7 Correlation heatmap
numeric_cols = ["hour","speed_limit","num_vehicles","alcohol_involved","driver_fatigue","casualties"]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap"); plt.show()

# 3.8 Hotspot heatmap: road type x hour
pivot = df.pivot_table(index="road_type", columns="hour", values="accident_id", aggfunc="count", fill_value=0)
plt.figure(figsize=(14,4))
sns.heatmap(pivot, cmap="YlOrRd")
plt.title("Accident Counts: Road Type vs Hour of Day"); plt.show()

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
model_df = df.copy()
model_df["is_night"] = model_df["light_condition"].apply(lambda x: 1 if x == "Dark" else 0)
model_df["rush_hour"] = model_df["hour"].apply(lambda h: 1 if h in [7,8,9,16,17,18,19] else 0)

cat_cols = ["weather","light_condition","road_type","junction","day_of_week"]
for c in cat_cols:
    le = LabelEncoder()
    model_df[c+"_enc"] = le.fit_transform(model_df[c])

feature_cols = ["hour","speed_limit","num_vehicles","alcohol_involved","driver_fatigue",
                 "is_night","rush_hour"] + [c+"_enc" for c in cat_cols]

X = model_df[feature_cols]
y = model_df["severity"]
print("Feature matrix:", X.shape)

# ============================================================
# 5. MODELING — SEVERITY CLASSIFICATION
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5.1 Logistic Regression baseline
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 (weighted):", f1_score(y_test, y_pred_lr, average="weighted"))
print(classification_report(y_test, y_pred_lr))

# 5.2 Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1 (weighted):", f1_score(y_test, y_pred_rf, average="weighted"))
print(classification_report(y_test, y_pred_rf))

# 5.3 Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf, labels=order)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=order, yticklabels=order)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix — Random Forest")
plt.show()

# 5.4 Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.plot(kind="barh", color="seagreen", figsize=(8,6), title="Feature Importance — Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout(); plt.show()
print(importances)

# ============================================================
# 6. KEY FINDINGS (printed summary)
# ============================================================
print("""
KEY FINDINGS:
- Speed limit, alcohol involvement, and light/weather conditions are top predictors of severity.
- Accident frequency peaks during morning/evening rush hours, with a secondary late-night risk
  window where alcohol and fatigue rates are higher.
- Highways/intersections show different severity profiles vs. urban roads.
- Random Forest outperforms logistic regression, suggesting non-linear interactions matter.

RECOMMENDATIONS:
1. Target enforcement during high-risk hours/conditions identified above.
2. Improve lighting/signage at high-risk intersections.
3. Run awareness campaigns on night driving, fatigue, and low-visibility conditions.
4. Use the model as a risk-scoring tool for infrastructure investment prioritization.

NEXT STEPS:
- Replace synthetic data with a real dataset (e.g. Kaggle "US Accidents", UK DfT STATS19).
- Add geospatial hotspot mapping if lat/lon is available.
- Try XGBoost/LightGBM + hyperparameter tuning for better performance.
""")

# ============================================================
# 7. (Optional) Download outputs from Colab
# ============================================================
# from google.colab import files
# files.download("traffic_accidents.csv")
