"""
FUNNEL ANALYSIS PROJECT
Clean Version - No Warnings
"""

# ============ 1. IMPORT LIBRARIES ============
import matplotlib
matplotlib.use('Agg')  # Fixes Windows error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

print("✅ Libraries loaded!")


# ============ 2. CREATE FAKE DATA ============
np.random.seed(42)
n_users = 5000

data = {
    'user_id': range(1, n_users + 1),
    'sessions': np.random.choice([1, 2, 3, 4, 5], size=n_users, 
                                 p=[0.5, 0.25, 0.15, 0.07, 0.03]),
    'days_active': np.random.randint(1, 365, size=n_users),
    'device': np.random.choice([0, 1, 2], size=n_users, p=[0.55, 0.35, 0.10]),
    'traffic_source': np.random.choice([0, 1, 2, 3], size=n_users, 
                                       p=[0.40, 0.30, 0.20, 0.10]),
    'max_step': np.random.choice([1, 2, 3, 4, 5, 6, 7], size=n_users,
                                 p=[0.05, 0.15, 0.30, 0.20, 0.15, 0.10, 0.05]),
    'avg_session_time': np.random.randint(30, 600, size=n_users),
}

df = pd.DataFrame(data)

# Save data to CSV
df.to_csv('data/fake_ecommerce_data.csv', index=False)
print("✅ Data saved to data/fake_ecommerce_data.csv")

# Create conversion (1 = bought, 0 = didn't buy)
df['converted'] = 0
df.loc[df['max_step'] == 7, 'converted'] = 1
df.loc[(df['max_step'] == 6) & (np.random.rand(n_users) < 0.7), 'converted'] = 1
df.loc[np.random.choice(df.index, 100), 'avg_session_time'] = np.nan


# ============ 3. CLEAN DATA ============
print("\n📋 Cleaning data...")
print(f"Missing values: {df.isnull().sum().sum()}")

# FIX: Removed inplace=True (no warning now)
df['avg_session_time'] = df['avg_session_time'].fillna(df['avg_session_time'].mean())

print("✅ Data cleaned!")


# ============ 4. BASIC ANALYSIS ============
print("\n📊 BASIC ANALYSIS")
print("-" * 40)

total_users = len(df)
total_buyers = df['converted'].sum()
conversion_rate = total_buyers / total_users * 100

print(f"Total Users:     {total_users}")
print(f"Total Buyers:    {total_buyers}")
print(f"Conversion Rate: {conversion_rate:.2f}%")

# Funnel breakdown
print("\n📌 FUNNEL (How many reached each step):")
step_names = {
    1: 'Opened App',
    2: 'Browsed',
    3: 'Saw Product',
    4: 'Added to Cart',
    5: 'Reached Checkout',
    6: 'Entered Payment',
    7: 'Purchased'
}

for step in range(1, 8):
    users = (df['max_step'] >= step).sum()
    pct = users / total_users * 100
    print(f"{step_names[step]:<20} {users:>6,} users ({pct:>5.1f}%)")


# ============ 5. CONVERSION BY DEVICE ============
print("\n📌 CONVERSION BY DEVICE:")
device_names = {0: 'Mobile', 1: 'Desktop', 2: 'Tablet'}
df['device_name'] = df['device'].map(device_names)

device_conv = df.groupby('device_name')['converted'].mean() * 100
for device, rate in device_conv.sort_values(ascending=False).items():
    print(f"  {device}: {rate:.2f}%")


# ============ 6. CONVERSION BY TRAFFIC ============
print("\n📌 CONVERSION BY TRAFFIC SOURCE:")
source_names = {0: 'Organic', 1: 'Paid Ad', 2: 'Referral', 3: 'Social'}
df['source_name'] = df['traffic_source'].map(source_names)

source_conv = df.groupby('source_name')['converted'].mean() * 100
for source, rate in source_conv.sort_values(ascending=False).items():
    print(f"  {source}: {rate:.2f}%")


# ============ 7. CREATE CHARTS ============
print("\n🎨 Creating charts...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Funnel Analysis', fontsize=16, fontweight='bold')

# Chart 1: Funnel
funnel_counts = [(df['max_step'] >= s).sum() for s in range(1, 8)]
funnel_labels = [step_names[s] for s in range(1, 8)]

axes[0, 0].barh(funnel_labels, funnel_counts, color='blue')
axes[0, 0].set_xlabel('Users')
axes[0, 0].set_title('Users at Each Step')
axes[0, 0].invert_yaxis()

# Chart 2: Device
axes[0, 1].bar(device_conv.index, device_conv.values, color=['green', 'orange', 'red'])
axes[0, 1].set_title('Conversion by Device')
axes[0, 1].set_ylabel('%')

# Chart 3: Traffic
axes[1, 0].bar(source_conv.index, source_conv.values, color='coral')
axes[1, 0].set_title('Conversion by Traffic')
axes[1, 0].set_ylabel('%')

# Chart 4: Session Time
buyers_time = df[df['converted'] == 1]['avg_session_time']
non_buyers_time = df[df['converted'] == 0]['avg_session_time']

axes[1, 1].hist(non_buyers_time, bins=30, alpha=0.6, label='No Buy', color='red')
axes[1, 1].hist(buyers_time, bins=30, alpha=0.6, label='Bought', color='green')
axes[1, 1].set_title('Session Time: Buyers vs Non-Buyers')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('funnel_charts.png', dpi=150)
print("✅ Charts saved as funnel_charts.png")
# FIX: Removed plt.show() (no warning - charts saved to file)


# ============ 8. MACHINE LEARNING MODEL ============
print("\n🤖 Training ML Model...")

features = ['sessions', 'days_active', 'device', 'traffic_source', 
            'max_step', 'avg_session_time']

X = df[features]
y = df['converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train)} users, Testing: {len(X_test)} users")

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred) * 100
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📊 MODEL RESULTS:")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"  ROC-AUC:  {roc_auc:.3f}")

# Feature importance
print(f"\n📌 What matters most for buying:")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"  {row['Feature']:<20} {row['Importance']:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Buy', 'Buy'],
            yticklabels=['No Buy', 'Buy'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\n✅ Confusion matrix saved")
# FIX: Removed plt.show() (no warning - saved to file)


# ============ 9. PREDICT NEW USER ============
print("\n🔮 Predicting for new user...")

new_user = pd.DataFrame([{
    'sessions': 3,
    'days_active': 15,
    'device': 1,
    'traffic_source': 0,
    'max_step': 4,
    'avg_session_time': 300
}])

prob = model.predict_proba(new_user)[0][1] * 100

print(f"Buy probability: {prob:.1f}%")

if prob > 50:
    print("→ LIKELY TO BUY ✅ (Send free shipping!)")
elif prob > 25:
    print("→ MIGHT BUY ⚠️ (Send 10% discount)")
else:
    print("→ UNLIKELY ❌ (Show onboarding tips)")


# ============ 10. BUSINESS INSIGHTS ============
print("\n💡 KEY INSIGHTS")
print("=" * 50)

# Find biggest drop-off
biggest_drop = 0
biggest_step = 0
for step in range(2, 7):
    curr = (df['max_step'] >= step).sum()
    prev = (df['max_step'] >= step - 1).sum()
    drop = (prev - curr) / prev * 100
    if drop > biggest_drop:
        biggest_drop = drop
        biggest_step = step

print(f"1. Biggest drop: Step {biggest_step} ({step_names[biggest_step]})")
print(f"   {biggest_drop:.1f}% users leave here")

print(f"\n2. Best device: {device_conv.idxmax()} ({device_conv.max():.1f}%)")
print(f"   Worst: {device_conv.idxmin()} ({device_conv.min():.1f}%)")

print(f"\n3. Best traffic: {source_conv.idxmax()} ({source_conv.max():.1f}%)")

avg_buyer = df[df['converted'] == 1]['avg_session_time'].mean()
avg_nonbuy = df[df['converted'] == 0]['avg_session_time'].mean()
print(f"\n4. Session time:")
print(f"   Buyers: {avg_buyer:.0f}s, Non-buyers: {avg_nonbuy:.0f}s")

print(f"\n5. Model accuracy: {accuracy:.1f}%")

print("\n" + "=" * 50)
print("🎉 DONE! Files saved:")
print("   - data/fake_ecommerce_data.csv")
print("   - funnel_charts.png")
print("   - confusion_matrix.png")
print("=" * 50)