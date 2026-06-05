#!/usr/bin/env python3
"""
User Funnel Analysis - Complete End-to-End Python Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("USER FUNNEL ANALYSIS - COMPLETE PROJECT")
print("=" * 80)


def generate_funnel_data(n_users=50000, n_days=365):
    events_list = []
    event_types = [
        'App Open', 'Browse', 'Product View', 'Add to Cart',
        'Checkout', 'Payment', 'Purchase'
    ]
    devices = ['Mobile', 'Desktop', 'Tablet']
    traffic_sources = ['Organic', 'Paid', 'Referral', 'Social', 'Direct']
    regions = ['North', 'South', 'East', 'West', 'Central']
    categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books', 'Beauty']

    start_date = datetime(2024, 1, 1)

    print(f"\n[DATA GENERATION] Creating dataset with {n_users:,} users...")

    for user_id in range(1, n_users + 1):
        n_sessions = np.random.choice([1, 2, 3, 4, 5], p=[0.7, 0.15, 0.08, 0.05, 0.02])

        for session_num in range(n_sessions):
            session_id = f"S_{user_id}_{session_num}"
            device = np.random.choice(devices, p=[0.55, 0.35, 0.10])
            traffic_source = np.random.choice(traffic_sources, p=[0.40, 0.30, 0.15, 0.10, 0.05])
            region = np.random.choice(regions)

            base_conversion_prob = 0.032
            if device == 'Desktop':
                base_conversion_prob *= 1.5
            if traffic_source == 'Organic':
                base_conversion_prob *= 1.3

            will_convert = np.random.rand() < base_conversion_prob
            session_start = start_date + timedelta(days=np.random.randint(0, n_days))
            current_time = session_start

            price = np.random.exponential(2000) + 500

            events_list.append({
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': current_time,
                'event_type': 'App Open',
                'device_type': device,
                'region': region,
                'traffic_source': traffic_source,
                'product_category': np.random.choice(categories),
                'product_price': price,
                'session_duration_seconds': 0,
                'conversion_flag': 1 if will_convert else 0
            })

            progression = {
                'App Open': ('Browse', 0.85),
                'Browse': ('Product View', 0.60),
                'Product View': ('Add to Cart', 0.40),
                'Add to Cart': ('Checkout', 0.50),
                'Checkout': ('Payment', 0.31),
                'Payment': ('Purchase', 1.0)
            }

            current_event = 'App Open'
            session_duration = 0

            while current_event != 'Purchase' and current_event in progression:
                next_event, probability = progression[current_event]

                if not will_convert:
                    probability *= np.random.uniform(0.3, 0.7)

                if np.random.rand() < probability:
                    current_time += timedelta(seconds=np.random.randint(30, 300))
                    session_duration = (current_time - session_start).total_seconds()

                    if next_event == 'Purchase':
                        conv_flag = 1
                    else:
                        conv_flag = 0

                    events_list.append({
                        'user_id': user_id,
                        'session_id': session_id,
                        'timestamp': current_time,
                        'event_type': next_event,
                        'device_type': device,
                        'region': region,
                        'traffic_source': traffic_source,
                        'product_category': np.random.choice(categories),
                        'product_price': price,
                        'session_duration_seconds': int(session_duration),
                        'conversion_flag': conv_flag
                    })
                    current_event = next_event
                else:
                    break

        if user_id % 10000 == 0:
            print(f"  Generated {user_id:,} users...")

    df = pd.DataFrame(events_list)
    print(f"✅ Dataset generated: {len(df):,} events from {df['user_id'].nunique():,} users")
    return df


def clean_data(df):
    print("\n[DATA CLEANING]")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    initial_rows = len(df)
    df = df.drop_duplicates(subset=['user_id', 'session_id', 'timestamp', 'event_type'])
    print(f"  Removed {initial_rows - len(df)} duplicate rows")

    suspicious = df.groupby('session_id').filter(
        lambda x: len(x) > 100 and (x['timestamp'].max() - x['timestamp'].min()).total_seconds() < 300
    )
    df = df[~df['session_id'].isin(suspicious['session_id'])]
    print(f"  Removed {suspicious['session_id'].nunique()} suspicious sessions")

    df['product_price'] = df['product_price'].fillna(df['product_price'].median())
    print(f"✅ Cleaned dataset: {len(df):,} events")
    return df


def engineer_features(df):
    print("\n[FEATURE ENGINEERING]")

    session_features = df.groupby('session_id').agg({
        'user_id': 'first',
        'device_type': 'first',
        'region': 'first',
        'traffic_source': 'first',
        'timestamp': ['min', 'max', 'count'],
        'conversion_flag': 'max'
    }).reset_index()

    session_features.columns = [
        'session_id', 'user_id', 'device_type', 'region',
        'traffic_source', 'session_start', 'session_end',
        'event_count', 'converted'
    ]

    session_features['session_duration'] = (
        session_features['session_end'] - session_features['session_start']
    ).dt.total_seconds()

    session_features['hour_of_day'] = session_features['session_start'].dt.hour
    session_features['day_of_week'] = session_features['session_start'].dt.dayofweek

    user_features = df.groupby('user_id').agg({
        'session_id': 'nunique',
        'timestamp': ['min', 'max'],
        'conversion_flag': ['sum', 'max'],
    }).reset_index()

    user_features.columns = [
        'user_id', 'session_frequency', 'first_visit',
        'last_visit', 'total_conversions', 'ever_converted'
    ]

    user_features['user_lifetime_days'] = (
        user_features['last_visit'] - user_features['first_visit']
    ).dt.days + 1

    user_features['avg_sessions_per_day'] = (
        user_features['session_frequency'] / user_features['user_lifetime_days']
    ).fillna(0)

    stage_mapping = {
        'App Open': 1, 'Browse': 2, 'Product View': 3, 'Add to Cart': 4,
        'Checkout': 5, 'Payment': 6, 'Purchase': 7
    }

    df['stage_number'] = df['event_type'].map(stage_mapping)
    max_stage = df.groupby('user_id')['stage_number'].max().reset_index()
    max_stage.columns = ['user_id', 'max_stage_reached']

    user_features = user_features.merge(max_stage, on='user_id')

    print(f"✅ Engineered {len(user_features)} user features")
    print(f"  Features: {list(user_features.columns)}")

    return user_features, df


def perform_eda(df):
    print("\n[EXPLORATORY DATA ANALYSIS]")

    conversion_rate = df['conversion_flag'].mean() * 100
    print(f"\n  Overall Conversion Rate: {conversion_rate:.2f}%")
    print(f"  Total Users: {df['user_id'].nunique():,}")
    print(f"  Total Sessions: {df['session_id'].nunique():,}")
    print(f"  Total Events: {len(df):,}")

    print("\n  Funnel Analysis:")
    event_order = [
        'App Open', 'Browse', 'Product View', 'Add to Cart',
        'Checkout', 'Payment', 'Purchase'
    ]

    app_open_users = df[df['event_type'] == 'App Open']['user_id'].nunique()
    for event in event_order:
        count = df[df['event_type'] == event]['user_id'].nunique()
        pct = 100 * count / app_open_users if app_open_users else 0
        print(f"    {event:20} {count:>10,} users ({pct:>5.1f}%)")

    print("\n  Conversion by Device:")
    device_conv = df.groupby('device_type')['conversion_flag'].mean() * 100
    for device, conv in device_conv.items():
        print(f"    {device:15} {conv:>5.2f}%")

    print("\n  Conversion by Traffic Source:")
    traffic_conv = df.groupby('traffic_source')['conversion_flag'].mean() * 100
    for source in traffic_conv.sort_values(ascending=False).index:
        print(f"    {source:15} {traffic_conv[source]:>5.2f}%")

    return conversion_rate


def create_visualizations(df, user_features):
    print("\n[CREATING VISUALIZATIONS]")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('User Funnel Analysis - Key Metrics', fontsize=16, fontweight='bold')

    event_order = [
        'App Open', 'Browse', 'Product View', 'Add to Cart',
        'Checkout', 'Payment', 'Purchase'
    ]
    funnel_data = df['event_type'].value_counts().reindex(event_order).fillna(0)

    axes[0, 0].barh(range(len(funnel_data)), funnel_data.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(funnel_data)))
    axes[0, 0].set_yticklabels(funnel_data.index)
    axes[0, 0].set_xlabel('User Count')
    axes[0, 0].set_title('Funnel Stages - User Count')
    axes[0, 0].invert_yaxis()

    for i, v in enumerate(funnel_data.values):
        if funnel_data.values[0] > 0:
            pct = 100 * v / funnel_data.values[0]
        else:
            pct = 0
        axes[0, 0].text(v, i, f' {pct:.1f}%', va='center', fontweight='bold')

    device_conv = df.groupby('device_type')['conversion_flag'].mean() * 100
    device_conv = device_conv.reindex(['Mobile', 'Desktop', 'Tablet']).fillna(0)
    axes[0, 1].bar(device_conv.index, device_conv.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_ylabel('Conversion Rate (%)')
    axes[0, 1].set_title('Conversion Rate by Device')
    axes[0, 1].set_ylim(0, max(device_conv.values) * 1.2 if len(device_conv) else 1)

    for i, v in enumerate(device_conv.values):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')

    traffic_conv = df.groupby('traffic_source')['conversion_flag'].mean() * 100
    traffic_conv = traffic_conv.sort_values(ascending=False)
    axes[0, 2].barh(range(len(traffic_conv)), traffic_conv.values, color='coral')
    axes[0, 2].set_yticks(range(len(traffic_conv)))
    axes[0, 2].set_yticklabels(traffic_conv.index)
    axes[0, 2].set_xlabel('Conversion Rate (%)')
    axes[0, 2].set_title('Conversion by Traffic Source')

    for i, v in enumerate(traffic_conv.values):
        axes[0, 2].text(v, i, f' {v:.2f}%', va='center')

    drop_off_data = []
    prev_count = funnel_data.values[0] if len(funnel_data) else 0
    for count in funnel_data.values[1:]:
        drop_off = ((prev_count - count) / prev_count * 100) if prev_count > 0 else 0
        drop_off_data.append(drop_off)
        prev_count = count

    axes[1, 0].bar(range(len(drop_off_data)), drop_off_data, color='#FF6B6B')
    axes[1, 0].set_xticks(range(len(drop_off_data)))
    axes[1, 0].set_xticklabels(
        ['App Open→Browse', 'Browse→Product View', 'Product View→Add to Cart',
         'Add to Cart→Checkout', 'Checkout→Payment', 'Payment→Purchase'],
        rotation=45, ha='right'
    )
    axes[1, 0].set_ylabel('Drop-off Rate (%)')
    axes[1, 0].set_title('Drop-off Rate Between Stages')

    duration = df[df['session_duration_seconds'] > 0]['session_duration_seconds']
    if len(duration) > 0:
        axes[1, 1].hist(duration, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Session Duration (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Session Duration Distribution')
        axes[1, 1].axvline(duration.median(), color='red', linestyle='--',
                           label=f'Median: {duration.median():.0f}s')
        axes[1, 1].legend()
    else:
        axes[1, 1].set_title('Session Duration Distribution')
        axes[1, 1].text(0.5, 0.5, 'No duration data', ha='center', va='center')
        axes[1, 1].axis('off')

    top_regions = df.groupby('region')['conversion_flag'].agg(['sum', 'count'])
    top_regions['conv_rate'] = (top_regions['sum'] / top_regions['count'] * 100).sort_values(ascending=False)
    top_regions = top_regions['conv_rate'].head(10)

    axes[1, 2].barh(range(len(top_regions)), top_regions.values, color='#45B7D1')
    axes[1, 2].set_yticks(range(len(top_regions)))
    axes[1, 2].set_yticklabels(top_regions.index)
    axes[1, 2].set_xlabel('Conversion Rate (%)')
    axes[1, 2].set_title('Top Regions by Conversion')

    plt.tight_layout()
    plt.savefig('funnel_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: funnel_analysis.png")
    plt.close()

    heatmap_data = df.pivot_table(
        values='conversion_flag',
        index='device_type',
        columns='traffic_source',
        aggfunc='mean'
    ) * 100

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Conversion Rate (%)'}
    )
    ax.set_title('Conversion Rate Heatmap: Device vs Traffic Source')
    ax.set_xlabel('Traffic Source')
    ax.set_ylabel('Device Type')
    plt.tight_layout()
    plt.savefig('conversion_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: conversion_heatmap.png")
    plt.close()


def build_ml_models(user_features):
    print("\n[MACHINE LEARNING MODELS]")

    features = [
        'session_frequency', 'user_lifetime_days', 'avg_sessions_per_day',
        'max_stage_reached'
    ]

    X = user_features[features].fillna(0)
    y = user_features['ever_converted']

    print(f"\n  Target Distribution:")
    print(f"    Converters: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
    print(f"    Non-converters: {(1-y).sum():,} ({(1-y).sum()/len(y)*100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Train set: {X_train.shape[0]:,} | Test set: {X_test.shape[0]:,}")

    print("\n  Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_auc = roc_auc_score(y_test, y_pred_proba_lr)

    print(f"    Accuracy: {lr_accuracy:.4f}")
    print(f"    ROC-AUC: {lr_auc:.4f}")

    print("\n  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_auc = roc_auc_score(y_test, y_pred_proba_rf)

    print(f"    Accuracy: {rf_accuracy:.4f}")
    print(f"    ROC-AUC: {rf_auc:.4f}")

    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [lr_accuracy, rf_accuracy],
        'ROC-AUC': [lr_auc, rf_auc]
    })

    print("\n  Model Comparison:")
    print(comparison_df.to_string(index=False))

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n  Feature Importance (Random Forest):")
    for _, row in feature_importance.iterrows():
        print(f"    {row['Feature']:30} {row['Importance']:>6.4f}")

    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred_proba_lr': y_pred_proba_lr,
        'y_pred_proba_rf': y_pred_proba_rf,
        'comparison': comparison_df
    }


def generate_insights(df):
    print("\n[BUSINESS INSIGHTS]")
    print("\n" + "=" * 80)
    print("TOP 5 KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. 🔴 CRITICAL BOTTLENECK: Product View → Add to Cart")
    product_count = df[df['event_type'] == 'Product View']['user_id'].nunique()
    cart_count = df[df['event_type'] == 'Add to Cart']['user_id'].nunique()
    dropoff = ((product_count - cart_count) / product_count * 100) if product_count else 0
    print(f"   Drop-off Rate: {dropoff:.1f}%")
    print(f"   Lost Users/Month: {(product_count - cart_count):,}")
    print(f"   Revenue Impact: ${((product_count - cart_count) * 2500 * 12 / 1_000_000):.0f}M annually")

    print("\n2. 📱 DEVICE DISPARITY: Mobile Conversion Gap")
    device_conv = df.groupby('device_type')['conversion_flag'].mean() * 100
    desktop_conv = device_conv.get('Desktop', 0)
    mobile_conv = device_conv.get('Mobile', 0)
    gap = desktop_conv - mobile_conv
    print(f"   Desktop: {desktop_conv:.2f}% | Mobile: {mobile_conv:.2f}% (Gap: {gap:.2f}%)")

    print("\n3. 📊 TRAFFIC QUALITY: Channel ROI Variation")
    traffic_conv = df.groupby('traffic_source')['conversion_flag'].mean() * 100
    print(f"   Best: {traffic_conv.idxmax()} ({traffic_conv.max():.2f}%)")
    print(f"   Worst: {traffic_conv.idxmin()} ({traffic_conv.min():.2f}%)")

    print("\n4. 🌍 REGIONAL STRATEGY: Geographic Optimization")
    region_conv = (df.groupby('region')['conversion_flag'].mean() * 100).sort_values(ascending=False)
    print(f"   Top Region: {region_conv.idxmax()} ({region_conv.max():.2f}%)")
    print(f"   Bottom Region: {region_conv.idxmin()} ({region_conv.min():.2f}%)")
    print(f"   Gap: {region_conv.max() - region_conv.min():.2f}%")

    print("\n5. ⏱️ ENGAGEMENT PREDICTOR: Session Duration Analysis")
    converters_duration = df[df['conversion_flag'] == 1]['session_duration_seconds'].mean()
    non_converters_duration = df[df['conversion_flag'] == 0]['session_duration_seconds'].mean()
    ratio = converters_duration / non_converters_duration if non_converters_duration else 0
    print(f"   Converters Avg: {converters_duration:.0f}s")
    print(f"   Non-converters Avg: {non_converters_duration:.0f}s")
    print(f"   Ratio: {ratio:.1f}X")

    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("STARTING COMPLETE USER FUNNEL ANALYSIS PROJECT")
    print("=" * 80)

    df = generate_funnel_data(n_users=50000, n_days=365)
    df = clean_data(df)
    user_features, df = engineer_features(df)
    conversion_rate = perform_eda(df)
    create_visualizations(df, user_features)
    models = build_ml_models(user_features)
    generate_insights(df)

    print("\n" + "=" * 80)
    print("PROJECT COMPLETE ✅")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  • funnel_analysis.png - Main visualizations")
    print("  • conversion_heatmap.png - Device × Traffic heatmap")
    print("\nKey Metrics:")
    print(f"  • Conversion Rate: {conversion_rate:.2f}%")
    print(f"  • Total Users: {df['user_id'].nunique():,}")
    print(f"  • Total Events: {len(df):,}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()