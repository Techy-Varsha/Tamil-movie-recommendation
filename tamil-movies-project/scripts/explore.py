#!/usr/bin/env python3
"""
Exploration script for Tamil Movies Dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/tamil_movies.csv')

print("=" * 60)
print("TAMIL MOVIES DATASET - EXPLORATION")
print("=" * 60)

# Basic info
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"📅 Year Range: {df['year'].min()} - {df['year'].max()}")
print(f"🎬 Total Movies: {len(df)}")

# Lead actors
print(f"\n🎭 Lead Actors:")
print(df['lead_actor'].value_counts())

# Directors
print(f"\n🎬 Top Directors:")
print(df['director'].value_counts().head(10))

# Genres
print(f"\n🎭 Genres:")
print(df['genre'].value_counts())

# Music directors
print(f"\n🎵 Top Music Directors:")
print(df['music_director'].value_counts().head(10))

# Statistics
print(f"\n📈 Statistics:")
print(f"  Average Rating: {df['rating'].mean():.2f}")
print(f"  Average Runtime: {df['runtime_minutes'].mean():.0f} minutes")
print(f"  Total Box Office: ₹{df['box_office_crores'].sum():.1f} crores")
print(f"  Average Box Office: ₹{df['box_office_crores'].mean():.1f} crores")

# Top rated movies
print(f"\n⭐ Top 5 Rated Movies:")
top_rated = df.nlargest(5, 'rating')[['title', 'year', 'lead_actor', 'rating']]
for _, row in top_rated.iterrows():
    print(f"  {row['title']} ({row['year']}) - {row['lead_actor']} - ⭐ {row['rating']}")

# Highest grossing
print(f"\n💰 Top 5 Highest Grossing:")
top_gross = df.nlargest(5, 'box_office_crores')[['title', 'year', 'lead_actor', 'box_office_crores']]
for _, row in top_gross.iterrows():
    print(f"  {row['title']} ({row['year']}) - {row['lead_actor']} - ₹{row['box_office_crores']:.1f} Cr")

# Decade analysis
df['decade'] = (df['year'] // 10) * 10
print(f"\n📆 Movies per Decade:")
decade_stats = df.groupby('decade').agg(
    count=('title', 'count'),
    avg_rating=('rating', 'mean'),
    total_box_office=('box_office_crores', 'sum'),
    avg_box_office=('box_office_crores', 'mean')
).round(2)
print(decade_stats)

print("\n" + "=" * 60)
print("Exploration complete!")
print("=" * 60)