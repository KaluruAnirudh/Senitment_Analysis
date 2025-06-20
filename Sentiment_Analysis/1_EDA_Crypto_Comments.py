# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the datasets
comments_df = pd.read_csv("comments.csv")
finance_df = pd.read_csv("destription+finance.csv")

# --- Step 1: Basic Info ---
print("📘 COMMENTS DATA")
print(comments_df.info())
print(comments_df.describe(include='all'))

print("\n📗 FINANCE DATA")
print(finance_df.info())
print(finance_df.describe(include='all'))

# --- Step 2: Cleaning ---
# Convert date columns
comments_df['time_posted'] = pd.to_datetime(comments_df['time_posted'], format="%d/%m/%Y", errors='coerce')
finance_df['createdAt'] = pd.to_datetime(finance_df['createdAt'], format="%d/%m/%Y, %H:%M:%S", errors='coerce')
finance_df['bonding_completeTime'] = pd.to_datetime(finance_df['bonding_completeTime'], format="%d/%m/%Y, %H:%M:%S", errors='coerce')

# Clean mCap
finance_df['mCap'] = finance_df['mCap'].str.replace(',', '', regex=False).astype(float)

# Fill missing description
finance_df['description'] = finance_df['description'].fillna("No description")

# --- Step 3: Missing Values ---
print("\nMissing values in comments dataset:\n", comments_df.isnull().sum())
print("\nMissing values in finance dataset:\n", finance_df.isnull().sum())

# --- Step 4: Univariate Analysis ---
# Plot likes distribution
plt.figure(figsize=(8,5))
sns.histplot(comments_df['likes'], bins=50, kde=True)
plt.title("Distribution of Likes")
plt.xlabel("Likes")
plt.ylabel("Frequency")
plt.show()

# Most commented tokens
top_commented = comments_df['token_name'].value_counts().head(10)
top_commented.plot(kind='barh', title="Top 10 Most Commented Tokens", figsize=(8,5))
plt.xlabel("Number of Comments")
plt.gca().invert_yaxis()
plt.show()

# --- Step 5: Bivariate Analysis ---
# Avg likes per token
avg_likes = comments_df.groupby('token_name')['likes'].mean().sort_values(ascending=False).head(10)
avg_likes.plot(kind='bar', title="Top Tokens by Avg Likes", figsize=(8,5))
plt.ylabel("Average Likes")
plt.xticks(rotation=45)
plt.show()

# --- Step 6: Merge Datasets ---
merged_df = pd.merge(comments_df, finance_df, on='token_address', how='left')

# Market Cap of top mentioned tokens
top_mcap = merged_df[['token_name_x', 'mCap']].dropna().groupby('token_name_x').mean().sort_values('mCap', ascending=False).head(10)
top_mcap.plot(kind='bar', figsize=(8,5), legend=False)
plt.title("Top Tokens by Market Cap")
plt.ylabel("Average Market Cap")
plt.xticks(rotation=45)
plt.show()

# --- Step 7: Time Analysis ---
comments_df['day'] = comments_df['time_posted'].dt.date
daily_comments = comments_df.groupby('day').size()

plt.figure(figsize=(10,4))
daily_comments.plot()
plt.title("Number of Comments Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Comments")
plt.grid(True)
plt.show()

# --- Step 8: Word Cloud from Comments ---
all_comments = ' '.join(comments_df['comment'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Comments")
plt.show()