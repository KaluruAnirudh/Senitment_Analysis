import pandas as pd

# Load datasets
df_comments = pd.read_csv("D:\\NIT_Trichy Project\\Sentiment_Analysis\\comments_with_scores.csv")
df_finance = pd.read_csv("D:\\NIT_Trichy Project\\Sentiment_Analysis\\cleaned_finance_cleaned.csv")

# Ensure sentiment_score is numeric
df_comments['sentiment_score'] = df_comments['sentiment_score'].astype(float)

# Average sentiment per token
avg_sentiment = df_comments.groupby("token_ticker")["sentiment_score"].mean().reset_index()

# Merge with finance data to get market cap
df_finance['mCap'] = pd.to_numeric(df_finance['mCap'], errors='coerce')  # Clean mCap values
sentiment_vs_finance = pd.merge(avg_sentiment, df_finance[['token_ticker', 'mCap']], on='token_ticker')

# Drop missing mCap if any
sentiment_vs_finance = sentiment_vs_finance.dropna(subset=['mCap'])

# Save to CSV
sentiment_vs_finance.to_csv("sentiment_vs_finance.csv", index=False)

print("✅ sentiment_vs_finance.csv created successfully!")