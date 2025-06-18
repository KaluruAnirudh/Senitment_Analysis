import pandas as pd
import re

# Load the file
df = pd.read_csv("Sentiment_Analysis\destription+finance.csv")

# Function to remove emojis and special characters
def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    # Remove emojis
    text = re.sub(r'[^\w\s,.!?@%&()\-:/]', '', text)
    # Remove unwanted special characters (keeping .,!?@%&()-:/)
    text = re.sub(r'[^\w\s]', '', text)
    # Strip extra spaces
    return text.strip()

# Apply to textual columns
text_columns = ['token_name', 'token_ticker', 'token_address', 'description']
for col in text_columns:
    df[col] = df[col].apply(clean_text)

# Clean and convert mCap column
df['mCap'] = df['mCap'].astype(str)
df['mCap'] = df['mCap'].str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters
df['mCap'] = pd.to_numeric(df['mCap'], errors='coerce')

# Check result
print("Cleaned data preview:")
print(df.head())
print("\nMissing values in 'mCap':", df['mCap'].isna().sum())
df.to_csv("Sentiment_Analysis\cleaned_finance_cleaned.csv", index=False)
print("Cleaned file saved.")