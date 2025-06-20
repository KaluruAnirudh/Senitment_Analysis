import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


# Load comments data
comments_df = pd.read_csv(r"D:\NIT_Trichy Project\Sentiment_Analysis\comments.csv")

# Drop rows with missing comments
comments_df = comments_df.dropna(subset=['comment'])

# Clean and tokenize function
def clean_comment(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Apply cleaning
comments_df['clean_comment'] = comments_df['comment'].apply(clean_comment)

# Save to your desired path
comments_df.to_csv(r"D:\NIT_Trichy Project\Sentiment_Analysis\cleaned_comments.csv", index=False)
print("✅ Cleaned comments saved to: D:\\NIT_Trichy Project\\Sentiment_Analysis\\cleaned_comments.csv")

# Load finance data
finance_df = pd.read_csv(r"D:\NIT_Trichy Project\Sentiment_Analysis\destription+finance.csv")  

# Convert dates and numeric columns
finance_df['createdAt'] = pd.to_datetime(finance_df['createdAt'], dayfirst=True, errors='coerce')
finance_df['mCap'] = pd.to_numeric(finance_df['mCap'], errors='coerce')

# Drop rows with missing essential values
#finance_df = finance_df.dropna(subset=['createdAt', 'mCap', 'token_name'])

# Save to your desired path
finance_df.to_csv(r"D:\NIT_Trichy Project\Sentiment_Analysis\cleaned_finance.csv", index=False)
print("✅ Cleaned financial data saved to: D:\\NIT_Trichy Project\\Sentiment_Analysis\\cleaned_finance.csv")