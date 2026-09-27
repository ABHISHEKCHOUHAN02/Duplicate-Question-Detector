import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(r"D:\vscode\ML\Duplicate Question Detector\data\raw_data\train.csv")

df.dropna( subset=['question1', 'question2'], inplace=True)

# First split: 70% train, 30% temp (will become val + test)
#strtify is used to ensure that the class distribution is preserved in the splits
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_duplicate'])

# Second split: 50% validation, 50% test from the temp_df
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['is_duplicate'])

train_df.to_csv("data/processed/train_split.csv", index=False)
val_df.to_csv("data/processed/val_split.csv", index=False)
test_df.to_csv("data/processed/test_split.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")