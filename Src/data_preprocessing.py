import pandas as pd

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Perform data cleaning."""
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    # Additional cleaning steps
    return df

def preprocess_data(df):
    """Perform data preprocessing."""
    # Encoding categorical variables, normalizing data, etc.
    df = pd.get_dummies(df, drop_first=True)
    return df

def save_processed_data(df, file_path):
    """Save the processed data to a file."""
    df.to_csv(file_path, index=False)

# Load the data
df = load_data("C:\\Users\\91800\\Desktop\\Health care\\train_data.csv")

# Clean the data
df = clean_data(df)

# Preprocess the data
df = preprocess_data(df)

# Save the processed data
save_processed_data(df, "C:\\Users\\91800\\Desktop\\Health care\\train_data.csv")