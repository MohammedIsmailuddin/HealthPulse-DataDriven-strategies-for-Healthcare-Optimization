import os
from Src.data_preprocessing import load_data, clean_data, preprocess_data, save_processed_data # type: ignore
from Src.models import train_model # type: ignore

def main():
    # Paths
    raw_data_path = 'data/raw/health_data.csv'
    processed_data_path = 'data/processed/processed_data.csv'
    
    # Load and preprocess data
    df = load_data(raw_data_path)
    df = clean_data(df)
    df = preprocess_data(df)
    save_processed_data(df, processed_data_path)
    
    # Load processed data for modeling
    df = load_data(processed_data_path)
    X = df.drop('target_column', axis=1)
    y = df['target_column']
    
    # Train model
    model, accuracy = train_model(X, y)
    print(f'Model Accuracy: {accuracy}')
    
if __name__ == "_main_":
    main()