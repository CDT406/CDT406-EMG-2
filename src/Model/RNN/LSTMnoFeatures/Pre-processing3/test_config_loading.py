import os
import sqlite3
from pprint import pprint

def load_combined_config(db_path):
    """Load both preprocessing config and feature stats from single SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Load preprocessing config
    config = {}
    cursor.execute('SELECT parameter, value, data_type FROM preprocessing_config')
    for param, value, dtype in cursor.fetchall():
        if dtype == 'int':
            config[param] = int(value)
        elif dtype == 'float':
            config[param] = float(value)
        elif dtype == 'list':
            config[param] = value.split(',')
        else:
            config[param] = value
    
    # Load feature stats
    feature_stats = {}
    cursor.execute('SELECT feature_name, mean, std FROM feature_stats')
    for feature, mean, std in cursor.fetchall():
        feature_stats[feature] = {'mean': float(mean), 'std': float(std)}
    
    conn.close()
    return config, feature_stats

def main():
    # Update path to point to Saved_models directory
    base_dir = os.path.join('src', 'Model', 'RNN', 'LSTM2', 'Saved_models')
    config_db = os.path.join(base_dir, 'preprocessing_config.db')
    
    # Print path for debugging
    print(f"Looking for config at: {os.path.abspath(config_db)}")
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Creating directory: {os.path.abspath(base_dir)}")
        os.makedirs(base_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(config_db):
        print(f"Error: Config database not found at: {os.path.abspath(config_db)}")
        return
    
    # Load and display all configuration
    print("\nLoading configuration and statistics...")
    config, feature_stats = load_combined_config(config_db)
    
    print("\nPreprocessing Configuration:")
    pprint(config)
    
    print("\nFeature Statistics:")
    pprint(feature_stats)

if __name__ == '__main__':
    main()