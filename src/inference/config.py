from enum import Enum
import sqlite3

class Normalization(Enum):
    No = 1
    MinMax = 2
    MeanStd = 3

class Config:
    def __init__(self, db_path='model_config.db'):
        # Load config from SQLite
        self.load_from_db(db_path)
        
        # Additional inference-specific settings
        self.buffer = []
        self.buffer_count = 2
        self.model_type = 'default'
        self.model = "2state_features_labels_W180_O90_WAMPth20.tflite"
        self.file_path = 'src/inference/output.csv'
    
    def load_from_db(self, db_path):
        """Load configuration from SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT parameter, value, data_type FROM config')
        for param, value, dtype in cursor.fetchall():
            if dtype == 'int':
                setattr(self, param, int(value))
            elif dtype == 'float':
                setattr(self, param, float(value))
            elif dtype == 'list':
                setattr(self, param, value.split(','))
            else:
                setattr(self, param, value)
        
        conn.close()
