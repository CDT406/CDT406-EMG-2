import sqlite3


class Logger:
    
    def __init__(self, path='output/sqlite.db'):
        self.con = sqlite3.connect(path)
        self.cursor = self.con.cursor()


    def __del__(self):
        self.con.close()


    def log_data(self, input_data, output_data):
        # Create table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_data TEXT,
                output_data TEXT
            )
        ''')
        # Insert the input and output data as text
        self.cursor.execute(
            'INSERT INTO logs (input_data, output_data) VALUES (?, ?)',
            (str(input_data), str(output_data))
        )
        self.con.commit()