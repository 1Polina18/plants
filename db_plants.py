import sqlite3
import csv

DATABASE = 'plants.db' 
PLANTS_INFO_CSV = 'plants_info.csv'

def create_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

def create_tables(conn):
    cursor = conn.cursor()
    
    # Создание таблицы растений
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        lighting TEXT,
        watering TEXT,
        fertilizing TEXT,
        temperature TEXT,
        humidity TEXT,
        transplanting TEXT,
        soil_type TEXT,
        propagation TEXT,
        common_pests TEXT,
        toxicity_to_pets TEXT,
        blooming_season TEXT,
        pruning TEXT,
        special_care TEXT,
        diseases_description TEXT,
        diseases_prevention TEXT,
        diseases_treatment TEXT
    )
    ''')
    
    # Создание таблицы пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL
    )
    ''')

    # Создание таблицы избранных растений
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS favorite_plants (
        user_id INTEGER NOT NULL,
        plant_id INTEGER NOT NULL,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, plant_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
    )
    ''')
    
    conn.commit()

def insert_plants_from_csv(conn):
    cursor = conn.cursor()
    try:
        with open(PLANTS_INFO_CSV, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            plants_data = [
                (
                    row['name'],
                    row['description'],
                    row['lighting'],
                    row['watering'],
                    row['fertilizing'],
                    row['temperature'],
                    row['humidity'],
                    row['transplanting'],
                    row['soil_type'],
                    row['propagation'],
                    row['common_pests'],
                    row['toxicity_to_pets'],
                    row['blooming_season'],
                    row['pruning'],
                    row['special_care'],
                    row['diseases_description'],
                    row['diseases_prevention'],
                    row['diseases_treatment']
                )
                for row in reader
            ]
            
            cursor.executemany('''
            INSERT INTO plants (
                name, description, lighting, watering, fertilizing, temperature, 
                humidity, transplanting, soil_type, propagation, common_pests, 
                toxicity_to_pets, blooming_season, pruning, special_care, 
                diseases_description, diseases_prevention, diseases_treatment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', plants_data)
        
        conn.commit()
        print("Данные о растениях успешно добавлены в базу данных.")

    except FileNotFoundError:
        print(f"Ошибка: Файл {PLANTS_INFO_CSV} не найден.")
    except Exception as e:
        print(f"Ошибка при вставке данных из CSV: {e}")

def main():
    conn = create_connection()
    if conn is not None:
        create_tables(conn)
        insert_plants_from_csv(conn)
        conn.close()
        print("База данных успешно создана/обновлена.")
    else:
        print("Не удалось установить соединение с базой данных.")

if __name__ == "__main__":
    main()