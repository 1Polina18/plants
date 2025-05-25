import os
import sqlite3
import logging
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, HiddenField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from flask_wtf import FlaskForm
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['WTF_CSRF_SECRET_KEY'] = os.environ.get('WTF_CSRF_SECRET_KEY')
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_TIME_LIMIT'] = 1800
app.config['FLASK_LIMITER_ENABLED'] = True
app.config['FLASK_LIMITER_DEFAULT_LIMITS'] = ["200 per day", "50 per hour"]
app.config['FLASK_LIMITER_KEY_FUNC'] = get_remote_address
app.config['FLASK_LIMITER_STORAGE_URI'] = "memory://"
DATABASE = 'plants.db'

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

limiter = Limiter(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email, password_hash):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    
    if user_data:
        return User(*user_data)
    return None

# Форма регистрации с защитой CSRF
class RegistrationForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=4, max=15)])
    email = StringField('Электронная почта', validators=[DataRequired(), Email()])
    password = PasswordField('Пароль', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Подтвердите пароль', validators=[DataRequired(), EqualTo('password', message='Пароли не совпадают')])
    csrf_token = HiddenField()

# Форма входа с защитой CSRF
class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    email = StringField('Электронная почта', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    csrf_token = HiddenField()

# Блокировка неудачных попыток входа
login_attempts = {}

def failed_login_limit(username):
    if username not in login_attempts:
        login_attempts[username] = 0
    login_attempts[username] += 1
    if login_attempts[username] > 5:
        return True
    return False

def reset_login_attempts(username):
    if username in login_attempts:
        del login_attempts[username]

# Декоратор для ограничения скорости (общий, для защиты)
def limit_login_attempts(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        username = request.form.get('username')
        if username and failed_login_limit(username):
            flash('Слишком много неудачных попыток входа. Пожалуйста, попробуйте позже.')
            form = LoginForm()
            logging.warning(f'Пользователь {username} превысил лимит попыток входа.')
            return render_template('login.html', form=form)
        return func(*args, **kwargs)
    return wrapper

@app.route('/')
def home():
    return render_template('home.html', username=current_user.username if current_user.is_authenticated else None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm(request.form)
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        existing_email = cursor.fetchone()
        
        if existing_user:
            flash('Имя пользователя уже занято')
            logging.warning(f'Попытка регистрации с занятым именем пользователя: {username}')
            return render_template('register.html', form=form)
        
        if existing_email:
            flash('Электронная почта уже занята')
            logging.warning(f'Попытка регистрации с занятой электронной почтой: {email}')
            return render_template('register.html', form=form)

        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", (username, email, hashed_password))
        conn.commit()

        user_id = cursor.lastrowid
        conn.close()

        user = User(user_id, username, email, hashed_password)
        login_user(user)
        logging.info(f'Пользователь {username} зарегистрирован и вошел в систему.')
        return redirect(url_for('index'))
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
@limit_login_attempts
def login():
    form = LoginForm(request.form)
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        conn.close()

        if user_data and check_password_hash(user_data[3], password):
            if user_data[2] == email:
                user = User(*user_data)
                login_user(user)
                reset_login_attempts(username)
                logging.info(f'Пользователь {username} успешно вошел в систему.')
                return redirect(url_for('index'))
            else:
                flash('Неверная электронная почта для этого пользователя')
                logging.warning(f'Неверная электронная почта для пользователя {username}.')
        else:
            flash('Неверное имя пользователя или пароль')
            logging.warning(f'Неудачная попытка входа для пользователя {username}.')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    logging.info(f'Пользователь {username} вышел из системы.')
    return redirect(url_for('home'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html', username=current_user.username)

@app.errorhandler(429)
def ratelimit_handler(e):
    logging.warning('Слишком много запросов.')
    return "Слишком много запросов. Пожалуйста, попробуйте позже.", 429 

# Конфигурация
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
MODEL_PATH = 'plant_recognition_model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','jfif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Фильтр для получения имени файла
@app.template_filter('basename')
def get_basename(path):
    return os.path.basename(path)

# Загрузка модели машинного обучения
try:
    model = load_model(MODEL_PATH)
    print("Модель успешно загружена.")
except Exception as e:
    logging.error(f"Ошибка при загрузке модели: {e}")
    model = None

# Функция для предварительной обработки изображения
def preprocess_image(image):
    try:
        img = image.resize((150, 150))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        logging.error(f"Ошибка при предварительной обработке изображения: {e}")
        return None

# Функция для предсказания класса растения
def predict_plant(image):
    try:
        if model is None:
            logging.error("Модель не загружена.")
            return "Модель не загружена."

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)

        class_names = os.listdir(DATA_FOLDER)
        if not class_names:
            logging.error("Не удалось получить названия классов растений. Проверьте структуру папок 'data'.")
            return "Не удалось распознать растение. Попробуйте загрузить другое изображение."

        try:
            predicted_class_name = class_names[predicted_class_index]
        except IndexError as e:
            logging.exception(f"Индекс класса вне диапазона: {predicted_class_index}. Проверьте соответствие между моделью и данными.")
            return "Не удалось распознать растение. Пожалуйста, проверьте изображение и повторите попытку."

        return predicted_class_name

    except Exception as e:
        logging.exception("Произошла непредвиденная ошибка в predict_plant.")
        return "Произошла непредвиденная ошибка при распознавании растения."

# Функция для получения информации о растении
def get_plant_info(plant_name):
    conn = None
    plant_info = {}

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM plants WHERE name = ?", (plant_name,))
        plant_data = cursor.fetchone()

        if plant_data:
            column_names = [col[0] for col in cursor.description]
            plant_info = dict(zip(column_names, plant_data))
            return plant_info
        else:
            plant_info['name'] = plant_name
            plant_info['description'] = "Информация о растении не найдена в базе данных."

    except sqlite3.Error as e:
        logging.error(f"Ошибка при запросе к базе данных: {e}")
        plant_info['name'] = "Ошибка."
        plant_info['description'] = f"Ошибка при чтении из базы данных: {e}"

    except Exception as e:
        logging.exception("Непредвиденная ошибка при получении информации о растении:")
        plant_info['name'] = "Ошибка."
        plant_info['description'] = f"Непредвиденная ошибка: {e}"

    finally:
        if conn:
            try:
                conn.close()
            except sqlite3.Error as e:
                logging.error(f"Ошибка при закрытии соединения с базой данных: {e}")
    return plant_info

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Маршруты Flask
@app.route('/result', methods=['GET'])
@login_required
def result():
    plant_name = request.args.get('plant_name')
    image_path = request.args.get('image_path')
    from_favorites = request.args.get('from_favorites')

    if plant_name:
        plant_info = get_plant_info(plant_name)
        return render_template('result.html', plant=plant_info, image_path=image_path, username=current_user.username, from_favorites=from_favorites)
    else:
        return redirect(url_for('index'))
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file') or request.files.get('fileMobile')

        if file is None:
            return render_template('index.html', message='Нет файла в запросе.')

        if file.filename == '':
            return render_template('index.html', message='Файл не выбран.')

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                image = Image.open(file_path)
                predicted_plant_name = predict_plant(image)

                return redirect(url_for('result', plant_name=predicted_plant_name, image_path=filename))

            except Exception as e:
                logging.error(f'Ошибка обработки изображения: {e}')
                return render_template('index.html', message=f'Ошибка обработки изображения. Пожалуйста, попробуйте загрузить другое изображение.')

        else:
            return render_template('index.html', message='Недопустимый формат файла. Разрешены только PNG, JPG, JPEG, JFIF.')

    return render_template('index.html', message='')

def get_favorite_plants(user_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT plant_id FROM favorite_plants WHERE user_id = ?", (user_id,))
        favorite_plant_ids = cursor.fetchall()
    except Exception as e:
        logging.error(f"Ошибка при получении избранных растений для пользователя {user_id}: {e}")
        return [] 
    finally:
        conn.close()

    return [plant_id[0] for plant_id in favorite_plant_ids]

@app.route('/favorites')
@login_required
def favorites():
    user_id = current_user.id
    favorite_plant_ids = get_favorite_plants(user_id)

    plants_info = []
    if favorite_plant_ids:
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plants WHERE id IN ({})".format(','.join('?' * len(favorite_plant_ids))), favorite_plant_ids)
            plants_info = cursor.fetchall()
        except Exception as e:
            logging.error(f"Ошибка при получении избранных растений для пользователя {user_id}: {e}")
            flash('Произошла ошибка при загрузке избранных растений.', 'error')
        finally:
            conn.close()

    return render_template('favorites.html', username=current_user.username, plants=plants_info)

@app.route('/add_to_favorites', methods=['POST'])
@login_required
def add_to_favorites():
    plant_id = request.form.get('plant_id')
    user_id = current_user.id

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM favorite_plants WHERE user_id = ? AND plant_id = ?", (user_id, plant_id))
        existing_favorite = cursor.fetchone()

        if not existing_favorite:
            cursor.execute("INSERT INTO favorite_plants (user_id, plant_id) VALUES (?, ?)", (user_id, plant_id))
            conn.commit()
            flash('Растение добавлено в избранное!')
        else:
            flash('Это растение уже есть в вашем избранном!')

    except Exception as e:
        logging.error(f"Ошибка при добавлении растения {plant_id} в избранное для пользователя {user_id}: {e}")
        flash('Произошла ошибка при добавлении растения в избранное.', 'error')
    
    finally:
        conn.close()
    return redirect(request.referrer or url_for('result'))

@app.route('/remove_from_favorites', methods=['POST'])
@login_required
def remove_from_favorites():
    plant_id = request.form.get('plant_id')
    user_id = current_user.id

    try:
        plant_id = int(plant_id)
    except (ValueError, TypeError):
        logging.error(f"Ошибка преобразования идентификатора растения: {e}")
        flash('Ошибка: неверный идентификатор растения.', 'error')
        return redirect(url_for('favorites'))

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM favorite_plants WHERE user_id = ? AND plant_id = ?", (user_id, plant_id))
        conn.commit()

    except Exception as e:
        logging.error(f"Ошибка при удалении растения {plant_id} из избранного для пользователя {user_id}: {e}")
        flash('Произошла ошибка при удалении растения из избранного.', 'error')

    finally:
        conn.close()
    return redirect(url_for('favorites'))


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/send_data/<plant_name>/<filename>')
def send_data(plant_name, filename):
    return send_from_directory(os.path.join('data', plant_name), filename)

if __name__ == '__main__':
    app.run(debug=True)
