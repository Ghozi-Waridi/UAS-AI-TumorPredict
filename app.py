from flask import Flask
from flask_cors import CORS
from api.routes import routes
# from .routes import routes
import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Izinkan CORS dari localhost:8000
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})

# Atur route
app.register_blueprint(routes)

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'upload')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if __name__ == "__main__":
    app.run(debug=True, port=5009, host='0.0.0.0')
