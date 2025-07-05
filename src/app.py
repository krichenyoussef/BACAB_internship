from flask import Flask
from controllers.api import api_bp  # Assurez-vous que ce module existe et est correct

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix='/api')  # Le préfixe `/api` pour toutes les routes du blueprint
if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Flask will now run on http://127.0.0.1:8000
