from flask import Blueprint, request, jsonify
from datasource.azure_blob import upload_file_to_blob
import os

api_bp = Blueprint('api', __name__)

# Endpoint pour l'upload de fichiers
@api_bp.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')  # Récupérer le fichier de la requête
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    # Sauvegarder le fichier dans Azure Blob Storage
    try:
        upload_file_to_blob(file)
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
