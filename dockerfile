# Utilise une image officielle Python
FROM python

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .

# Installer les dépendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le reste du code
COPY . .

# Exposer le port utilisé par Flask
EXPOSE 5000

# Commande pour lancer le serveur Flask
CMD ["python", "main.py"]
