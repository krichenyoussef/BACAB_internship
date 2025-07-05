from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from config.config import AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME
import os

def upload_file_to_blob(file):
    # Connexion à Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
    
    # Créer un client Blob pour le fichier à uploader
    blob_client = container_client.get_blob_client(file.filename)
    
    # Upload du fichier
    with open(file.filename, "wb") as f:
        f.write(file.read())
    
    with open(file.filename, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    # Supprimer le fichier local après upload
    os.remove(file.filename)
