Sur le script worker.py il y a une partie d'appel llm qui récuper un model Mistral telechargé sur le dossier avec la commande:

huggingface-cli download TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q4_K_M.gguf --local-dir ./models/

Avant il faut appeler ces commandes:

pip install langchain ctransformers langchain_community chromadb

pip install --upgrade langchain langchain-core langchain-community

pip install -U sentence-transformers

