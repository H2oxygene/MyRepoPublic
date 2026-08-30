# ========================================
# TEST LLM HUGGING FACE POUR GOOGLE COLAB
# ========================================
# Ce script est conçu pour tourner sur Google Colab
# Colab fournit une GPU T4 gratuite !
#
# Instructions :
# 1. Va sur colab.research.google.com
# 2. Crée un nouveau notebook
# 3. Copie ce code dans une cellule
# 4. Configure la GPU : Runtime > Change runtime type > GPU
# 5. Exécute !

# ========================================
# ÉTAPE 1 : Installer les dépendances
# ========================================
print("Installation des packages...")
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "torch"])
print("✅ Packages installés !")
print()

# ========================================
# ÉTAPE 2 : Vérifier la GPU
# ========================================
import torch
print("="*60)
print("VÉRIFICATION DE LA GPU")
print("="*60)
print(f"GPU disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU utilisée : {torch.cuda.get_device_name(0)}")
    print(f"Mémoire GPU : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✅ Tu as une GPU gratuite ! (T4 ou A100)")
else:
    print("⚠️  GPU non détectée. Reconfigure le runtime !")
print()

# ========================================
# ÉTAPE 3 : Charger le modèle LLM
# ========================================
from transformers import pipeline
import time

print("="*60)
print("CHARGEMENT DU MODÈLE")
print("="*60)
start = time.time()

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto"  # Utilise automatiquement la GPU
)

load_time = time.time() - start
print(f"✅ Modèle chargé en {load_time:.2f}s")
print()

# ========================================
# ÉTAPE 4 : Générer du texte avec le LLM
# ========================================
print("="*60)
print("GÉNÉRATION DE TEXTE")
print("="*60)

# Prompt en français
prompt = "Explique-moi ce qu'est un Large Language Model (LLM) dans le domaine de l'intelligence artificielle. Réponds en 2-3 phrases simples en français."

print(f"Prompt : {prompt}\n")

start = time.time()
response = generator(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    num_return_sequences=1
)
gen_time = time.time() - start

print("="*60)
print("RÉPONSE DU MODÈLE")
print("="*60)
print(response[0]["generated_text"])
print()

# ========================================
# ÉTAPE 5 : Résumé des performances
# ========================================
print("="*60)
print("RÉSUMÉ")
print("="*60)
print(f"⏱️  Temps de génération : {gen_time:.2f}s")
if torch.cuda.is_available():
    print("✅ Exécuté sur GPU (rapide !)")
else:
    print("⚠️  Exécuté sur CPU (lent)")
print()

# ========================================
# ÉTAPE 6 : Tester d'autres prompts
# ========================================
print("="*60)
print("TEST SUPPLÉMENTAIRE : LLM EN CONVERSATION")
print("="*60)

prompt2 = "Qui es-tu ? Réponds brièvement."
print(f"Prompt : {prompt2}\n")

response2 = generator(
    prompt2,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    num_return_sequences=1
)

print("Réponse :")
print(response2[0]["generated_text"])
