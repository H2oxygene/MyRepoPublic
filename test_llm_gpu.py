# Test 6 : inférence sur GPU pour bien plus de rapidité
# La GPU est 10-100x plus rapide que le CPU pour les LLMs

import torch
from transformers import pipeline
import time

# Vérifier si une GPU est disponible
print("="*60)
print("VÉRIFICATION DE LA GPU")
print("="*60)
print(f"GPU disponible : {torch.cuda.is_available()}")
print(f"Nombre de GPUs : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU utilisée : {torch.cuda.get_device_name(0)}")
    print(f"Mémoire GPU : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  Aucune GPU détectée. L'inférence se fera sur CPU.")
print()

# Prompt simple
prompt = "Explique-moi ce qu'est un Large Language Model (LLM) dans le domaine de l'intelligence artificielle. Un LLM est un modèle de langue basé sur le machine learning. Réponds en 2-3 phrases simples."

# Charger le modèle avec device_map="auto" pour utiliser la GPU si disponible
print("="*60)
print("CHARGEMENT DU MODÈLE")
print("="*60)
start_load = time.time()

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto"  # ← Force l'utilisation de la GPU si disponible
)

load_time = time.time() - start_load
print(f"Temps de chargement : {load_time:.2f}s")
print()

# Génération avec mesure de temps
print("="*60)
print("GÉNÉRATION DE TEXTE")
print("="*60)
start_gen = time.time()

response = generator(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    num_return_sequences=1
)

gen_time = time.time() - start_gen
print(f"Temps de génération : {gen_time:.2f}s")
print()

# Affichage du résultat
print("="*60)
print("RÉPONSE")
print("="*60)
print(response[0]["generated_text"])
print()

# Résumé des perfs
print("="*60)
print("RÉSUMÉ DES PERFORMANCES")
print("="*60)
if torch.cuda.is_available():
    print(f"✅ Inférence sur GPU : {gen_time:.2f}s")
    print("(Avec CPU, ce serait ~10-100x plus lent)")
else:
    print(f"⚠️  Inférence sur CPU : {gen_time:.2f}s")
    print("Si tu as une GPU NVIDIA, installe CUDA pour accélérer")
