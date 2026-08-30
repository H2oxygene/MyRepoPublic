# Étape 1 : on importe la fonction "pipeline" de la librairie Hugging Face
# "pipeline" sert à simplifier l'utilisation des modèles de langage
from transformers import pipeline

# Étape 2 : on crée un texte de départ
# C'est ce qu'on appelle le "prompt"
# Le modèle va essayer de compléter ce texte
prompt = "Écris une courte introduction sur l'intelligence artificielle :"

# Étape 3 : on construit un générateur de texte
# "text-generation" = on demande à un modèle de générer du texte
# "distilgpt2" = un petit modèle léger, facile à tester
# On utilise aussi le tokenizer associé au même modèle
generator2 = pipeline(
    "text-generation",
    model="distilgpt2",
    tokenizer="distilgpt2"
)

# Étape 4 : on demande au modèle de générer une réponse
# max_new_tokens = combien de mots / morceaux de texte il peut inventer
# do_sample = active la génération plus naturelle et variée
# temperature = règle le niveau de créativité (plus c'est haut, plus c'est libre)
# num_return_sequences = combien de réponses on veut
result = generator2(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.9,
    num_return_sequences=1
)

# Étape 5 : on affiche le résultat
# result est une liste ; result[0] = la première réponse
# ["generated_text"] = le texte que le modèle a produit
print(result[0]["generated_text"])