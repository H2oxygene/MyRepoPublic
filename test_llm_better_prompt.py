# Test 4 : prompt mieux défini pour éviter les ambiguïtés
# On clarifie dès le départ qu'on parle d'un Large Language Model
# et pas d'un diplôme ou autre

from transformers import pipeline

# Prompt plus clair et contextualisé pour éviter la confusion
prompt = """Explique-moi ce qu'est un Large Language Model (LLM) dans le domaine de l'intelligence artificielle.
Un LLM est un modèle de langue basé sur le machine learning. Réponds en 2-3 phrases simples."""

# On utilise le même modèle puissant que le test 3
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct"
)

# Génération optimisée
response = generator(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    num_return_sequences=1
)

# Affichage du résultat
print("Prompt :")
print(prompt)
print("\n" + "="*60)
print("Réponse du modèle :")
print("="*60)
print(response[0]["generated_text"])
