# Test 5 : augmenter max_new_tokens pour laisser le modèle terminer ses phrases
# La limite précédente (150) était trop courte

from transformers import pipeline

# Prompt clair et explicite
prompt = """Explique-moi ce qu'est un Large Language Model (LLM) dans le domaine de l'intelligence artificielle.
Un LLM est un modèle de langue basé sur le machine learning. Réponds en 2-3 phrases simples."""

# On utilise le même modèle
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct"
)

# Augmentation de max_new_tokens à 250 pour laisser le modèle finir
response = generator(
    prompt,
    max_new_tokens=250,  # Augmenté de 150 à 250
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
