# Test 3 : modèle plus puissant et orienté instruction
# On utilise un modèle "instruct" conçu pour suivre des instructions
# Qwen2.5-0.5B-Instruct est léger mais bien plus puissant que les précédents

from transformers import pipeline

# Prompt simple et direct en français
prompt = "Explique-moi en français ce qu'est un LLM en 2-3 phrases simples."

# On charge un modèle orienté instruction et capable de générer du texte cohérent
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct"
)

# Génération avec paramètres optimisés pour ce modèle
response = generator(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    num_return_sequences=1
)

# On affiche le résultat
print("Prompt : ", prompt)
print("\nRéponse du modèle :")
print(response[0]["generated_text"])
