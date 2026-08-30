# Test 2 : modèle plus orienté conversation
# On utilise un modèle conçu pour des dialogues : DialoGPT
# C'est plus adapté qu'un modèle de complétion simple pour ce type de prompt.

from transformers import pipeline

# Prompt orienté conversation, en français
prompt = "Utilisateur : Bonjour, peux-tu me dire en français ce qu'est un LLM ?\nAssistant :"

# On charge un modèle de dialogue léger et assez fiable pour débuter
# Il est plus adapté aux échanges qu'un simple GPT-2 de base.
generator = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-small",
    tokenizer="microsoft/DialoGPT-small"
)

# On génère une réponse courte mais utile
response = generator(
    prompt,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    num_return_sequences=1,
    pad_token_id=generator.tokenizer.eos_token_id
)

# On affiche seulement le texte généré
print(response[0]["generated_text"])
