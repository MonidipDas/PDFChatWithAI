---
description: Always verify LLM model names before hardcoding or proposing them.
---

# Verify LLM Model Names

When suggesting, reviewing, or hardcoding LLM model names (e.g., for Groq, OpenAI, Anthropic, or HuggingFace):
1. **Never assume a model is valid** just because it was used in the past, as models are frequently deprecated or renamed (e.g., `llama-3.1-8b-instant` vs `llama3-8b-8192`).
2. **Verify validity**: Proactively check if the models are currently active. If an API key is available, query the provider's `/models` endpoint to fetch the active list of models. If an API key is not available, check the provider's public documentation or ask the user to verify the exact model string.
3. **Graceful Fallbacks**: When hardcoding models, ensure the application code provides a graceful fallback or dynamically fetches the available models if the default fails.
