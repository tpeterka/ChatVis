# Provider-agnostic LLM chat client for the ChatVis agent.
#
# Configuration is read from environment variables:
#   LLM_PROVIDER  - one of "openai", "anthropic", "gemini", "llama" (default: "openai")
#   API keys are provider-specific:
#     openai      - OPENAI_API_KEY (required)
#     anthropic   - resolved by the SDK: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
#                   or an `ant auth login` OAuth profile
#     gemini      - GEMINI_API_KEY (required)
#     llama       - LLAMA_API_KEY (optional; local servers accept any key)
#   LLM_MODEL     - model name override; each provider has a built-in default,
#                   except "llama" which requires an explicit model
#   LLM_BASEURL   - API endpoint override for OpenAI-compatible providers
#                   (openai/gemini/llama); useful when pointing at a local
#                   Llama server (e.g. ollama at http://localhost:11434/v1).
#                   Ignored for "anthropic" — use ANTHROPIC_BASE_URL instead.

import os

PROVIDERS = ("openai", "anthropic", "gemini", "llama")

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-8",
    "gemini": "gemini-2.5-flash",
    # "llama" has no default; LLM_MODEL is required
}

DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "llama": "http://localhost:11434/v1",
}


API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "llama": "LLAMA_API_KEY",
    # "anthropic" has no entry: the anthropic SDK resolves credentials itself
}


class LLMClient:
    """Chat with the LLM selected via LLM_PROVIDER/LLM_MODEL/LLM_BASEURL."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. Valid values: {', '.join(PROVIDERS)}"
            )

        self.model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(self.provider)
        if not self.model:
            raise ValueError(
                f"LLM_MODEL must be set for provider '{self.provider}' "
                "(e.g. the model name served by your local Llama/ollama server)"
            )

        key_var = API_KEY_VARS.get(self.provider)
        api_key = os.getenv(key_var) if key_var else None
        if not api_key and self.provider == "llama":
            # Local OpenAI-compatible servers (ollama, llama.cpp, vLLM)
            # accept any non-empty key
            api_key = "ollama"
        if not api_key and self.provider not in ("anthropic", "llama"):
            raise ValueError(f"{key_var} must be set for provider '{self.provider}'")

        if self.provider == "anthropic":
            import anthropic

            # Credentials and endpoint are resolved by the SDK itself
            # (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`
            # profile, and ANTHROPIC_BASE_URL for a custom endpoint).
            # LLM_BASEURL is deliberately ignored here so a value meant for
            # an OpenAI-compatible server can't misroute Anthropic requests.
            self._client = anthropic.Anthropic()
        else:
            import openai

            base_url = os.getenv("LLM_BASEURL") or DEFAULT_BASE_URLS.get(self.provider)
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, system_prompt, user_prompt):
        """Send one system+user exchange and return the assistant's text."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
