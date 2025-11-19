"""
model_runner.py
---------------
Handles prompting local LLMs through the Ollama CLI.

Uses:
    ollama run <model>

Notes:
 - The prompt is passed through STDIN to avoid shell escaping issues.
 - Compatible with all modern Ollama versions (including 0.12+).
 - Returns clean error messages instead of crashing the app.
"""

import subprocess
from typing import Optional

# Default model name (can be changed from app UI)
MODEL = "mistral"


def run_ollama_query(prompt_text: str, model_name: Optional[str] = None, timeout: int = 120) -> str:
    """
    Run a prompt through the local Ollama LLM.
    
    Parameters
    ----------
    prompt_text : str
        Full prompt to send to the model.
    
    model_name : Optional[str]
        Name of the Ollama model to use. Defaults to global MODEL.
    
    timeout : int
        Max seconds before the CLI call is force-terminated.
    
    Returns
    -------
    str
        The model output, or formatted error message.
    """

    # Determine model to use
    model = model_name if model_name else MODEL

    # New Ollama syntax uses:  ollama run <model>
    cmd = ["ollama", "run", model]

    try:
        # Send prompt via stdin so no need for escapes or temp files
        proc = subprocess.run(
            cmd,
            input=prompt_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        # If model error / CLI error
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            return f"[Model error] {err or 'Unknown error'}"

        # Sometimes stderr is empty even when output exists (normal)
        output = (proc.stdout or "").strip()
        if not output:
            return "[Model error] Empty output from model."

        return output

    except FileNotFoundError:
        return "[Error] Ollama CLI not found — is Ollama installed and added to PATH?"

    except subprocess.TimeoutExpired:
        return "[Error] Model timed out."

    except Exception as e:
        return f"[Error] {str(e)}"
