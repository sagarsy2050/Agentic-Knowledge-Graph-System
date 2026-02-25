import json
import time
import httpx
from typing import List, Dict, Optional
from loguru import logger
from config.settings import CONFIG


class OllamaEngine:

    def __init__(self):
        self.base_url = CONFIG.ollama.base_url
        self.primary_model = CONFIG.ollama.primary_model
        self.embedding_model = CONFIG.ollama.embedding_model
        self.timeout = CONFIG.ollama.timeout

    def check_connection(self):
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.success(f"Ollama connected. Models: {models}")
                return True
        except Exception as e:
            logger.error(f"Ollama not reachable: {e}")
        return False

    def list_models(self):
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            return [m["name"] for m in r.json().get("models", [])]
        except:
            return []

    def pull_model(self, model_name):
        logger.info(f"Pulling model: {model_name}")
        try:
            with httpx.stream("POST", f"{self.base_url}/api/pull",
                              json={"name": model_name}, timeout=600) as r:
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"\r{data['status']}", end="", flush=True)
            return True
        except Exception as e:
            logger.error(f"Pull failed: {e}")
            return False

    def generate(self, prompt, model=None, system=None, temperature=0.1, max_tokens=4096):
        model = model or self.primary_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        if system:
            payload["system"] = system
        try:
            r = httpx.post(f"{self.base_url}/api/generate",
                           json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json().get("response", "")
            logger.error(f"Generate error {r.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Generate failed: {e}")
            return ""

    def chat(self, messages, model=None, temperature=0.1, max_tokens=4096):
        model = model or self.primary_model
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat",
                           json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()["message"]["content"]
            return ""
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return ""

    def embed(self, text, model=None):
        model = model or self.embedding_model
        try:
            r = httpx.post(f"{self.base_url}/api/embeddings",
                           json={"model": model, "prompt": text}, timeout=60)
            if r.status_code == 200:
                return r.json().get("embedding", [])
            return []
        except Exception as e:
            logger.error(f"Embed failed: {e}")
            return []

    def extract_json(self, prompt, schema_hint=""):
        system = f"""You are a precise data extractor. Always respond with valid JSON only.
No explanation, no markdown, just raw JSON.
{f'Expected schema: {schema_hint}' if schema_hint else ''}"""
        response = self.generate(prompt, system=system, temperature=0.0)
        response = response.strip()
        for tag in ["```json", "```"]:
            if response.startswith(tag):
                response = response[len(tag):]
        if response.endswith("```"):
            response = response[:-3]
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            match2 = re.search(r'\[.*\]', response, re.DOTALL)
            if match2:
                try:
                    return json.loads(match2.group())
                except:
                    pass
            return {}


LLM = OllamaEngine()
