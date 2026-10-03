"""Evaluation-only runner for any /v1/systemone endpoint: Jev (TypeSafe), laya-serve, Kev, openjev.

    compare:
      runners:
        - {backend: systemone, name: jev, url: https://api.typesafe.ai/v1/systemone,
           model: jev-latest, api_key_env: TYPESAFE_API_KEY}
"""
import json
import os
import urllib.request


class SystemOneRunner:
    def __init__(self, cfg):
        self.url, self.model = cfg["url"], cfg.get("model")
        self.key = os.environ.get(cfg.get("api_key_env", ""), "")
        self.timeout = cfg.get("timeout", 60)
        self.name = cfg.get("name", self.model or self.url)

    def predict(self, state, questions, **_):
        if isinstance(state, dict) and any(k in state for k in ("image", "video", "audio")):
            raise ValueError(f"{self.name}: systemone endpoints are text-only; preprocess media first")
        body = {"state": state, "questions": questions, **({"model": self.model} if self.model else {})}
        headers = {"content-type": "application/json", **({"authorization": f"Bearer {self.key}"} if self.key else {})}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)
