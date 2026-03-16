"""AWS Bedrock client for Amazon Nova foundation models."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

import boto3
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class NovaClient:
    """Wrapper around AWS Bedrock for Nova model invocations."""

    def __init__(self) -> None:
        self._region = os.getenv("AWS_REGION", "us-east-1")
        self._lite_model = os.getenv(
            "NOVA_LITE_MODEL_ID", "us.amazon.nova-2-lite-v1:0"
        )
        self._embed_model = os.getenv(
            "NOVA_EMBED_MODEL_ID", "amazon.nova-2-multimodal-embeddings-v1:0"
        )
        self._sonic_model = os.getenv(
            "NOVA_SONIC_MODEL_ID", "us.amazon.nova-2-sonic-v1:0"
        )
        self._client = boto3.client(
            "bedrock-runtime", region_name=self._region
        )

    def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Invoke Nova 2 Lite for structured reasoning / decision synthesis."""
        body = {
            "messages": [
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        try:
            response = self._client.invoke_model(
                modelId=self._lite_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result["output"]["message"]["content"][0]["text"]
        except Exception as exc:
            logger.warning("Nova reason call failed, using fallback: %s", exc)
            return self._fallback_reason(system_prompt, user_prompt)

    def reason_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_format: str = "jpeg",
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Invoke Nova 2 Lite with an image for multimodal vision analysis."""
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": encoded},
                            }
                        },
                        {"text": user_prompt},
                    ],
                },
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        try:
            response = self._client.invoke_model(
                modelId=self._lite_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result["output"]["message"]["content"][0]["text"]
        except Exception as exc:
            logger.warning("Nova vision call failed: %s", exc)
            return json.dumps({
                "similarity_score": 0.5,
                "anomalies": ["vision analysis unavailable"],
                "description": "Could not analyze image via Nova Lite vision.",
            })

    def reason_with_image_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_format: str = "jpeg",
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Invoke Nova 2 Lite with an image and parse the response as JSON."""
        raw = self.reason_with_image(
            system_prompt,
            user_prompt + "\n\nRespond ONLY with valid JSON.",
            image_bytes,
            image_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Nova vision JSON")
            return {"similarity_score": 0.5, "anomalies": [], "description": cleaned}

    def generate_voice_escalation_script(
        self, case_summary: str
    ) -> str:
        """Generate a structured voice escalation script using Nova Lite."""
        system_prompt = (
            "You are a voice escalation script generator for Amazon's return "
            "fraud operations center. Generate a professional, empathetic phone "
            "script that an agent (powered by Nova Sonic) would use to call a "
            "customer about a flagged return. Include: greeting, verification "
            "questions, explanation of concern, resolution options, and closing. "
            "Format with clear sections: GREETING, VERIFICATION, CONCERN, "
            "RESOLUTION, CLOSING."
        )
        try:
            return self.reason(
                system_prompt,
                f"Generate a voice escalation script for this case:\n{case_summary}",
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning("Voice script generation failed: %s", exc)
            return self._fallback_voice_script(case_summary)

    @staticmethod
    def _fallback_voice_script(case_summary: str) -> str:
        return (
            "GREETING:\n"
            "Hello, this is the Amazon Returns Verification Team. "
            "May I speak with the account holder?\n\n"
            "VERIFICATION:\n"
            "For security, could you please verify your name and "
            "the last four digits of the payment method on file?\n\n"
            "CONCERN:\n"
            "We're calling regarding a recent return request that "
            "requires additional verification before we can process it.\n\n"
            "RESOLUTION:\n"
            "We'd like to offer the following options:\n"
            "1. Provide additional documentation or photos\n"
            "2. Schedule a return pickup with inspection\n"
            "3. Speak with a supervisor for further assistance\n\n"
            "CLOSING:\n"
            "Thank you for your time and patience. We'll follow up "
            "via email with next steps within 24 hours."
        )

    def embed_text(self, text: str) -> list[float]:
        """Get text embedding via Nova Multimodal Embeddings, or Titan Embed fallback."""
        # Skip Bedrock embed calls when no embedding model is available in account
        if os.getenv("SKIP_EMBEDDING", "").lower() in ("1", "true", "yes"):
            return [0.0] * 256
        # 1. Try Nova Multimodal Embeddings (hackathon-preferred)
        try:
            body = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "TEXT_RETRIEVAL",
                    "embeddingDimension": 384,
                    "text": {"truncationMode": "END", "value": text},
                },
            }
            response = self._client.invoke_model(
                modelId=self._embed_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            emb = result.get("embedding") or result.get("embeddings", [{}])[0].get(
                "embedding", []
            )
            if emb:
                return emb
        except Exception as exc:
            logger.debug("Nova embed unavailable, trying Titan fallback: %s", exc)

        # 2. Fallback to Titan Embed (widely available when Nova Embed isn't)
        try:
            titan_body = {
                "inputText": text,
                "dimensions": 256,
                "normalize": True,
            }
            response = self._client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(titan_body),
            )
            result = json.loads(response["body"].read())
            emb = result.get("embedding", [])
            if emb:
                return emb
        except Exception as exc:
            logger.warning("Titan embed also failed, using zero vector: %s", exc)

        return [0.0] * 256

    def text_to_speech(self, text: str, output_format: str = "mp3") -> bytes:
        """Generate speech from text via AWS Polly (neural engine).

        Nova 2 Sonic is a speech-to-speech model requiring bidirectional
        WebSocket streaming.  For single-turn TTS in the dashboard we use
        Polly's neural engine which produces high-quality audio suitable for
        the demo.
        """
        try:
            polly = boto3.client("polly", region_name=self._region)
            response = polly.synthesize_speech(
                Text=text,
                OutputFormat=output_format,
                VoiceId="Joanna",
                Engine="neural",
            )
            audio_bytes = response["AudioStream"].read()
            logger.info("Generated %d bytes of audio via Polly", len(audio_bytes))
            return audio_bytes
        except Exception as exc:
            logger.warning("Polly TTS failed: %s", exc)
            return b""

    def reason_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Invoke Nova and parse the response as JSON."""
        raw = self.reason(
            system_prompt,
            user_prompt + "\n\nRespond ONLY with valid JSON.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Nova JSON, returning raw text")
            return {"raw_response": cleaned}

    @staticmethod
    def _fallback_reason(system_prompt: str, user_prompt: str) -> str:
        """Rule-based fallback when Bedrock is unavailable (demo/offline)."""
        prompt_lower = (system_prompt + user_prompt).lower()

        if "serial" in prompt_lower and "mismatch" in prompt_lower:
            return json.dumps({
                "risk_score": 0.90,
                "fraud_type": "SERIAL_MISMATCH",
                "confidence": 0.88,
                "reasoning": "Serial number on returned item does not match the shipped item record.",
            })
        if "empty" in prompt_lower and "box" in prompt_lower:
            return json.dumps({
                "risk_score": 0.95,
                "fraud_type": "EMPTY_BOX",
                "confidence": 0.92,
                "reasoning": "Return package weight is significantly below expected weight.",
            })
        if "counterfeit" in prompt_lower or "swap" in prompt_lower:
            return json.dumps({
                "risk_score": 0.88,
                "fraud_type": "COUNTERFEIT_SWAP",
                "confidence": 0.85,
                "reasoning": "Returned item visual/serial evidence differs from shipped product.",
            })
        if "policy" in prompt_lower and "gaming" in prompt_lower:
            return json.dumps({
                "risk_score": 0.55,
                "fraud_type": "POLICY_GAMING",
                "confidence": 0.70,
                "reasoning": "Account shows pattern of exploiting return window limits.",
            })
        return json.dumps({
            "risk_score": 0.15,
            "fraud_type": "LEGITIMATE",
            "confidence": 0.80,
            "reasoning": "No strong fraud indicators detected in available evidence.",
        })
