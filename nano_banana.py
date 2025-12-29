"""
nano_banana.py - Nano Banana API Integration for Project WARGATE

This module provides integration with Google's Gemini API for generating
summary infographic images at the end of each JPP phase.

Nano Banana Models:
- gemini-2.5-flash-image (Nano Banana): Fast/cheap image generation
- gemini-3-pro-image-preview (Nano Banana Pro): Higher fidelity, better text rendering

API Documentation:
- Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- Auth: API key via x-goog-api-key header
- Response: Base64-encoded PNG image in inline_data

Usage:
    from nano_banana import NanoBananaClient, NanoBananaConfig

    config = NanoBananaConfig()  # Uses GEMINI_API_KEY env var
    client = NanoBananaClient(config)

    image_bytes = client.generate_phase_summary_image(
        phase_name="Mission Analysis",
        phase_delta={
            "what_learned": "...",
            "what_changed": "...",
            "tensions": "..."
        },
        phase_result={...}
    )

    # Save or display the image
    with open("phase_summary.png", "wb") as f:
        f.write(image_bytes)

Author: Project WARGATE Team
"""

from __future__ import annotations

import os
import base64
import time
import json
from dataclasses import dataclass, field
from typing import Any, Literal
from pathlib import Path

import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NanoBananaConfig:
    """Configuration for the Nano Banana API client."""

    # Model selection
    model: Literal["gemini-2.5-flash-image", "gemini-3-pro-image-preview"] = "gemini-2.5-flash-image"

    # API settings
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Generation settings
    aspect_ratio: str = "16:9"  # Good for briefing slides

    # Retry settings
    max_retries: int = 3
    initial_delay: float = 2.0

    # Whether to enable/disable image generation
    enabled: bool = True

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")

    @property
    def endpoint(self) -> str:
        """Get the full API endpoint URL."""
        return f"{self.base_url}/models/{self.model}:generateContent"


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_phase_delta_content(
    phase_name: str,
    phase_delta: dict[str, str],
    guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build structured content from phase outputs for image generation.

    Args:
        phase_name: Name of the JPP phase
        phase_delta: Step delta dict with what_learned, what_changed, tensions, coas
        guidance: Commander's guidance dict (optional)

    Returns:
        Structured dict with key points for image generation
    """
    content = {
        "phase": phase_name,
        "key_insights": phase_delta.get("what_learned", ""),
        "evolution": phase_delta.get("what_changed", ""),
        "risks_tensions": phase_delta.get("tensions", ""),
    }

    # Extract COA information if present (for COA phases)
    coas = phase_delta.get("coas", "")
    if coas:
        content["coas"] = coas

    # Check if this is a COA-related phase
    is_coa_phase = any(keyword in phase_name.upper() for keyword in ["COA", "COURSE"])
    content["is_coa_phase"] = is_coa_phase

    if guidance:
        guidance_text = guidance.get("guidance_text", "")
        priority_tasks = guidance.get("priority_tasks", [])
        if priority_tasks:
            content["priority_tasks"] = priority_tasks[:4]  # Top 4 tasks
        if guidance_text:
            # Extract first sentence or two
            content["commander_direction"] = guidance_text[:200]

    return content


def build_infographic_prompt(
    phase_name: str,
    content: dict[str, Any],
    operation_name: str = "Operation",
) -> str:
    """
    Build a prompt for generating a military briefing infographic.

    Args:
        phase_name: Name of the JPP phase
        content: Structured content dict from build_phase_delta_content
        operation_name: Name of the operation

    Returns:
        Prompt string for image generation
    """
    # Build bullet points from content
    bullets = []
    is_coa_phase = content.get("is_coa_phase", False)

    # For COA phases, prioritize COA information
    if is_coa_phase and content.get("coas"):
        coas_text = content["coas"]
        # Truncate but keep more for COA info (up to 300 chars)
        bullets.append(f"COURSES OF ACTION: {coas_text[:300]}")

    if content.get("key_insights"):
        # Allow more space for COA phases
        max_len = 150 if is_coa_phase else 100
        bullets.append(f"KEY INSIGHT: {content['key_insights'][:max_len]}")

    if content.get("evolution"):
        max_len = 150 if is_coa_phase else 100
        bullets.append(f"EVOLUTION: {content['evolution'][:max_len]}")

    if content.get("risks_tensions"):
        bullets.append(f"RISKS: {content['risks_tensions'][:100]}")

    if content.get("priority_tasks"):
        tasks = content["priority_tasks"][:3]
        bullets.append(f"PRIORITY TASKS: {', '.join(tasks)}")

    if content.get("commander_direction"):
        bullets.append(f"CMDR DIRECTION: {content['commander_direction'][:80]}")

    bullets_text = "\n".join(f"- {b}" for b in bullets)

    # Add COA-specific instructions for COA phases
    coa_instructions = ""
    if is_coa_phase:
        coa_instructions = """
IMPORTANT FOR THIS COA PHASE:
- Give prominent visual treatment to the COURSES OF ACTION section
- Each COA should be clearly distinguishable (use numbered boxes or columns)
- Highlight any recommended or approved COA with a distinct visual indicator
"""

    prompt = f"""Create a clean, professional military briefing infographic.

STYLE REQUIREMENTS:
- Clean government/military briefing style
- Dark professional background (navy blue or charcoal)
- Light text for readability
- Simple geometric icons (no complex graphics)
- 16:9 aspect ratio (widescreen slide format)
- Bold, clear typography
- Organized in 3-4 distinct sections/boxes
{coa_instructions}
CONTENT TO DISPLAY:

HEADER:
"{operation_name}" - Phase: {phase_name}

MAIN CONTENT (render as separate boxes/sections):
{bullets_text}

FOOTER:
"Project WARGATE - Joint Planning Process"

Make it look like a professional military staff briefing slide - simple, clear, and authoritative.
Use military-style visual language: clean lines, structured layout, minimal decoration.
Text should be clearly readable. Use icons sparingly to represent concepts."""

    return prompt


# =============================================================================
# API CLIENT
# =============================================================================

class NanoBananaClient:
    """Client for the Nano Banana (Gemini) image generation API."""

    def __init__(self, config: NanoBananaConfig | None = None):
        """
        Initialize the client.

        Args:
            config: Optional configuration. Defaults to NanoBananaConfig().
        """
        self.config = config or NanoBananaConfig()

    def _make_request(self, prompt: str) -> bytes:
        """
        Make a request to the Gemini API to generate an image.

        Args:
            prompt: The image generation prompt

        Returns:
            Raw image bytes (PNG)

        Raises:
            ValueError: If API key is not configured
            requests.RequestException: If the API request fails
        """
        if not self.config.api_key:
            raise ValueError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY environment variable or pass api_key to NanoBananaConfig."
            )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key,
        }

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseModalities": ["Image"],
            }
        }

        # Add aspect ratio if supported (for pro model)
        if self.config.model == "gemini-3-pro-image-preview":
            payload["generationConfig"]["imageConfig"] = {
                "aspectRatio": self.config.aspect_ratio
            }

        last_exception = None
        delay = self.config.initial_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60,  # Image generation can take time
                )
                response.raise_for_status()

                # Parse response and extract image data
                data = response.json()

                # Navigate to the inline image data
                # Structure: candidates[0].content.parts[0].inlineData.data
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates in response")

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    raise ValueError("No parts in response content")

                inline_data = parts[0].get("inlineData", {})
                image_base64 = inline_data.get("data")
                if not image_base64:
                    raise ValueError("No image data in response")

                # Decode base64 to bytes
                return base64.b64decode(image_base64)

            except requests.RequestException as e:
                error_msg = str(e).lower()

                # Check if this is a transient error worth retrying
                is_transient = any([
                    "timeout" in error_msg,
                    "connection" in error_msg,
                    "rate limit" in error_msg,
                    response.status_code in (429, 500, 502, 503, 504) if hasattr(e, 'response') and e.response else False,
                ])

                if is_transient and attempt < self.config.max_retries:
                    print(f"[NanoBanana] Retry {attempt + 1}/{self.config.max_retries}: {e}")
                    time.sleep(delay)
                    delay *= 2
                    last_exception = e
                else:
                    raise

            except Exception as e:
                # Non-transient error, don't retry
                raise

        if last_exception:
            raise last_exception

    def generate_phase_summary_image(
        self,
        phase_name: str,
        phase_delta: dict[str, str],
        phase_result: dict[str, Any] | None = None,
        operation_name: str = "Operation",
    ) -> bytes:
        """
        Generate a summary infographic image for a completed JPP phase.

        Args:
            phase_name: Name of the phase (e.g., "Mission Analysis")
            phase_delta: Step delta dict with what_learned, what_changed, tensions
            phase_result: Full phase result dict (optional, for extracting guidance)
            operation_name: Name of the operation

        Returns:
            PNG image bytes
        """
        # Extract guidance if available
        guidance = None
        if phase_result:
            guidance = phase_result.get("guidance", {})

        # Build structured content
        content = build_phase_delta_content(phase_name, phase_delta, guidance)

        # Build the prompt
        prompt = build_infographic_prompt(phase_name, content, operation_name)

        # Generate the image
        return self._make_request(prompt)

    def is_available(self) -> bool:
        """
        Check if the Nano Banana API is available and configured.

        Returns:
            True if API key is set and enabled, False otherwise
        """
        return self.config.enabled and self.config.api_key is not None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_phase_summary_image(
    image_bytes: bytes,
    op_name: str,
    phase_name: str,
    log_root: Path = Path("wargate_logs"),
) -> Path:
    """
    Save a phase summary image to the operation's log directory.

    Args:
        image_bytes: PNG image bytes
        op_name: Operation name (will be sanitized)
        phase_name: Phase name for folder lookup
        log_root: Root directory for logs

    Returns:
        Path to the saved image file
    """
    import re

    # Sanitize operation name
    safe_op = re.sub(r"[^A-Za-z0-9]+", "_", op_name.strip())
    safe_op = re.sub(r"_+", "_", safe_op).strip("_") or "Operation"

    # Map phase names to folder names
    phase_folder_map = {
        "PLANNING_INITIATION": "PlanningInitiation",
        "MISSION_ANALYSIS": "MissionAnalysis",
        "COA_DEVELOPMENT": "COADevelopment",
        "COA_ANALYSIS": "COAAnalysis",
        "COA_COMPARISON": "COAComparison",
        "COA_APPROVAL": "COAApproval",
        "PLAN_DEVELOPMENT": "PlanDevelopment",
    }

    phase_folder = phase_folder_map.get(phase_name, phase_name.replace(" ", ""))

    target_dir = log_root / safe_op / phase_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / "Phase_Summary_Infographic.png"
    file_path.write_bytes(image_bytes)

    return file_path


def generate_phase_summary_image_async(
    phase_name: str,
    phase_delta: dict[str, str],
    phase_result: dict[str, Any] | None = None,
    operation_name: str = "Operation",
    config: NanoBananaConfig | None = None,
) -> bytes | None:
    """
    Generate a phase summary image, returning None if generation fails.

    This is a convenience wrapper that handles errors gracefully,
    suitable for use in contexts where image generation is optional.

    Args:
        phase_name: Name of the phase
        phase_delta: Step delta dict
        phase_result: Full phase result dict (optional)
        operation_name: Name of the operation
        config: Optional NanoBananaConfig

    Returns:
        PNG image bytes, or None if generation fails
    """
    try:
        client = NanoBananaClient(config)

        if not client.is_available():
            print("[NanoBanana] API not available (no API key or disabled)")
            return None

        return client.generate_phase_summary_image(
            phase_name=phase_name,
            phase_delta=phase_delta,
            phase_result=phase_result,
            operation_name=operation_name,
        )
    except Exception as e:
        print(f"[NanoBanana] Image generation failed: {e}")
        return None
