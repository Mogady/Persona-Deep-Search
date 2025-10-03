"""
Unified JSON parsing utility for handling AI model responses.

Supports:
- Raw JSON strings
- JSON inside markdown code blocks (```json ... ```)
- Extraction from mixed text
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_from_markdown(
    text: str,
    expected_type: type = dict,
    fallback_value: Optional[Any] = None
) -> Union[Dict, List, Any]:
    """
    Parse JSON from AI response text (handles markdown code blocks).

    Args:
        text: Raw text from AI response
        expected_type: Expected type (dict or list)
        fallback_value: Value to return if parsing fails

    Returns:
        Parsed JSON object or fallback_value

    Raises:
        ValueError: If parsing fails and no fallback provided
    """
    if not text:
        if fallback_value is not None:
            return fallback_value
        raise ValueError("Empty text provided for JSON parsing")

    # Strategy 1: Try raw JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, expected_type):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    # Patterns: ```json ... ```, ``` ... ```
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_text = match.group(1).strip()
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, expected_type):
                    logger.debug("Extracted JSON from markdown code block")
                    return parsed
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find JSON object/array anywhere in text
    # Look for { ... } or [ ... ]
    if expected_type == dict:
        # Match nested objects
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    else:  # list
        # Match nested arrays
        json_pattern = r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]'

    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, expected_type):
                logger.debug("Extracted JSON from text content")
                return parsed
        except json.JSONDecodeError:
            continue

    # All strategies failed
    if fallback_value is not None:
        logger.warning(f"Failed to parse JSON, using fallback value")
        return fallback_value

    raise ValueError(f"Failed to parse {expected_type.__name__} from text")


def parse_json_array(text: str, fallback: Optional[List] = None) -> List:
    """
    Parse JSON array from text.

    Args:
        text: Text containing JSON array
        fallback: Value to return if parsing fails (default: [])

    Returns:
        Parsed list or fallback value
    """
    return parse_json_from_markdown(
        text,
        expected_type=list,
        fallback_value=fallback if fallback is not None else []
    )


def parse_json_object(text: str, fallback: Optional[Dict] = None) -> Dict:
    """
    Parse JSON object from text.

    Args:
        text: Text containing JSON object
        fallback: Value to return if parsing fails (default: {})

    Returns:
        Parsed dict or fallback value
    """
    return parse_json_from_markdown(
        text,
        expected_type=dict,
        fallback_value=fallback if fallback is not None else {}
    )
