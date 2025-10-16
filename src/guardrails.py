"""
Safety guardrails for the RAG chatbot.
Implements input moderation and output validation.
"""

from typing import Tuple
import re
import logging

# Try to import profanity check, but make it optional
try:
    from profanity_check import predict
    PROFANITY_CHECK_AVAILABLE = True
except ImportError:
    PROFANITY_CHECK_AVAILABLE = False
    logging.warning("Profanity check not available. Install with: pip install alt-profanity-check onnxruntime")

logger = logging.getLogger(__name__)


class InputGuardrail:
    """
    Input validation and moderation guardrail.
    Checks for profanity, inappropriate content, and prompt injection attempts.
    """
    
    def __init__(self, enable_profanity_check: bool = True):
        """
        Initialize the input guardrail.
        
        Args:
            enable_profanity_check: Whether to enable profanity filtering
        """
        self.enable_profanity_check = enable_profanity_check
        
        # Common prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions?",
            r"disregard\s+(previous|above|all)\s+instructions?",
            r"forget\s+(previous|above|all)\s+instructions?",
            r"you\s+are\s+now",
            r"new\s+instructions?:",
            r"system\s*:\s*",
            r"<\s*system\s*>",
        ]
    
    def check_profanity(self, text: str) -> bool:
        """
        Check if text contains profanity.
        
        Args:
            text: Input text to check
            
        Returns:
            True if profanity is detected, False otherwise
        """
        if not self.enable_profanity_check or not PROFANITY_CHECK_AVAILABLE:
            return False
        
        try:
            # predict returns array of 0 or 1 (1 = profanity detected)
            result = predict([text])
            return bool(result[0])
        except Exception as e:
            logger.error(f"Profanity check failed: {e}")
            return False
    
    def check_prompt_injection(self, text: str) -> bool:
        """
        Check for common prompt injection patterns.
        
        Args:
            text: Input text to check
            
        Returns:
            True if potential injection detected, False otherwise
        """
        text_lower = text.lower()
        
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"Potential prompt injection detected: {pattern}")
                return True
        
        return False
    
    def validate(self, user_input: str) -> Tuple[bool, str]:
        """
        Validate user input against all guardrails.
        
        Args:
            user_input: The user's query
            
        Returns:
            Tuple of (is_valid, reason)
            - is_valid: True if input passes all checks
            - reason: Empty string if valid, otherwise the reason for rejection
        """
        if not user_input or not user_input.strip():
            return False, "Empty input"
        
        # Check length (prevent extremely long inputs)
        if len(user_input) > 5000:
            return False, "Input too long (max 5000 characters)"
        
        # Check for profanity
        if self.check_profanity(user_input):
            logger.warning("Profanity detected in user input")
            return False, "Inappropriate content detected"
        
        # Check for prompt injection
        if self.check_prompt_injection(user_input):
            return False, "Potential security risk detected"
        
        return True, ""


class OutputGuardrail:
    """
    Output validation guardrail.
    Checks generated responses for potential issues like PII leakage or hallucination indicators.
    """
    
    def __init__(self):
        """Initialize the output guardrail."""
        # Patterns that might indicate PII (simplified for demonstration)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
            r'\b\d{16}\b',  # Credit card format
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
    
    def check_pii(self, text: str) -> bool:
        """
        Check if text potentially contains PII.
        
        Args:
            text: Generated text to check
            
        Returns:
            True if potential PII detected, False otherwise
        """
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                logger.warning(f"Potential PII detected: {pattern}")
                return True
        
        return False
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from XML-tagged response.
        
        Args:
            text: Full LLM response potentially containing <thinking> and <answer> tags
            
        Returns:
            Extracted answer or full text if tags not found
        """
        # Try to extract content from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            return answer_match.group(1).strip()
        
        # If no tags found, return the full text
        return text.strip()
    
    def validate(self, output: str) -> Tuple[bool, str, str]:
        """
        Validate LLM output.
        
        Args:
            output: The generated response
            
        Returns:
            Tuple of (is_valid, sanitized_output, reason)
            - is_valid: True if output passes checks
            - sanitized_output: Cleaned version of the output
            - reason: Empty if valid, otherwise reason for rejection
        """
        # Extract answer from tags
        sanitized = self.extract_answer(output)
        
        # Check for PII
        if self.check_pii(sanitized):
            return False, "", "Response contains potentially sensitive information"
        
        # Check if response is empty
        if not sanitized or not sanitized.strip():
            return False, "", "Empty response generated"
        
        return True, sanitized, ""


def create_guardrails(enable_input: bool = True, enable_output: bool = True):
    """
    Factory function to create guardrail instances.
    
    Args:
        enable_input: Whether to enable input guardrails
        enable_output: Whether to enable output guardrails
        
    Returns:
        Tuple of (InputGuardrail, OutputGuardrail) or None for disabled guardrails
    """
    input_guard = InputGuardrail(enable_profanity_check=enable_input) if enable_input else None
    output_guard = OutputGuardrail() if enable_output else None
    
    return input_guard, output_guard

