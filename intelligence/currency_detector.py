"""
Currency Detector - Roboflow API
─────────────────────────────────
Detects USD bills using Roboflow serverless inference.
Based on working currency_test.py.
"""

import cv2
import logging
from collections import Counter

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

logger = logging.getLogger(__name__)


class CurrencyDetector:
    """USD bill detection via Roboflow API"""

    def __init__(self, api_key="kv4gQ6W5qMzpcT71o2yb"):
        """
        Args:
            api_key: Roboflow API key
        """
        if InferenceHTTPClient is None:
            raise ImportError("Install: pip install inference-sdk")
        
        logger.info("Initializing currency detector (Roboflow API)...")
        
        try:
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key
            )
            self.model_id = "usd-money/2"
            logger.info("Currency detector ready")
        except Exception as e:
            logger.error(f"Failed to init Roboflow client: {e}")
            raise

    def detect_and_count(self, frame, confidence=0.5):
        """
        Detect bills in frame and count total.
        
        Args:
            frame: OpenCV image (BGR)
            confidence: Detection confidence threshold (default 0.5)
        
        Returns:
            (total_amount, breakdown_dict, raw_result)
            
            Example:
            (85, {'20': 3, '10': 2, '5': 1}, {...})
        """
        try:
            # Save frame temporarily (API needs file)
            temp_path = "temp_currency.jpg"
            frame = cv2.resize(frame, (640, 480))
            cv2.imwrite(temp_path, frame)
            
            # Run inference
            result = self.client.infer(temp_path, model_id=self.model_id)
            
            # Extract bills (same logic as currency_test.py)
            notes = self._extract_currency(result, threshold=confidence)
            
            # Count and compute total
            count, total = self._compute_total(notes)
            
            # Format breakdown
            breakdown = {str(k): v for k, v in count.items()}
            
            logger.info(f"Detected {len(notes)} bills, total: ${total}")
            
            return total, breakdown, result
        
        except Exception as e:
            logger.error(f"Currency detection failed: {e}")
            return 0, {}, {}

    def _extract_currency(self, result, threshold=0.5):
        """
        Extract bill values from API result.
        Same as currency_test.py extract_currency()
        """
        notes = []
        
        predictions = result.get("predictions", [])
        
        for pred in predictions:
            if pred.get("confidence", 0) >= threshold:
                label = pred.get("class", "")
                
                # Parse denomination
                # Labels like "5Dollar", "10Dollar", etc.
                try:
                    value = int(label.replace("Dollar", ""))
                    notes.append(value)
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse label: {label}")
        
        return notes

    def _compute_total(self, notes):
        """
        Count notes and compute total.
        Same as currency_test.py compute_total()
        """
        count = Counter(notes)
        total = sum(k * v for k, v in count.items())
        return count, total

    def format_result(self, total, breakdown):
        """
        Convert detection results to speakable text.
        """
        if total == 0:
            return "No bills detected. Please ensure bills are clearly visible."
        
        # Build denomination breakdown
        parts = []
        for denom in sorted(breakdown.keys(), key=lambda x: int(x), reverse=True):
            count = breakdown[denom]
            if count == 1:
                parts.append(f"one {denom} dollar bill")
            else:
                parts.append(f"{count} {denom} dollar bills")
        
        if len(parts) == 0:
            return f"Total: {total} dollars."
        elif len(parts) == 1:
            return f"I see {parts[0]}. Total: {total} dollars."
        elif len(parts) == 2:
            return f"I see {parts[0]} and {parts[1]}. Total: {total} dollars."
        else:
            last = parts.pop()
            breakdown_text = ", ".join(parts) + f", and {last}"
            return f"I see {breakdown_text}. Total: {total} dollars."


# ══════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Currency Detector Test")

    
    # Initialize
    try:
        detector = CurrencyDetector()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure: pip install inference-sdk")
        sys.exit(1)
    
    # Test with webcam
    print("\nPoint camera at bills and press SPACE to count")
    print("Press Q to quit\n")
    
    cap = cv2.VideoCapture(0)  # Change to 0 if needed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Currency Detector - Press SPACE", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            print("\n[Counting...]")
            total, breakdown, result = detector.detect_and_count(frame)
            result_text = detector.format_result(total, breakdown)
            
            print(f"Result: {result_text}")
            print(f"Breakdown: {breakdown}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
