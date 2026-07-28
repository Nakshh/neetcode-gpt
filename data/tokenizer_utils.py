from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        output = []
        for num in numbers:
            text = str(num)
            l = 0
            o = []
            while l < len(text):
                r = len(text)
                while r > l and text[l:r] not in vocab:
                    r -= 1
                
                o.append(text[l:r])
                l = r
            output.append(o)
        
        return output

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count = 0
        l = 0
        while l < len(text):
            r = len(text)
            while l < r and text[l:r] not in vocab:
                r -= 1
            count += 1
            l = r

        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        return round(self.count_tokens(text, vocab) / len(text.split()), 4)
