from collections import Counter
from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []             # what is appended and returned

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # count freq
            count = Counter((tokens[i], tokens[i+1]) for i in range(len(tokens) -1))

            # find the most frequent pair
            best = min(count, key=lambda pair: (-count[pair], pair)) # we can negate the count 

            merges.append([best[0], best[1]])

            # merging time
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == best:
                    tokens[i] = tokens[i] + tokens[i+1]
                    tokens.pop(i+1)
                i+= 1

        
        return merges



