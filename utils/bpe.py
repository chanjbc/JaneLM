class BPE:
    def __init__(self):
        # Initially, vocab just contains 256 single byte tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def get_stats(self, ids):
        """
        Given a list of token IDs, return a dictionary of counts of consecutive pairs.
        """
        counts = {}
        for i, j in zip(ids, ids[1:]):
            counts[(i, j)] = counts.get((i, j), 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        In the list of integers `ids`, replace all consecutive occurrences
        of `pair` with the new integer token `idx`.
        """
        new_ids = []
        i, j = 0, 1
        while j < len(ids):
            if pair == (ids[i], ids[j]):
                new_ids.append(idx)
                i += 1
                j += 1
            else:
                new_ids.append(ids[i])
            i += 1
            j += 1
        if i < len(ids):
            new_ids.append(ids[-1])
        return new_ids

    def train(self, text, vocab_size):
        """
        Train the BPE tokenizer on the given text to achieve the target vocabulary size.
        """
        # Ensure vocab_size is at least 256 since we start with byte-level tokens
        assert vocab_size >= 256, "Vocab size >= 256 != True"
        num_merges = vocab_size - 256
        
        # Convert text to raw bytes, then to a list of integers (0-255)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        print(f"Starting with {len(ids)} tokens")

        for i in range(num_merges):
            
            # Find pair with highest frequency
            stats = self.get_stats(ids)
            most_freq_pair = None
            most_freq_count = 0
            for pair, count in stats.items():
                if count > most_freq_count:
                    most_freq_pair = pair
                    most_freq_count = count

            # Merge pair and store in merges/vocab
            new_id = 256 + i
            ids = self.merge(ids, most_freq_pair, new_id)
            self.merges[most_freq_pair] = new_id
            self.vocab[new_id] = self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]

        print(f"Finished training. Vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        """
        Encode a string into a list of token IDs using the learned merges.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:

            # Find pair in ids that was historically merged first
            stats = self.get_stats(ids)
            first_pair = None
            first_pair_id = len(self.vocab)
            for pair in stats:
                pair_id = self.merges.get(pair, len(self.vocab))
                if pair_id < first_pair_id:
                    first_pair = pair
                    first_pair_id = pair_id

            if not first_pair:
                break
            ids = self.merge(ids, first_pair, first_pair_id)
        return ids

    def decode(self, ids):
        """
        Decode a list of token IDs back into a string.
        """
        decode_ids = [ self.vocab[tok_id] for tok_id in ids ]
        decode_str = b"".join(decode_ids).decode("utf-8", errors="replace")
        return decode_str



if __name__ == "__main__":
    sample_text = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
    target_vocab_size = 275 # e.g., 20 merges
    
    tokenizer = BPE()
    tokenizer.train(sample_text, target_vocab_size)
    
    encoded_ids = tokenizer.encode("I love blocks of cheese and it is not universally acknowledged")
    print("Encoded:", encoded_ids)
    
    decoded_text = tokenizer.decode(encoded_ids)
    print("Decoded:", decoded_text)