import unittest

import torch

from src.popcorn_brain import CharTokenizer, ModelConfig, PopcornBrain


class PopcornBrainTests(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        tokenizer = CharTokenizer.from_texts(["hello", "world"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")

    def test_model_output_and_loss(self):
        config = ModelConfig(vocab_size=8, block_size=12, n_embd=16, n_head=4, n_layer=2, dropout=0.0)
        model = PopcornBrain(config)
        tokens = torch.randint(0, config.vocab_size, (2, 10))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 10, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

    def test_context_limit_is_enforced(self):
        model = PopcornBrain(ModelConfig(vocab_size=8, block_size=8, n_embd=16, n_head=4, n_layer=1))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 9), dtype=torch.long))

    def test_causal_mask_blocks_future_positions(self):
        config = ModelConfig(vocab_size=8, block_size=8, n_embd=16, n_head=4, n_layer=1)
        attention = __import__("src.popcorn_brain", fromlist=["CausalSelfAttention"]).CausalSelfAttention(config)
        self.assertTrue(torch.equal(attention.causal_mask, torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)))


if __name__ == "__main__":
    unittest.main()
