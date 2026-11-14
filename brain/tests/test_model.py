import unittest
import torch
from src.popcorn_brain import ByteTokenizer,CharTokenizer,ModelConfig,PopcornBrain,tokenizer_from_checkpoint

class PopcornBrainTests(unittest.TestCase):
 def test_character_tokenizer_legacy_round_trip(self):
  tok=CharTokenizer.from_texts(["hello","world"]); self.assertEqual(tok.decode(tok.encode("hello")),"hello")
 def test_byte_tokenizer_handles_unicode_and_code(self):
  tok=ByteTokenizer(); sample="def café(x): return x + 1\n"
  self.assertEqual(tok.decode(tok.encode(sample)),sample)
  self.assertEqual(tok.encode("x",True)[-1],tok.eos_token_id)
 def test_legacy_checkpoint_tokenizer_is_readable(self):
  tok=CharTokenizer.from_texts(["hello"]); restored=tokenizer_from_checkpoint({"format_version":1,"vocabulary":tok.vocabulary})
  self.assertEqual(restored.decode(restored.encode("hello")),"hello")
 def test_byte_checkpoint_tokenizer_is_readable(self):
  tok=tokenizer_from_checkpoint({"format_version":2,"tokenizer":{"type":"byte-v1"}})
  self.assertEqual(tok.decode(tok.encode("print('hi')")),"print('hi')")
 def test_model_output_and_loss(self):
  c=ModelConfig(vocab_size=16,block_size=12,n_embd=16,n_head=4,n_layer=2,dropout=0.)
  m=PopcornBrain(c); tokens=torch.randint(0,c.vocab_size,(2,10)); logits,loss=m(tokens,tokens)
  self.assertEqual(tuple(logits.shape),(2,10,c.vocab_size)); self.assertIsNotNone(loss); self.assertTrue(torch.isfinite(loss).item())
 def test_context_limit_is_enforced(self):
  m=PopcornBrain(ModelConfig(vocab_size=8,block_size=8,n_embd=16,n_head=4,n_layer=1))
  with self.assertRaises(ValueError): m(torch.zeros((1,9),dtype=torch.long))
 def test_causal_mask_blocks_future_positions(self):
  from src.popcorn_brain import CausalSelfAttention
  c=ModelConfig(vocab_size=8,block_size=8,n_embd=16,n_head=4,n_layer=1)
  attention=CausalSelfAttention(c)
  self.assertTrue(torch.equal(attention.causal_mask,torch.triu(torch.ones(8,8,dtype=torch.bool),diagonal=1)))
if __name__=="__main__": unittest.main()
