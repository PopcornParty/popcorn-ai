"""Small decoder-only Transformer and tokenizers for Popcorn AI research."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
@dataclass
class ModelConfig:
 vocab_size:int
 block_size:int=128
 n_embd:int=128
 n_head:int=4
 n_layer:int=4
 dropout:float=.1
 def validate(self):
  if self.vocab_size<3: raise ValueError("vocab_size must be at least 3")
  if self.n_head<1 or self.n_embd%self.n_head: raise ValueError("n_embd must be divisible by a positive n_head")
  if min(self.block_size,self.n_embd,self.n_layer)<1: raise ValueError("Model dimensions must be positive")
  if not 0<=self.dropout<1: raise ValueError("dropout must be in [0, 1)")
class ByteTokenizer:
 eos_token_id=256; vocab_size=257; tokenizer_type="byte-v1"
 def encode(self,text:str,add_eos:bool=False):
  ids=list(text.encode("utf-8")); return ids+([self.eos_token_id] if add_eos else [])
 def decode(self,ids:Iterable[int]):
  values=[]
  for token in ids:
   token=int(token)
   if token==self.eos_token_id: break
   if 0<=token<=255: values.append(token)
  return bytes(values).decode("utf-8",errors="replace")
 def to_dict(self): return {"type":self.tokenizer_type,"vocab_size":self.vocab_size,"eos_token_id":self.eos_token_id}
class CharTokenizer:
 UNK="<unk>"; EOS="<eos>"; tokenizer_type="char-v1"
 def __init__(self,vocabulary:list[str]):
  if len(vocabulary)<3 or vocabulary[0]!=self.UNK or vocabulary[1]!=self.EOS: raise ValueError("Vocabulary must start with <unk>, <eos>")
  if len(set(vocabulary))!=len(vocabulary): raise ValueError("Vocabulary entries must be unique")
  self.vocabulary=vocabulary; self.stoi={t:i for i,t in enumerate(vocabulary)}; self.vocab_size=len(vocabulary); self.eos_token_id=1
 @classmethod
 def from_texts(cls,texts:Iterable[str]): return cls([cls.UNK,cls.EOS,*sorted({c for t in texts for c in t})])
 def encode(self,text:str,add_eos:bool=False):
  ids=[self.stoi.get(c,0) for c in text]; return ids+([self.eos_token_id] if add_eos else [])
 def decode(self,ids:Iterable[int]):
  out=[]
  for i in ids:
   token=self.vocabulary[int(i)] if 0<=int(i)<len(self.vocabulary) else self.UNK
   if token==self.EOS: break
   if token!=self.UNK: out.append(token)
  return "".join(out)
 def to_dict(self): return {"type":self.tokenizer_type,"vocabulary":self.vocabulary,"vocab_size":self.vocab_size,"eos_token_id":self.eos_token_id}
def tokenizer_from_checkpoint(checkpoint:dict):
 meta=checkpoint.get("tokenizer")
 if meta and meta.get("type")==ByteTokenizer.tokenizer_type: return ByteTokenizer()
 vocab=meta.get("vocabulary") if meta else checkpoint.get("vocabulary")
 if not vocab: raise ValueError("Checkpoint has no supported tokenizer metadata")
 return CharTokenizer(vocab)
class CausalSelfAttention(nn.Module):
 def __init__(self,config):
  super().__init__(); self.attn=nn.MultiheadAttention(config.n_embd,config.n_head,dropout=config.dropout,batch_first=True)
  self.register_buffer("causal_mask",torch.triu(torch.ones(config.block_size,config.block_size,dtype=torch.bool),diagonal=1),persistent=False); self.dropout=nn.Dropout(config.dropout)
 def forward(self,x):
  n=x.size(1); out,_=self.attn(x,x,x,attn_mask=self.causal_mask[:n,:n],need_weights=False); return self.dropout(out)
class TransformerBlock(nn.Module):
 def __init__(self,c):
  super().__init__(); self.norm1=nn.LayerNorm(c.n_embd); self.attention=CausalSelfAttention(c); self.norm2=nn.LayerNorm(c.n_embd)
  self.feed_forward=nn.Sequential(nn.Linear(c.n_embd,4*c.n_embd),nn.GELU(),nn.Linear(4*c.n_embd,c.n_embd),nn.Dropout(c.dropout))
 def forward(self,x):
  x=x+self.attention(self.norm1(x)); return x+self.feed_forward(self.norm2(x))
class PopcornBrain(nn.Module):
 def __init__(self,config:ModelConfig):
  super().__init__(); config.validate(); self.config=config
  self.token_embedding=nn.Embedding(config.vocab_size,config.n_embd); self.position_embedding=nn.Embedding(config.block_size,config.n_embd)
  self.blocks=nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)]); self.final_norm=nn.LayerNorm(config.n_embd); self.language_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
  self.apply(self._init_weights); self.language_head.weight=self.token_embedding.weight
 @staticmethod
 def _init_weights(m):
  if isinstance(m,nn.Linear):
   nn.init.normal_(m.weight,mean=0.,std=.02)
   if m.bias is not None: nn.init.zeros_(m.bias)
  elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,mean=0.,std=.02)
 def forward(self,token_ids,targets=None):
  _,n=token_ids.shape
  if n>self.config.block_size: raise ValueError(f"Input length {n} exceeds block size {self.config.block_size}")
  pos=torch.arange(n,device=token_ids.device); x=self.token_embedding(token_ids)+self.position_embedding(pos)[None,:,:]
  logits=self.language_head(self.final_norm(self.blocks(x))); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) if targets is not None else None
  return logits,loss
 @torch.no_grad()
 def generate(self,token_ids,max_new_tokens=160,temperature=.8,top_k=40,eos_token_id=1):
  if temperature<=0: raise ValueError("temperature must be greater than zero")
  self.eval()
  for _ in range(max_new_tokens):
   logits,_=self(token_ids[:,-self.config.block_size:]); logits=logits[:,-1,:]/temperature
   if top_k>0:
    cutoff=torch.topk(logits,min(top_k,logits.size(-1))).values[:,-1,None]; logits=logits.masked_fill(logits<cutoff,float("-inf"))
   nxt=torch.multinomial(F.softmax(logits,dim=-1),1); token_ids=torch.cat((token_ids,nxt),dim=1)
   if nxt.item()==eos_token_id: break
  return token_ids
def model_config_from_dict(values:dict):
 allowed=set(ModelConfig.__dataclass_fields__); return ModelConfig(**{k:v for k,v in values.items() if k in allowed})
