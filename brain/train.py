from __future__ import annotations
import argparse,json,random
from pathlib import Path
import torch
from src.popcorn_brain import ByteTokenizer,ModelConfig,PopcornBrain,tokenizer_from_checkpoint
from validate_dataset import inspect_dataset
def load_texts(path:Path)->list[str]:
 report=inspect_dataset(path)
 if not report["ready_for_training"]:
  raise ValueError("Dataset is not approved for training. Run brain/validate_dataset.py and fix all reported issues.")
 texts=[]; seen=set()
 for line in path.read_text(encoding="utf-8").splitlines():
  if not line.strip(): continue
  value=json.loads(line)["text"].strip()
  if value not in seen: texts.append(value); seen.add(value)
 if len(texts)<3: raise ValueError("Provide at least three unique non-empty training examples")
 return texts
def choose_device(requested:str):
 if requested=="auto":
  if torch.cuda.is_available(): return torch.device("cuda")
  if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
  return torch.device("cpu")
 d=torch.device(requested)
 if d.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
 if d.type=="mps" and not (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()): raise RuntimeError("MPS was requested but is unavailable")
 return d
def main():
 p=argparse.ArgumentParser(description="Train or resume the experimental Popcorn Transformer")
 p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,default=Path("brain/checkpoints/experiment")); p.add_argument("--max-iters",type=int,default=1000,help="Total optimiser steps, not additional steps")
 p.add_argument("--batch-size",type=int,default=32); p.add_argument("--block-size",type=int,default=64); p.add_argument("--eval-interval",type=int,default=100); p.add_argument("--learning-rate",type=float,default=3e-4)
 p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto"); p.add_argument("--seed",type=int,default=42); p.add_argument("--resume",type=Path)
 a=p.parse_args()
 if min(a.max_iters,a.batch_size,a.block_size,a.eval_interval)<1 or a.block_size<8 or a.learning_rate<=0: p.error("Training values must be positive; block size must be at least 8")
 random.seed(a.seed); torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
 docs=load_texts(a.data); random.Random(a.seed).shuffle(docs); nval=max(1,round(len(docs)*.1)); val_docs,train_docs=docs[:nval],docs[nval:]
 tok=ByteTokenizer(); train=torch.tensor([t for s in train_docs for t in tok.encode(s,True)],dtype=torch.long); val=torch.tensor([t for s in val_docs for t in tok.encode(s,True)],dtype=torch.long)
 for name,data in (("training",train),("validation",val)):
  if len(data)<a.block_size+2: raise ValueError(f"{name.title()} split has {len(data)} bytes; need {a.block_size+2}. Add distinct documents or lower --block-size.")
 device=choose_device(a.device); config=ModelConfig(tok.vocab_size,a.block_size); model=PopcornBrain(config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
 a.out.mkdir(parents=True,exist_ok=True); start=0; history=[]; best=float("inf")
 if a.resume:
  ckpt=torch.load(a.resume,map_location=device,weights_only=False)
  if ckpt.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
  if tokenizer_from_checkpoint(ckpt).tokenizer_type!=tok.tokenizer_type: raise ValueError("Checkpoint uses another tokenizer; start a new run")
  if ckpt["model_config"].get("block_size")!=a.block_size: raise ValueError("Resume block size must match checkpoint")
  model.load_state_dict(ckpt["model_state"])
  if ckpt.get("optimizer_state"): opt.load_state_dict(ckpt["optimizer_state"])
  state=ckpt.get("training",{}); start=int(state.get("next_step",state.get("step",0))); history=list(state.get("metrics",[])); best=float(state.get("best_validation_loss",float("inf")))
 def batch(data):
  starts=torch.randint(len(data)-a.block_size,(a.batch_size,)); x=torch.stack([data[int(i):int(i)+a.block_size] for i in starts]).to(device); y=torch.stack([data[int(i)+1:int(i)+a.block_size+1] for i in starts]).to(device); return x,y
 @torch.no_grad()
 def losses():
  model.eval(); out={}
  for name,data in (("train",train),("validation",val)):
   vals=[]
   for _ in range(8):
    x,y=batch(data); _,loss=model(x,y); vals.append(float(loss.item()))
   out[name]=sum(vals)/len(vals)
  model.train(); return out
 print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; tokenizer: byte-v1; train bytes: {len(train)}; validation bytes: {len(val)}")
 if start>=a.max_iters: raise ValueError(f"Checkpoint is at step {start}; set --max-iters higher")
 for step in range(start,a.max_iters):
  x,y=batch(train); _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
  completed=step+1
  if completed%a.eval_interval==0 or completed==a.max_iters:
   metric=losses(); history.append({"step":completed,**metric}); improved=metric["validation"]<best
   if improved: best=metric["validation"]
   payload={"format_version":2,"tokenizer":tok.to_dict(),"model_config":config.__dict__,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"training":{"step":completed,"next_step":completed,"seed":a.seed,"device":str(device),"best_validation_loss":best,"metrics":history}}
   torch.save(payload,a.out/"latest.pt")
   if improved or not (a.out/"best.pt").exists(): torch.save(payload,a.out/"best.pt")
   (a.out/"metrics.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
   print(f"step {completed:5d} | train {metric['train']:.4f} | validation {metric['validation']:.4f} | best {best:.4f}")
 print(f"Saved latest and best checkpoints in {a.out}")
if __name__=="__main__": main()
