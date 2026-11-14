import argparse
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain,model_config_from_dict,tokenizer_from_checkpoint
from train import choose_device
def main():
 p=argparse.ArgumentParser(description="Generate text with a Popcorn Brain checkpoint"); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--prompt",default=""); p.add_argument("--max-new-tokens",type=int,default=240); p.add_argument("--temperature",type=float,default=.8); p.add_argument("--top-k",type=int,default=40); p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto"); a=p.parse_args()
 if a.max_new_tokens<1: p.error("--max-new-tokens must be positive")
 device=choose_device(a.device); ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
 if ckpt.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
 tok=tokenizer_from_checkpoint(ckpt); config=model_config_from_dict(ckpt["model_config"]); model=PopcornBrain(config).to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
 ids=tok.encode(a.prompt) or [tok.eos_token_id]; result=model.generate(torch.tensor([ids],dtype=torch.long,device=device),a.max_new_tokens,a.temperature,a.top_k,tok.eos_token_id); print(tok.decode(result[0,len(ids):].tolist()))
if __name__=="__main__": main()
