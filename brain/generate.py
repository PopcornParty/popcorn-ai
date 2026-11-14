import argparse
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain, model_config_from_dict, tokenizer_from_checkpoint
from train import choose_device

def main():
    parser=argparse.ArgumentParser(description="Generate text with a Popcorn Brain checkpoint")
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--prompt",type=str,default="")
    parser.add_argument("--max-new-tokens",type=int,default=240); parser.add_argument("--temperature",type=float,default=.8)
    parser.add_argument("--top-k",type=int,default=40); parser.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto")
    args=parser.parse_args()
    if args.max_new_tokens<1: parser.error("--max-new-tokens must be positive")
    device=choose_device(args.device); checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if checkpoint.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
    tokenizer=tokenizer_from_checkpoint(checkpoint); config=model_config_from_dict(checkpoint["model_config"])
    model=PopcornBrain(config).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    prompt_ids=tokenizer.encode(args.prompt) or [tokenizer.eos_token_id]
    result=model.generate(torch.tensor([prompt_ids],dtype=torch.long,device=device),args.max_new_tokens,args.temperature,args.top_k,tokenizer.eos_token_id)
    print(tokenizer.decode(result[0,len(prompt_ids):].tolist()))
if __name__=="__main__": main()
