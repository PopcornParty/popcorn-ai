import argparse,json
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain,model_config_from_dict,tokenizer_from_checkpoint
from train import choose_device,load_texts
def main():
 p=argparse.ArgumentParser(description="Measure held-out next-token loss"); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto"); a=p.parse_args()
 device=choose_device(a.device); ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
 if ckpt.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
 tok=tokenizer_from_checkpoint(ckpt); config=model_config_from_dict(ckpt["model_config"]); model=PopcornBrain(config).to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
 ids=[t for text in load_texts(a.data) for t in tok.encode(text,True)]; data=torch.tensor(ids,dtype=torch.long,device=device); weighted=count=0
 if len(data)<2: raise ValueError("Evaluation text is too short")
 with torch.no_grad():
  for start in range(0,len(data)-1,config.block_size):
   x=data[start:start+config.block_size]; y=data[start+1:start+config.block_size+1]
   if len(x)==0 or len(x)!=len(y): continue
   _,loss=model(x.unsqueeze(0),y.unsqueeze(0)); weighted+=float(loss.item())*len(y); count+=len(y)
 mean=weighted/count
 print(json.dumps({"checkpoint":str(a.checkpoint),"evaluation_file":str(a.data),"tokens_scored":count,"mean_next_token_loss":round(mean,6),"perplexity":round(float(torch.exp(torch.tensor(mean)).item()),6),"warning":"Next-token loss is not a measure of factual accuracy, reasoning or coding competence. Keep evaluation documents separate from training documents."},indent=2))
if __name__=="__main__": main()
