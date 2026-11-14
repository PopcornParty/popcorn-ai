import argparse,json
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain,model_config_from_dict,tokenizer_from_checkpoint
from train import choose_device,load_texts

def main():
    parser=argparse.ArgumentParser(description="Measure held-out next-token loss")
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto"); args=parser.parse_args()
    device=choose_device(args.device); checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if checkpoint.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
    tokenizer=tokenizer_from_checkpoint(checkpoint); config=model_config_from_dict(checkpoint["model_config"])
    model=PopcornBrain(config).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    ids=[token for text in load_texts(args.data) for token in tokenizer.encode(text,True)]
    encoded=torch.tensor(ids,dtype=torch.long,device=device); weighted=0.0; count=0
    if len(encoded)<2: raise ValueError("Evaluation text is too short")
    with torch.no_grad():
        for start in range(0,len(encoded)-1,config.block_size):
            x=encoded[start:start+config.block_size]; y=encoded[start+1:start+config.block_size+1]
            if len(x)==0 or len(x)!=len(y): continue
            _,loss=model(x.unsqueeze(0),y.unsqueeze(0)); weighted+=float(loss.item())*len(y); count+=len(y)
    mean=weighted/count
    print(json.dumps({"checkpoint":str(args.checkpoint),"evaluation_file":str(args.data),"tokens_scored":count,"mean_next_token_loss":round(mean,6),"perplexity":round(float(torch.exp(torch.tensor(mean)).item()),6),"warning":"Next-token loss is not a measure of factual accuracy, reasoning or coding competence. Keep evaluation documents separate from training documents."},indent=2))
if __name__=="__main__": main()
