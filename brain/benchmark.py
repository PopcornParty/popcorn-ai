import argparse,json
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain,model_config_from_dict,tokenizer_from_checkpoint
from train import choose_device
def main():
 p=argparse.ArgumentParser(description="Run heuristic coding smoke checks; generated code is never executed"); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--benchmarks",type=Path,default=Path("brain/benchmarks/coding_smoke.jsonl")); p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto"); p.add_argument("--max-new-tokens",type=int,default=180); a=p.parse_args()
 device=choose_device(a.device); ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
 if ckpt.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
 tok=tokenizer_from_checkpoint(ckpt); config=model_config_from_dict(ckpt["model_config"]); model=PopcornBrain(config).to(device); model.load_state_dict(ckpt["model_state"]); model.eval(); cases=[]
 for n,line in enumerate(a.benchmarks.read_text(encoding="utf-8").splitlines(),1):
  if not line.strip(): continue
  item=json.loads(line)
  if not isinstance(item.get("prompt"),str) or not isinstance(item.get("must_include"),list): raise ValueError(f"Invalid benchmark case on line {n}")
  cases.append(item)
 if not cases: raise ValueError("No benchmark cases found")
 results=[]
 for case in cases:
  ids=tok.encode(case["prompt"]) or [tok.eos_token_id]; generated=model.generate(torch.tensor([ids],dtype=torch.long,device=device),a.max_new_tokens,.01,1,tok.eos_token_id); answer=tok.decode(generated[0,len(ids):].tolist())
  missing=[x for x in case["must_include"] if x not in answer]; forbidden=[x for x in case.get("must_not_include",[]) if x in answer]
  results.append({"id":case.get("id",f"case-{len(results)+1}"),"language":case.get("language","unspecified"),"passed":not missing and not forbidden,"missing_fragments":missing,"forbidden_fragments":forbidden,"output":answer[:500]})
 passed=sum(r["passed"] for r in results); print(json.dumps({"passed":passed,"total":len(results),"pass_rate":round(passed/len(results),4),"results":results,"warning":"Heuristic string matching is not semantic correctness or professional coding ability. Generated code is never executed."},indent=2))
if __name__=="__main__": main()
