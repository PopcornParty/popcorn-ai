import argparse,json
from pathlib import Path
import torch
from src.popcorn_brain import PopcornBrain,model_config_from_dict,tokenizer_from_checkpoint
from train import choose_device

def main():
    parser=argparse.ArgumentParser(description="Run heuristic coding-output smoke checks; generated code is never executed")
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--benchmarks",type=Path,default=Path("brain/benchmarks/coding_smoke.jsonl"))
    parser.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto"); parser.add_argument("--max-new-tokens",type=int,default=180); args=parser.parse_args()
    device=choose_device(args.device); checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if checkpoint.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
    tokenizer=tokenizer_from_checkpoint(checkpoint); config=model_config_from_dict(checkpoint["model_config"])
    model=PopcornBrain(config).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval(); cases=[]
    for line_no,line in enumerate(args.benchmarks.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        item=json.loads(line)
        if not isinstance(item.get("prompt"),str) or not isinstance(item.get("must_include"),list): raise ValueError(f"Invalid benchmark case on line {line_no}")
        cases.append(item)
    if not cases: raise ValueError("No benchmark cases found")
    results=[]
    for case in cases:
        ids=tokenizer.encode(case["prompt"]) or [tokenizer.eos_token_id]
        generated=model.generate(torch.tensor([ids],dtype=torch.long,device=device),args.max_new_tokens,.01,1,tokenizer.eos_token_id)
        answer=tokenizer.decode(generated[0,len(ids):].tolist())
        missing=[fragment for fragment in case["must_include"] if fragment not in answer]
        forbidden=[fragment for fragment in case.get("must_not_include",[]) if fragment in answer]
        results.append({"id":case.get("id",f"case-{len(results)+1}"),"language":case.get("language","unspecified"),"passed":not missing and not forbidden,"missing_fragments":missing,"forbidden_fragments":forbidden,"output":answer[:500]})
    passed=sum(result["passed"] for result in results)
    print(json.dumps({"passed":passed,"total":len(results),"pass_rate":round(passed/len(results),4),"results":results,"warning":"This is a heuristic string-presence smoke benchmark, not a measure of semantic correctness or professional coding ability. No generated code is executed."},indent=2))
if __name__=="__main__": main()
