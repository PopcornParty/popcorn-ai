from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from src.popcorn_brain import ByteTokenizer, ModelConfig, PopcornBrain, tokenizer_from_checkpoint

def load_texts(path: Path) -> list[str]:
    texts, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: item = json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON on line {number}: {exc}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip(): raise ValueError(f"Line {number} must contain a non-empty string field named 'text'")
        value = item["text"].strip()
        if value not in seen: texts.append(value); seen.add(value)
    if len(texts) < 3: raise ValueError("Provide at least three unique non-empty training examples")
    return texts

def choose_device(requested: str):
    if requested == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()): raise RuntimeError("MPS was requested but is unavailable")
    return device

def main():
    parser = argparse.ArgumentParser(description="Train or resume the experimental Popcorn Transformer")
    parser.add_argument("--data", type=Path, required=True); parser.add_argument("--out", type=Path, default=Path("brain/checkpoints/experiment"))
    parser.add_argument("--max-iters", type=int, default=1000, help="Total optimiser steps, not additional steps")
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100); parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Resume from a trusted Popcorn checkpoint")
    args = parser.parse_args()
    if min(args.max_iters,args.batch_size,args.block_size,args.eval_interval) < 1 or args.block_size < 8 or args.learning_rate <= 0: parser.error("Training values must be positive; block size must be at least 8")
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    shuffled = load_texts(args.data); random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled)*0.1)); val_texts, train_texts = shuffled[:val_count], shuffled[val_count:]
    tokenizer = ByteTokenizer()
    train_stream = torch.tensor([t for text in train_texts for t in tokenizer.encode(text, True)], dtype=torch.long)
    val_stream = torch.tensor([t for text in val_texts for t in tokenizer.encode(text, True)], dtype=torch.long)
    for name, stream in (("training",train_stream),("validation",val_stream)):
        if len(stream) < args.block_size + 2: raise ValueError(f"{name.title()} split has {len(stream)} bytes; need at least {args.block_size+2}. Add distinct documents or lower --block-size.")
    device=choose_device(args.device); config=ModelConfig(vocab_size=tokenizer.vocab_size,block_size=args.block_size)
    model=PopcornBrain(config).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    args.out.mkdir(parents=True,exist_ok=True); start_step=0; history=[]; best_val=float("inf")
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        if checkpoint.get("format_version") not in (1,2): raise ValueError("Unsupported checkpoint format")
        if tokenizer_from_checkpoint(checkpoint).tokenizer_type != tokenizer.tokenizer_type: raise ValueError("Resume checkpoint uses a different tokenizer; start a new run.")
        if checkpoint["model_config"].get("block_size") != args.block_size: raise ValueError("Resume block size must match checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"): optimizer.load_state_dict(checkpoint["optimizer_state"])
        state=checkpoint.get("training",{}); start_step=int(state.get("next_step",state.get("step",0)))
        history=list(state.get("metrics",[])); best_val=float(state.get("best_validation_loss",float("inf")))
    def get_batch(source):
        starts=torch.randint(len(source)-args.block_size,(args.batch_size,))
        x=torch.stack([source[int(i):int(i)+args.block_size] for i in starts]).to(device)
        y=torch.stack([source[int(i)+1:int(i)+args.block_size+1] for i in starts]).to(device)
        return x,y
    @torch.no_grad()
    def estimate_loss():
        model.eval(); report={}
        for name,source in (("train",train_stream),("validation",val_stream)):
            values=[]
            for _ in range(8):
                x,y=get_batch(source); _,loss=model(x,y); values.append(float(loss.item()))
            report[name]=sum(values)/len(values)
        model.train(); return report
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; tokenizer: byte-v1; training bytes: {len(train_stream)}; validation bytes: {len(val_stream)}")
    if start_step >= args.max_iters: raise ValueError(f"Checkpoint is already at step {start_step}; set --max-iters higher to continue")
    for step in range(start_step,args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters-1:
            metrics=estimate_loss(); history.append({"step":step,**metrics}); improved=metrics["validation"]<best_val
            if improved: best_val=metrics["validation"]
            payload={"format_version":2,"tokenizer":tokenizer.to_dict(),"model_config":config.__dict__,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"training":{"step":step,"next_step":step,"seed":args.seed,"device":str(device),"best_validation_loss":best_val,"metrics":history}}
            torch.save(payload,args.out/"latest.pt")
            if improved or not (args.out/"best.pt").exists(): torch.save(payload,args.out/"best.pt")
            (args.out/"metrics.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
            print(f"step {step:5d} | train {metrics['train']:.4f} | validation {metrics['validation']:.4f} | best {best_val:.4f}")
        x,y=get_batch(train_stream); _,loss=model(x,y); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    print(f"Saved latest and best checkpoints in {args.out}")
    print("Review validation metrics and independent benchmarks before trusting this model.")

if __name__=="__main__": main()
