import argparse,json,re
from pathlib import Path
PATTERNS=[re.compile(r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),re.compile(r"(?i)-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),re.compile(r"(?i)password\s*[:=]\s*\S{6,}")]
def inspect_dataset(path:Path):
 seen=set(); duplicates=empty=total=characters=0; suspicious=[]
 for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  if not line.strip(): continue
  total+=1
  try: item=json.loads(line)
  except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON on line {n}: {exc}") from exc
  if not isinstance(item,dict) or not isinstance(item.get("text"),str): raise ValueError(f"Line {n} must contain a string text field")
  value=item["text"].strip()
  if not value: empty+=1; continue
  characters+=len(value)
  if value in seen: duplicates+=1
  seen.add(value)
  if any(p.search(value) for p in PATTERNS): suspicious.append(n)
 return {"file":str(path),"records":total,"unique_nonempty_records":len(seen),"duplicate_records":duplicates,"empty_records":empty,"characters":characters,"possible_secret_line_numbers":suspicious,"ready_for_training":total>=3 and len(seen)>=3 and not suspicious,"warning":"Automated secret detection is incomplete. Manually review data rights and contents before training."}
def main():
 p=argparse.ArgumentParser(description="Validate JSONL data without printing its contents"); p.add_argument("--data",type=Path,required=True); a=p.parse_args(); report=inspect_dataset(a.data); print(json.dumps(report,indent=2))
 if not report["ready_for_training"]: raise SystemExit(1)
if __name__=="__main__": main()
