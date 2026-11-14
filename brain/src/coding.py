"""Safe coding task and patch proposal validation; no code is executed or written."""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
SUPPORTED_LANGUAGES={"python","javascript","typescript","java","csharp","cpp","c","html","css","sql","go","rust","php","ruby","swift","kotlin"}
SECRET_RE=re.compile(r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}")
@dataclass(frozen=True)
class CodeTask:
 goal:str
 language:str
 repository:str|None=None
 constraints:tuple[str,...]=()
 def validate(self):
  if not isinstance(self.language,str) or self.language.lower() not in SUPPORTED_LANGUAGES: raise ValueError("Unsupported programming language")
  if not isinstance(self.goal,str) or not self.goal.strip() or len(self.goal)>4000: raise ValueError("Task goal must contain 1–4000 characters")
  if len(self.constraints)>30 or any(not isinstance(x,str) or len(x)>500 for x in self.constraints): raise ValueError("Invalid task constraints")
@dataclass(frozen=True)
class ProposedFile:
 path:str
 content:str
@dataclass(frozen=True)
class PatchProposal:
 summary:str
 files:tuple[ProposedFile,...]
 tests_to_run:tuple[str,...]=()
 def validate(self):
  if not self.summary.strip() or len(self.summary)>1000: raise ValueError("Patch summary must contain 1–1000 characters")
  if not 1<=len(self.files)<=20: raise ValueError("A patch must contain between 1 and 20 files")
  seen=set(); total=0
  for item in self.files:
   path=PurePosixPath(item.path)
   if path.is_absolute() or not item.path or any(p in {"",".",".."} for p in item.path.split("/")): raise ValueError("Patch contains an unsafe file path")
   if item.path in seen: raise ValueError("Patch contains duplicate file paths")
   seen.add(item.path)
   if len(item.content)>100000: raise ValueError("A proposed file exceeds 100 KB")
   if SECRET_RE.search(item.content): raise ValueError("Patch appears to contain a credential or secret")
   total+=len(item.content)
  if total>500000: raise ValueError("Patch exceeds the 500 KB total limit")
  if len(self.tests_to_run)>30 or any(not isinstance(x,str) or len(x)>300 for x in self.tests_to_run): raise ValueError("Invalid test list")
