"""Validated task plans. Plans describe work and never execute tools themselves."""
from dataclasses import dataclass
ALLOWED_TOOL_NAMES={"search_knowledge","read_repository_file","propose_patch","run_approved_test","summarise_result"}
@dataclass(frozen=True)
class PlanStep:
 title:str
 description:str
 tool:str|None=None
 requires_approval:bool=False
@dataclass(frozen=True)
class TaskPlan:
 goal:str
 steps:tuple[PlanStep,...]
 risk:str="low"
 @classmethod
 def from_dict(cls,value:dict):
  if not isinstance(value,dict): raise ValueError("Plan must be an object")
  goal,raw,risk=value.get("goal"),value.get("steps"),value.get("risk","low")
  if not isinstance(goal,str) or not goal.strip() or len(goal)>1000: raise ValueError("Plan goal must contain 1–1000 characters")
  if not isinstance(raw,list) or not 1<=len(raw)<=20: raise ValueError("A plan must contain between 1 and 20 steps")
  if risk not in {"low","medium","high"}: raise ValueError("risk must be low, medium or high")
  steps=[]
  for n,item in enumerate(raw,1):
   if not isinstance(item,dict): raise ValueError(f"Step {n} must be an object")
   title,desc,tool,approval=item.get("title"),item.get("description"),item.get("tool"),item.get("requires_approval",False)
   if not isinstance(title,str) or not title.strip() or len(title)>120: raise ValueError(f"Step {n} has an invalid title")
   if not isinstance(desc,str) or not desc.strip() or len(desc)>1000: raise ValueError(f"Step {n} has an invalid description")
   if tool is not None and (not isinstance(tool,str) or tool not in ALLOWED_TOOL_NAMES): raise ValueError(f"Step {n} requests an unapproved tool")
   if not isinstance(approval,bool): raise ValueError(f"Step {n} requires_approval must be boolean")
   if tool in {"propose_patch","run_approved_test"} and not approval: raise ValueError(f"Step {n} requires explicit approval for code changes or test execution")
   steps.append(PlanStep(title.strip(),desc.strip(),tool,approval))
  return cls(goal.strip(),tuple(steps),risk)
 def to_dict(self): return {"goal":self.goal,"risk":self.risk,"steps":[s.__dict__ for s in self.steps]}
