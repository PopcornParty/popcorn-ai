"""Evidence-based candidate evaluation. This module never deploys or promotes weights."""
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class EvaluationReport:
 eligible_for_human_review:bool
 reasons:tuple[str,...]
 baseline_score:float
 candidate_score:float
 metrics:dict[str,float]
def evaluate_candidate(*,baseline_score:float,candidate_score:float,baseline_metrics:dict[str,float],candidate_metrics:dict[str,float],required_checks:dict[str,bool],minimum_gain:float=.01,max_regression:float=.02):
 vals=[baseline_score,candidate_score,minimum_gain,max_regression,*baseline_metrics.values(),*candidate_metrics.values()]
 if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals): raise ValueError("Scores and metrics must be finite numbers")
 if not 0<=baseline_score<=1 or not 0<=candidate_score<=1 or not 0<=minimum_gain<=1 or not 0<=max_regression<=1: raise ValueError("Scores and thresholds must be between 0 and 1")
 reasons=[]
 if not required_checks: reasons.append("No required checks were supplied")
 for name,passed in required_checks.items():
  if passed is not True: reasons.append(f"Required check failed: {name}")
 if candidate_score<baseline_score+minimum_gain: reasons.append("Candidate did not meet the minimum overall improvement")
 if set(baseline_metrics)!=set(candidate_metrics): reasons.append("Baseline and candidate metric sets do not match")
 else:
  for name,base in baseline_metrics.items():
   candidate=candidate_metrics[name]
   if not 0<=base<=1 or not 0<=candidate<=1: raise ValueError(f"Metric '{name}' must be between 0 and 1")
   if candidate<base-max_regression: reasons.append(f"Regression exceeds tolerance: {name}")
 return EvaluationReport(not reasons,tuple(reasons),baseline_score,candidate_score,dict(candidate_metrics))
