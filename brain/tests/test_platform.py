import tempfile,unittest
from pathlib import Path
from src.coding import CodeTask,PatchProposal,ProposedFile
from src.evolution import evaluate_candidate
from src.memory import MemoryConsentError,MemoryStore
from src.planner import TaskPlan
class PlatformTests(unittest.TestCase):
 def test_memory_requires_confirmation_and_can_be_deleted(self):
  with tempfile.TemporaryDirectory() as folder:
   store=MemoryStore(Path(folder)/"memory.sqlite3")
   with self.assertRaises(MemoryConsentError): store.remember("Prefers concise answers",kind="preference",source="chat",user_confirmed=False)
   mid=store.remember("Prefers concise answers",kind="preference",source="confirmed",user_confirmed=True)
   self.assertEqual(len(store.search("concise")),1); self.assertTrue(store.delete(mid)); self.assertEqual(store.search("concise"),[])
 def test_memory_rejects_secret_like_values(self):
  with tempfile.TemporaryDirectory() as folder:
   with self.assertRaises(MemoryConsentError): MemoryStore(Path(folder)/"m.db").remember("api_key = 'abcdefghijk12345'",kind="fact",source="chat",user_confirmed=True)
 def test_planner_rejects_unknown_tool(self):
  with self.assertRaises(ValueError): TaskPlan.from_dict({"goal":"x","steps":[{"title":"x","description":"x","tool":"shell"}]})
 def test_candidate_patch_is_descriptive_and_does_not_need_live_approval(self):
  plan=TaskPlan.from_dict({"goal":"x","steps":[{"title":"patch","description":"edit","tool":"propose_patch"}]})
  self.assertFalse(plan.steps[0].requires_approval)
 def test_plan_round_trip(self):
  plan=TaskPlan.from_dict({"goal":"inspect","steps":[{"title":"Read","description":"Read files","tool":"read_repository_file"}]}); self.assertEqual(TaskPlan.from_dict(plan.to_dict()),plan)
 def test_code_task_and_safe_patch(self):
  CodeTask("Fix a bug","python").validate(); PatchProposal("Fix",(ProposedFile("src/fix.py","def fixed(): return True"),),("unit tests",)).validate()
  with self.assertRaises(ValueError): PatchProposal("unsafe",(ProposedFile("../secret.txt","no"),)).validate()
 def test_patch_rejects_secret(self):
  with self.assertRaises(ValueError): PatchProposal("bad",(ProposedFile("config.py","api_key = 'abcdefghijk12345'"),)).validate()
 def test_evolution_gate_blocks_regression(self):
  report=evaluate_candidate(baseline_score=.8,candidate_score=.9,baseline_metrics={"coding":.8,"chat":.8},candidate_metrics={"coding":.9,"chat":.7},required_checks={"tests":True})
  self.assertFalse(report.eligible_for_human_review); self.assertTrue(any("Regression" in x for x in report.reasons))
 def test_evolution_gate_requires_all_checks(self):
  report=evaluate_candidate(baseline_score=.7,candidate_score=.9,baseline_metrics={"coding":.7},candidate_metrics={"coding":.9},required_checks={"tests":True,"safety":False}); self.assertFalse(report.eligible_for_human_review)
if __name__=="__main__": unittest.main()
