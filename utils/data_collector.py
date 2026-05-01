import os
import random
import numpy as np
import torch
from env.skill_executor import SkillExecutor

class DataCollector:
    """
    Collects rollout outcomes (obs, payoff, motives, skill_id, terminated)
    and saves them to disk as .npz files for unbiased generator training.
    """
    def __init__(
        self,
        executor: SkillExecutor,
        seed: int = 42,
        save_dir: str = "data/raw"
    ) -> None:
        self.executor = executor
        self.seed = seed
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Robust seeding for full reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def collect_episode(self, skill_id: str = None) -> dict:
        """
        Run one episode via executor and return a data record.
        """
        payoff, motives, terminated = self.executor.run_episode()
        
        # Ensure we get initial_obs from the latest run
        initial_obs = self.executor.last_run_info.get("initial_obs")
        if initial_obs is None:
            raise ValueError("SkillExecutor did not record initial_obs in last_run_info.")
            
        record = {
            'obs': np.asarray(initial_obs, dtype=np.float32),
            'payoff': float(payoff),
            'motives': np.asarray(motives, dtype=np.float32),
            'skill_id': skill_id if skill_id is not None else "unknown",
            'terminated': bool(terminated)
        }
        return record

    def save_episode(self, record: dict, episode_idx: int, prefix: str = "random") -> str:
        """
        Save one record to data/raw/{prefix}_epNNN.npz.
        """
        filename = f"{prefix}_ep{episode_idx:03d}.npz"
        filepath = os.path.join(self.save_dir, filename)
        np.savez(
            filepath,
            obs=record['obs'],
            payoff=record['payoff'],
            motives=record['motives'],
            skill_id=record['skill_id'],
            terminated=record['terminated']
        )
        return filepath

    def collect_n_episodes(
        self,
        n: int,
        print_summary: bool = True,
        skill_prefix: str = "random"
    ) -> list[dict]:
        """
        Run N episodes, save each to disk with prefix, optionally print summary.
        """
        records = []
        for i in range(1, n + 1):
            skill_id = f"{skill_prefix}_{i}"
            record = self.collect_episode(skill_id=skill_id)
            self.save_episode(record, i, prefix=skill_prefix)
            records.append(record)
            
        if print_summary:
            self.print_summary(records)
            
        return records

    def print_summary(self, records: list[dict]) -> None:
        """
        Print to console the outcome summary for the collected batch.
        """
        if not records:
            print("No records collected.")
            return
            
        payoffs = [r['payoff'] for r in records]
        motives = np.array([r['motives'] for r in records])
        terminated_count = sum(1 for r in records if r['terminated'])
        
        print("\n=== Data Collection Summary ===")
        print(f"Total episodes       : {len(records)}")
        print(f"Mean Payoff          : {np.mean(payoffs):.4f} ± {np.std(payoffs):.4f}")
        print(f"Mean Motives         : Safety_delta={np.mean(motives[:, 0]):.4f}, Fuel_delta={np.mean(motives[:, 1]):.4f}")
        print(f"Naturally Terminated : {terminated_count} ({(terminated_count/len(records))*100:.1f}%)")
        print("===============================\n")
