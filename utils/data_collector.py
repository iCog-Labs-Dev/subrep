import os
import random
import re
import numpy as np
import torch
from env.skill_executor import SkillExecutor

class DataCollector:
    """
    Collects rollout outcomes (obs, payoff, motives, skill_id, terminated)
    and saves them to disk as .npz files for unbiased generator training.

    When available, also records `behavior_probability`, which is the
    probability that the behavior policy assigned to the selected skill/action
    at collection time. This field is required for future true IPS support.
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
        
    def collect_episode(self, skill_id: str = None, episode_seed: int = None) -> dict:
        """
        Run one episode via executor and return a data record.
        """
        if episode_seed is None:
            payoff, motives, terminated = self.executor.run_episode()
        else:
            initial_obs, _ = self.executor.env.reset(seed=episode_seed)
            payoff, motives, terminated = self.executor.run_episode(initial_obs=initial_obs)
        
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
        if episode_seed is not None:
            record['context_seed'] = int(episode_seed)
        behavior_probability = self.executor.last_run_info.get("behavior_probability")
        if behavior_probability is not None:
            record['behavior_probability'] = float(behavior_probability)
        return record

    def save_episode(self, record: dict, episode_idx: int, prefix: str = "random") -> str:
        """
        Save one record to data/raw/{prefix}_epNNN.npz.
        """
        filename = f"{prefix}_ep{episode_idx:03d}.npz"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "xb") as output_file:
            np.savez(
                output_file,
                obs=record['obs'],
                payoff=record['payoff'],
                motives=record['motives'],
                skill_id=record['skill_id'],
                terminated=record['terminated'],
                **({"context_seed": record["context_seed"]} if "context_seed" in record else {}),
                **({"behavior_probability": record["behavior_probability"]} if "behavior_probability" in record else {})
            )
        return filepath

    def _next_episode_index(self, prefix: str) -> int:
        pattern = re.compile(rf"{re.escape(prefix)}_ep(\d+)\.npz$")
        existing_indices = [
            int(match.group(1))
            for filename in os.listdir(self.save_dir)
            if (match := pattern.fullmatch(filename))
        ]
        return max(existing_indices, default=0) + 1

    def _next_context_seed(self) -> int:
        next_seed = int(self.seed)
        for filename in os.listdir(self.save_dir):
            if not filename.endswith(".npz"):
                continue
            filepath = os.path.join(self.save_dir, filename)
            with np.load(filepath, allow_pickle=True) as saved_record:
                if "context_seed" in saved_record.files:
                    next_seed = max(next_seed, int(saved_record["context_seed"]) + 1)
        return next_seed

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
        first_episode_idx = self._next_episode_index(skill_prefix)
        first_context_seed = self._next_context_seed()
        if n > 0 and (first_context_seed < 0 or first_context_seed + n > 2**32):
            raise ValueError("Context seed range must fit in the unsigned 32-bit range")
        for episode_idx in range(first_episode_idx, first_episode_idx + n):
            skill_id = f"{skill_prefix}_{episode_idx}"
            episode_seed = first_context_seed + episode_idx - first_episode_idx
            record = self.collect_episode(skill_id=skill_id, episode_seed=episode_seed)
            self.save_episode(record, episode_idx, prefix=skill_prefix)
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
