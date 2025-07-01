# In src/coordination/gnn_coordinator.py

# ... (imports and class init remain the same as the previous turn) ...
# ... (including the 'seed' argument in __init__) ...
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from .bio_optimizer import BioOptimizer

logger = logging.getLogger("hybrid_ai_brain.gnn_coordinator")
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class GNNCoordinator:
    def __init__(
        self,
        spectral_norm_bound: float = 0.7,
        temperature: float = 1.0,
        embedding_dim: int = 64,
        gnn_layers: int = 2,
        seed: Optional[int] = None
    ):
        if not 0 < spectral_norm_bound < 1:
            raise ValueError("spectral_norm_bound must be between 0 and 1.")

        self.l_total_bound = spectral_norm_bound
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.rng = np.random.RandomState(seed)
        logger.info(f"GNNCoordinator initialized (L_total < {self.l_total_bound}, β={self.temperature})")

    def _get_base_embeddings(self, nodes: List[str]) -> Dict[str, np.ndarray]:
        embeddings = {}
        for node in nodes:
            embeddings[node] = self.rng.randn(self.embedding_dim)
        return embeddings

    def assign_tasks(
        self,
        tasks: List[str],
        agents: List[str],
        bio_signals: Dict[str, Any]
    ) -> Dict[str, str]:
        logger.info("GNNCoordinator: Starting task assignment, leveraging bio-signals.")
        all_nodes = tasks + agents
        embeddings = self._get_base_embeddings(all_nodes)
        
        pso_g_best = bio_signals.get("pso_global_best")
        pheromones = bio_signals.get("pheromone_levels", {})
        lambda_pso, lambda_aco = bio_signals.get("conflict_weights", (0.5, 0.5))

        assignments = {}
        for task in tasks:
            agent_scores = {}
            for agent in agents:
                agent_emb = embeddings[agent]
                
                # --- CORRECTED & STABLE SCORING LOGIC ---

                # 1. Calculate the ACO score (historical influence).
                aco_score = pheromones.get(f"({task}, {agent})", 0.0)
                
                # 2. Calculate the PSO score (tactical influence).
                pso_score = 0.0
                if pso_g_best is not None:
                    # Score is the agent's similarity to the PSO's best-found solution vector.
                    pso_similarity = np.dot(agent_emb, pso_g_best) / (np.linalg.norm(agent_emb) * np.linalg.norm(pso_g_best) + 1e-8)
                    pso_score = pso_similarity

                # 3. Final score is a pure weighted sum of the bio-signal scores.
                # This makes the decision deterministic and directly controlled by ABC's weights.
                final_score = (lambda_pso * pso_score) + (lambda_aco * aco_score)
                agent_scores[agent] = final_score
            
            # --- Softmax and assignment logic (unchanged) ---
            scores_np = np.array(list(agent_scores.values()))
            scores_np *= self.temperature
            exp_scores = np.exp(scores_np - np.max(scores_np))
            probs = exp_scores / np.sum(exp_scores)
            
            best_agent = agents[np.argmax(probs)]
            assignments[task] = best_agent
            
        logger.info(f"GNNCoordinator: Final assignments computed: {assignments}")
        return assignments