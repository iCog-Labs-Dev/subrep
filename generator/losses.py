import torch
import torch.nn.functional as F
from torch import Tensor

class GeneratorLoss:
    """
    Combined MSE loss for the 2-head SkillGenerator.
    
    total_loss = payoff_weight * MSE(pred_payoff, actual_payoff)
               + motive_weight * MSE(pred_motives, actual_motives)
    """

    def __init__(self, payoff_weight: float = 1.0, motive_weight: float = 1.0):
        self.payoff_weight = float(payoff_weight)
        self.motive_weight = float(motive_weight)

    def __call__(
        self,
        pred_payoff: Tensor,
        pred_motives: Tensor,
        target_payoff: Tensor,
        target_motives: Tensor,
    ) -> Tensor:
        """
        Compute the weighted sum of MSE losses.
        """
        loss_dict = self.breakdown(pred_payoff, pred_motives, target_payoff, target_motives)
        return loss_dict["total_loss"]

    def breakdown(
        self,
        pred_payoff: Tensor,
        pred_motives: Tensor,
        target_payoff: Tensor,
        target_motives: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute individual MSE losses and the weighted total loss.
        """
        # Ensure target shapes match prediction shapes
        target_payoff = target_payoff.view_as(pred_payoff)
        target_motives = target_motives.view_as(pred_motives)

        payoff_loss = F.mse_loss(pred_payoff, target_payoff)
        motive_loss = F.mse_loss(pred_motives, target_motives)

        total_loss = self.payoff_weight * payoff_loss + self.motive_weight * motive_loss

        return {
            "payoff_loss": payoff_loss,
            "motive_loss": motive_loss,
            "total_loss": total_loss,
        }
