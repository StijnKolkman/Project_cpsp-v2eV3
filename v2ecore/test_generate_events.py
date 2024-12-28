import torch
from typing import Optional
import numpy as np

class EventEmulator:
    SCIDVS_TAU_S = 0.001  # Example time constant for small signals

    def scidvs_dvdt(self, v: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the time derivative of the signal.
        """
        if tau is None:
            tau = EventEmulator.SCIDVS_TAU_S  # Default time constant
        efold = 1 / 0.7  # efold of sinh conductance in log_e units, based on 1/kappa
        dvdt = torch.div(1, tau) * torch.sinh(v / efold)
        return dvdt


def test_scidvs_dvdt():
    """Test the scidvs_dvdt function."""
    emulator = EventEmulator()

    # Test parameters
    test_values = [
        torch.tensor([0.0]),  # Zero input
        torch.tensor([1.0, -1.0]),  # Symmetric inputs
        torch.tensor([10.0, -10.0, 0.0]),  # Extreme values
    ]
    taus = [
        None,  # Use default tau
        torch.tensor([0.01]),  # Custom scalar tau
        torch.tensor([0.005, 0.01, 0.02]),  # Custom vector tau for batch inputs
    ]

    for i, v in enumerate(test_values):
        for j, tau in enumerate(taus):
            try:
                print(f"Test {i}-{j}: Input = {v.numpy()}, Tau = {tau.numpy() if tau is not None else 'default'}")
                dvdt = emulator.scidvs_dvdt(v, tau)
                print(f"Output dv/dt: {dvdt.numpy()}")
            except Exception as e:
                print(f"Test {i}-{j} failed with error: {e}")

if __name__ == "__main__":
    test_scidvs_dvdt()
