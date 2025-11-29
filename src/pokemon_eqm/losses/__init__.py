"""
Auxiliary losses for attention mechanism improvement.

Includes:
- AuxiliaryLossComputer: Entropy, gate, HSIC, and head specialization losses
- LejEPALoss: SIGReg, invariance, and JEPA prediction losses
"""
from .auxiliary_losses import AuxiliaryLossComputer
from .lejepa_loss import (
    LejEPALoss,
    # Core SIGReg components (matching lejepa repo naming)
    EppsPulley,
    EppsPulleyTest,  # Alias
    SlicingUnivariateTest,
    SlicedSIGReg,  # Alias
    CFSIGReg,
    # Additional components
    MultiViewInvarianceLoss,
    JEPAPredictor,
    JEPAPredictionLoss,
    MaskGenerator,
)

__all__ = [
    'AuxiliaryLossComputer',
    'LejEPALoss',
    # Core SIGReg (matching lejepa repo)
    'EppsPulley',
    'EppsPulleyTest',
    'SlicingUnivariateTest',
    'SlicedSIGReg',
    'CFSIGReg',
    # Additional components
    'MultiViewInvarianceLoss',
    'JEPAPredictor',
    'JEPAPredictionLoss',
    'MaskGenerator',
]
