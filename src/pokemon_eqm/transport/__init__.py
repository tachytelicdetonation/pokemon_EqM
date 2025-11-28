from .transport import Transport, ModelType, WeightType, PathType, Sampler

def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    # Model architecture info (for aux losses)
    num_registers=0,
    # SIGReg parameters
    use_sigreg=False,
    sigreg_lambda=0.05,
    sigreg_num_slices=1024,
    # Auxiliary losses for attention improvement (2025 research)
    use_aux_losses=False,
    aux_entropy_floor_threshold=0.3,
    aux_entropy_floor_weight=0.02,
    aux_entropy_ceiling_threshold=0.85,
    aux_entropy_ceiling_weight=0.02,
    aux_gate_entropy_weight=0.01,
    aux_gate_sparsity_weight=0.005,
    aux_hsic_weight=0.01,
    aux_position_disagreement_weight=0.005,
    aux_lambda_smoothness_weight=0.001,
    aux_lambda_entropy_weight=0.01,
    aux_warmup_steps=1000,
    # Head specialization losses (2024-2025 research: MoH, orthogonality)
    aux_hard_focus_weight=0.02,
    aux_complexity_diversity_weight=0.01,
    aux_complexity_ortho_weight=0.01,
    aux_load_balance_weight=0.005,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    - use_sigreg: enable SIGReg loss on register tokens
    - sigreg_lambda: weight for SIGReg loss (default 0.05)
    - sigreg_num_slices: number of random projections for SIGReg (default 1024)
    - use_aux_losses: enable auxiliary attention losses
    - aux_entropy_floor_threshold: entropy floor for collapse prevention
    - aux_entropy_floor_weight: weight for entropy floor loss
    - aux_entropy_ceiling_threshold: entropy ceiling for uniformity prevention
    - aux_entropy_ceiling_weight: weight for entropy ceiling loss
    - aux_gate_entropy_weight: weight for gate binary entropy (GateRA)
    - aux_gate_sparsity_weight: weight for gate sparsity (D-Gating)
    - aux_hsic_weight: weight for HSIC head decorrelation
    - aux_position_disagreement_weight: weight for position disagreement
    - aux_lambda_smoothness_weight: weight for M-DGSA lambda smoothness
    - aux_lambda_entropy_weight: weight for M-DGSA lambda entropy
    - aux_warmup_steps: warmup steps for auxiliary losses
    - num_registers: number of register tokens (for aux loss computation)
    - aux_hard_focus_weight: weight for hard focus loss (anti-curriculum attention)
    - aux_complexity_diversity_weight: weight for complexity diversity loss
    - aux_complexity_ortho_weight: weight for complexity orthogonal loss
    - aux_load_balance_weight: weight for head load balancing loss
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else: # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0
    
    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        num_registers=num_registers,
        use_sigreg=use_sigreg,
        sigreg_lambda=sigreg_lambda,
        sigreg_num_slices=sigreg_num_slices,
        # Auxiliary losses
        use_aux_losses=use_aux_losses,
        aux_entropy_floor_threshold=aux_entropy_floor_threshold,
        aux_entropy_floor_weight=aux_entropy_floor_weight,
        aux_entropy_ceiling_threshold=aux_entropy_ceiling_threshold,
        aux_entropy_ceiling_weight=aux_entropy_ceiling_weight,
        aux_gate_entropy_weight=aux_gate_entropy_weight,
        aux_gate_sparsity_weight=aux_gate_sparsity_weight,
        aux_hsic_weight=aux_hsic_weight,
        aux_position_disagreement_weight=aux_position_disagreement_weight,
        aux_lambda_smoothness_weight=aux_lambda_smoothness_weight,
        aux_lambda_entropy_weight=aux_lambda_entropy_weight,
        aux_warmup_steps=aux_warmup_steps,
        # Head specialization losses
        aux_hard_focus_weight=aux_hard_focus_weight,
        aux_complexity_diversity_weight=aux_complexity_diversity_weight,
        aux_complexity_ortho_weight=aux_complexity_ortho_weight,
        aux_load_balance_weight=aux_load_balance_weight,
    )

    return state