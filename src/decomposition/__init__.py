# Decomposition methods for transformer layers
from .tucker import (
    decompose_fc_layer,
    reconstruct_weights,
    remove_component,
    remove_multiple_components,
    get_fc_layer_weights,
    compute_reconstruction_error
)

from .cp import (
    decompose_fc_layer_cp,
    reconstruct_weights_cp,
    remove_cp_component,
    reconstruct_from_cp
)

from .svd import (
    decompose_weight_svd,
    reconstruct_from_svd,
    truncate_svd,
    low_rank_approximation,
    compute_energy_retention,
    compute_reconstruction_error_svd,
    get_svd_stats,
    apply_svd_to_layer,
    update_layer_with_svd,
    restore_original_weight,
    reduction_to_keep_ratio,
    LASER_REDUCTION_PERCENTAGES
)

from .model_utils import (
    update_fc_layer_weights,
    test_model_generation
)
