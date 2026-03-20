import warnings

warnings.warn(
    "insurance-demand is deprecated. Use insurance-optimise instead:\n"
    "  pip install insurance-optimise\n"
    "  from insurance_optimise.demand import ConversionModel, RetentionModel\n"
    "  from insurance_optimise.demand import ElasticityEstimator, DemandCurve, OptimalPrice\n"
    "This package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location for backwards compatibility
from insurance_optimise.demand import *  # noqa: F401,F403
