# insurance-demand — Deprecated

This package has been superseded by [insurance-optimise](https://github.com/burning-cost/insurance-optimise).

All functionality — `ConversionModel`, `RetentionModel`, `ElasticityEstimator`, `DemandCurve`, `OptimalPrice`, `OptimisationResult`, and `ENBPChecker` — is now part of insurance-optimise under the `insurance_optimise.demand` subpackage.

## Migration

```bash
pip install insurance-optimise
```

```python
# Before
from insurance_demand import ConversionModel, RetentionModel, DemandCurve

# After
from insurance_optimise.demand import ConversionModel, RetentionModel, DemandCurve
```

This repository is archived and will not receive further updates.
