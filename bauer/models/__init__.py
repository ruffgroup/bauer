from .psychophysics import (
    PsychophysicalModel, PsychophysicalLapseModel,
    PsychophysicalRegressionModel, PsychophysicalLapseRegressionModel,
    # Deprecated aliases — emit DeprecationWarning on instantiation
    PsychometricModel, PsychometricLapseModel,
    PsychometricRegressionModel, PsychometricLapseRegressionModel,
)
from .magnitude import (
    MagnitudeComparisonModel, MagnitudeComparisonRegressionModel,
    MagnitudeComparisonLapseModel, MagnitudeComparisonLapseRegressionModel,
    FlexibleNoiseComparisonModel, FlexibleNoiseComparisonRegressionModel,
    PowerLawNoiseComparisonModel, PowerLawNoiseComparisonRegressionModel,
    PowerLawEncodingComparisonModel, PowerLawEncodingComparisonRegressionModel,
)
from .risky_choice import (
    RiskModelProbabilityDistortion, ProspectTheoryModel,
    LossAversionModel, LossAversionRegressionModel,
    RiskModel, RiskRegressionModel, RiskLapseModel, RiskLapseRegressionModel,
    FlexibleNoiseRiskModel, FlexibleNoiseRiskRegressionModel,
    ExpectedUtilityRiskModel, ExpectedUtilityRiskRegressionModel,
    PowerLawNoiseRiskModel, PowerLawNoiseRiskRegressionModel,
    AffineNoiseRiskModel, AffineNoiseRiskRegressionModel,
    PowerLawEncodingRiskModel, PowerLawEncodingRiskRegressionModel,
)

try:
    from .ddm import (
        DDMMixin,
        DDMLapseMixin,
        DDMMagnitudeComparisonModel,
        DDMMagnitudeComparisonRegressionModel,
        DDMMagnitudeComparisonLapseModel,
        DDMMagnitudeComparisonLapseRegressionModel,
        DDMFlexibleNoiseComparisonModel,
        DDMPowerLawNoiseComparisonModel,
        DDMPowerLawNoiseComparisonRegressionModel,
        DDMRiskModel,
        DDMRiskRegressionModel,
        DDMRiskLapseModel,
        DDMRiskLapseRegressionModel,
        DDMFlexibleNoiseRiskModel,
        DDMFlexibleNoiseRiskRegressionModel,
        DDMPowerLawNoiseRiskModel,
        DDMPowerLawNoiseRiskRegressionModel,
    )
except ImportError:
    pass

from .legacy import (
    SafeVsRiskyModel,
    SafeVsRiskyRegressionModel,
    SafeVsRiskyMemoryModel,
    JointSafeVsRiskyModel,
)

from .race import (
    RaceMixin,
    RaceLapseMixin,
    RaceDiffusionMagnitudeComparisonModel,
    RaceDiffusionMagnitudeComparisonLapseModel,
    RaceDiffusionFlexibleNoiseComparisonModel,
    RaceDiffusionPowerLawNoiseComparisonModel,
    RaceDiffusionPowerLawNoiseComparisonRegressionModel,
    RaceDiffusionRiskModel,
    RaceDiffusionRiskRegressionModel,
    RaceDiffusionRiskLapseModel,
    RaceDiffusionRiskLapseRegressionModel,
    RaceDiffusionFlexibleNoiseRiskModel,
    RaceDiffusionFlexibleNoiseRiskRegressionModel,
    RaceDiffusionPowerLawNoiseRiskModel,
    RaceDiffusionPowerLawNoiseRiskRegressionModel,
    logp_race_diffusion_2,
)
