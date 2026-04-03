API Reference
=============

Core
----

.. autoclass:: bauer.core.BaseModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.core.LapseModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.core.RegressionModel
   :members:
   :show-inheritance:

Psychometric models
-------------------

.. autoclass:: bauer.models.PsychometricModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.PsychometricLapseModel
   :show-inheritance:

.. autoclass:: bauer.models.PsychometricRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.PsychometricLapseRegressionModel
   :show-inheritance:

Magnitude comparison models
---------------------------

.. autoclass:: bauer.models.MagnitudeComparisonModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.MagnitudeComparisonLapseModel
   :show-inheritance:

.. autoclass:: bauer.models.MagnitudeComparisonRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.MagnitudeComparisonLapseRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.FlexibleNoiseComparisonModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.FlexibleNoiseComparisonRegressionModel
   :show-inheritance:

Risky choice models
-------------------

.. autoclass:: bauer.models.RiskModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.RiskLapseModel
   :show-inheritance:

.. autoclass:: bauer.models.RiskRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.RiskLapseRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.ProspectTheoryModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.LossAversionModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.LossAversionRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.RiskModelProbabilityDistortion
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.RNPModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.RNPRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.FlexibleNoiseRiskModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.FlexibleNoiseRiskRegressionModel
   :show-inheritance:

.. autoclass:: bauer.models.ExpectedUtilityRiskModel
   :members:
   :show-inheritance:

.. autoclass:: bauer.models.ExpectedUtilityRiskRegressionModel
   :show-inheritance:

Utilities
---------

.. autofunction:: bauer.utils.data.load_garcia2022

.. autofunction:: bauer.utils.data.load_dehollander2024

.. autofunction:: bauer.utils.bayes.get_posterior
.. autofunction:: bauer.utils.bayes.get_posterior_np
.. autofunction:: bauer.utils.bayes.get_diff_dist
.. autofunction:: bauer.utils.bayes.get_diff_dist_np
.. autofunction:: bauer.utils.bayes.cumulative_normal
.. autofunction:: bauer.utils.bayes.summarize_ppc

.. autofunction:: bauer.utils.math.logistic
.. autofunction:: bauer.utils.math.logistic_np
.. autofunction:: bauer.utils.math.softplus_np
.. autofunction:: bauer.utils.math.inverse_softplus_np
.. autofunction:: bauer.utils.math.logit
.. autofunction:: bauer.utils.math.logit_np
.. autofunction:: bauer.utils.math.logit_derivative
.. autofunction:: bauer.utils.math.gaussian_pdf

.. autofunction:: bauer.utils.plotting.plot_ppc
.. autofunction:: bauer.utils.plotting.plot_subjectwise_parameters
.. autofunction:: bauer.utils.plotting.plot_prediction
.. autofunction:: bauer.utils.plotting.cluster_offers
.. autofunction:: bauer.utils.plotting.get_hdi
