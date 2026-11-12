"""Manager for the Model Registration pipeline step."""

import logging

from app.logger import setup_logging
from app.models.registration import RegistrationParams, RegistrationResult
from app.services.model_registration import ModelRegistrationService


class Manager:
    def __init__(self) -> None:
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._service = ModelRegistrationService()

    def run(self, params: RegistrationParams) -> RegistrationResult:
        """Execute the model registration step."""
        self._logger.info(
            "Starting model-registration | run_id=%s model=%s",
            params.mlflow_run_id,
            params.registered_model_name,
        )
        result = self._service.run(params=params)
        self._logger.info(
            "Registration complete | model=%s best_version=%s last_version=%s promoted_to=%s",
            result.registered_model_name,
            result.best_version,
            result.last_version,
            result.promoted_to,
        )
        return result
