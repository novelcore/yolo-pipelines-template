"""Manager for the config-validation step."""

import logging

from app.logger import setup_logging
from app.services.config_validation import ConfigValidationService


class Manager:
    def __init__(self) -> None:
        setup_logging()
        self._log = logging.getLogger(__name__)
        self._service = ConfigValidationService()

    def run(self, resolved: dict, app: str, experiment: str) -> None:
        self._log.info("Starting config-validation")
        self._service.run(resolved, app, experiment)
        self._log.info("Config validation completed successfully")
