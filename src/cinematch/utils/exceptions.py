"""Custom exception hierarchy for CineMatch."""

from __future__ import annotations


class CineMatchError(Exception):
    """Base exception for all CineMatch domain and application errors."""

    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class ConfigurationError(CineMatchError):
    """Raised when application configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500, error_code="CONFIGURATION_ERROR")


class DatasetNotFoundError(CineMatchError):
    """Raised when the required dataset file cannot be found."""

    def __init__(self, filepath: str) -> None:
        super().__init__(
            f"Dataset file not found at '{filepath}'. Place 'movies_metadata.csv' in 'data/movies_metadata.csv' "
            "or set the CINEMATCH_DATASET_PATH environment variable.",
            status_code=503,
            error_code="DATASET_NOT_FOUND",
        )
        self.filepath = filepath


class DataValidationError(CineMatchError):
    """Raised when the dataset fails schema or data quality validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422, error_code="DATA_VALIDATION_ERROR")


class ModelNotTrainedError(CineMatchError):
    """Raised when a recommendation model is queried before fitting."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Recommendation model '{model_name}' has not been fitted.",
            status_code=503,
            error_code="MODEL_NOT_TRAINED",
        )


class MovieNotFoundError(CineMatchError):
    """Raised when a requested movie ID or title does not exist in the catalog."""

    def __init__(self, identifier: str | int) -> None:
        super().__init__(
            f"Movie '{identifier}' was not found in the catalog.",
            status_code=404,
            error_code="MOVIE_NOT_FOUND",
        )
        self.identifier = identifier


class InvalidParameterError(CineMatchError):
    """Raised when request query or path parameters fail validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, error_code="INVALID_PARAMETER")
