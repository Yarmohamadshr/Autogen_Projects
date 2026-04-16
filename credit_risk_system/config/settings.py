from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    OPENAI_API_KEY: str = "sk-placeholder"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60
    LLM_CACHE_SEED: int = 42

    # System
    HUMAN_IN_LOOP: bool = False
    MAX_ROUNDS: int = 8
    LOG_LEVEL: str = "INFO"

    # Paths
    DB_PATH: str = "credit_risk.db"
    MODEL_ARTIFACTS_PATH: str = "models/artifacts/"

    # Policy thresholds (overridable via env)
    DTI_MAX_PCT: float = 43.0
    FICO_MIN: int = 620
    LTV_MAX_HARD: float = 0.97
    PD_DENY_THRESHOLD: float = 0.35
    PD_REFER_THRESHOLD: float = 0.20

    @property
    def llm_config(self) -> dict:
        return {
            "config_list": [
                {
                    "model": self.LLM_MODEL,
                    "api_key": self.OPENAI_API_KEY,
                    "temperature": self.LLM_TEMPERATURE,
                    "max_tokens": self.LLM_MAX_TOKENS,
                    "timeout": self.LLM_TIMEOUT,
                }
            ],
            "cache_seed": self.LLM_CACHE_SEED,
        }


settings = Settings()
