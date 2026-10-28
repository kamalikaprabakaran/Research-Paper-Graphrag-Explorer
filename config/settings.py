from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All variables are optional; missing credentials will not raise errors unless accessed.
    """

    OPENROUTER_API_KEY: Optional[str] = Field(default=None)
    OPENROUTER_MODEL: str = Field(default="")
    NEO4J_URI: Optional[str] = Field(default=None)
    NEO4J_USERNAME: Optional[str] = Field(default=None)
    NEO4J_PASSWORD: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def __repr__(self) -> str:
        hidden = {
            "OPENROUTER_API_KEY": "***" if self.OPENROUTER_API_KEY else None,
            "NEO4J_PASSWORD": "***" if self.NEO4J_PASSWORD else None,
        }
        return f"Settings({{**self.model_dump(), **hidden}})"
