"""
Utility functions for the Streamlit app.

This module provides helper functions for initializing the application components.
"""

from src.core.orchestrator import Orchestrator
from src.agents.query_agent import QueryAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.validation_agent import ValidationAgent
from src.data.data_store import DataStore
from src.llm.llm_provider import GeminiProvider, OpenAIProvider
from src.core.config import Config


def create_orchestrator() -> Orchestrator:
    """
    Create and initialize an Orchestrator with all required components.
    
    Returns:
        Initialized Orchestrator instance
    """
    # Initialize LLM provider based on configuration
    if Config.LLM_PROVIDER == "openai" and Config.OPENAI_API_KEY:
        llm_provider = OpenAIProvider(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            max_retries=Config.MAX_RETRIES
        )
    elif Config.LLM_PROVIDER == "gemini" and Config.GOOGLE_API_KEY:
        llm_provider = GeminiProvider(
            model=Config.GEMINI_MODEL,
            api_key=Config.GOOGLE_API_KEY,
            max_retries=Config.MAX_RETRIES
        )
    elif Config.GOOGLE_API_KEY:
        # Default to Gemini if available
        llm_provider = GeminiProvider(
            model=Config.GEMINI_MODEL,
            api_key=Config.GOOGLE_API_KEY,
            max_retries=Config.MAX_RETRIES
        )
    elif Config.OPENAI_API_KEY:
        # Fallback to OpenAI
        llm_provider = OpenAIProvider(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            max_retries=Config.MAX_RETRIES
        )
    else:
        raise ValueError(
            "No API key configured. Please set OPENAI_API_KEY or GOOGLE_API_KEY in .env file"
        )
    
    # Initialize data store
    data_store = DataStore()
    
    # Initialize agents
    extraction_agent = ExtractionAgent(data_store)
    validation_agent = ValidationAgent()
    query_agent = QueryAgent(llm_provider)
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        query_agent=query_agent,
        extraction_agent=extraction_agent,
        validation_agent=validation_agent,
        llm_provider=llm_provider,
        max_retries=Config.MAX_RETRIES
    )
    
    return orchestrator
