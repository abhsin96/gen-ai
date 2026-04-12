"""Modern JSON Output Parser implementation following 2026 Python standards.

This module demonstrates best practices for using LangChain's JsonOutputParser
with proper type hints, error handling, structured output parsing, and LCEL chains.
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Person(BaseModel):
    """Schema for a fictional person with structured fields."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    city: str = Field(description="City where the person lives")


def create_llm_model() -> ChatHuggingFace:
    """Initialize and return the HuggingFace LLM model.

    Returns:
        ChatHuggingFace: Configured chat model instance.

    Raises:
        ValueError: If model initialization fails.
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7,
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        logger.error(f"Failed to initialize LLM model: {e}")
        raise ValueError(f"Model initialization error: {e}") from e


def create_json_parser_chain() -> Any:
    """Create a LangChain LCEL chain for JSON output parsing.

    Returns:
        Runnable chain that generates and parses structured JSON output.
    """
    # Initialize structured output parser with Pydantic schema
    parser = JsonOutputParser(pydantic_object=Person)

    # Create chat prompt template with system and user messages
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that generates structured data in JSON format.",
            ),
            (
                "user",
                "Generate information for a fictional person.\n\n{format_instructions}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # Initialize model
    model = create_llm_model()

    # Create LCEL chain: prompt -> model -> parser
    chain = prompt | model | parser

    return chain


def main() -> None:
    """Execute the JSON output parser demonstration."""
    try:
        logger.info("Initializing JSON output parser chain...")
        chain = create_json_parser_chain()

        logger.info("Invoking chain to generate fictional person data...")
        result: dict[str, Any] = chain.invoke({})

        logger.info("Successfully generated structured output:")
        print(f"\nGenerated Person Data:")
        print(f"  Name: {result.get('name')}")
        print(f"  Age: {result.get('age')}")
        print(f"  City: {result.get('city')}")
        print(f"\nFull JSON: {result}")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
