"""
Q&A Mode for conversational data analysis.

This module implements the Q&A mode that enables users to have natural
language conversations about their data with context-aware responses.
"""

import logging
from typing import Optional
from datetime import datetime

from src.core.models import Message, Response
from src.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class QAMode:
    """
    Q&A Mode for conversational data analysis.
    
    This class enables users to:
    - Ask natural language questions about their data
    - Receive data-grounded responses with numerical evidence
    - Ask follow-up questions using conversation context
    - Request clarifications with detailed explanations
    - Maintain conversation history
    """
    
    def __init__(self, orchestrator: Orchestrator):
        """
        Initialize Q&A Mode.
        
        Args:
            orchestrator: Orchestrator instance for query processing
        """
        self.orchestrator = orchestrator
        logger.info("QAMode initialized")
    
    def answer_question(self, question: str) -> Response:
        """
        Answer a natural language question about the data.
        
        This method:
        - Accepts natural language questions from users
        - Uses the Orchestrator to process queries through the agent pipeline
        - Returns data-grounded responses with specific data points
        - Maintains conversation context for follow-up questions
        - Supports pronoun resolution and context continuity
        
        Args:
            question: Natural language question from the user
            
        Returns:
            Response object containing the answer and metadata
            
        Raises:
            ValueError: If question is empty or invalid
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = question.strip()
        logger.info(f"Processing Q&A question: {question}")
        
        # Check if this is a clarification request
        if self._is_clarification_request(question):
            logger.info("Detected clarification request")
            return self._handle_clarification(question)
        
        # Process the question through the orchestrator
        # The orchestrator handles:
        # - Conversation context for follow-up questions
        # - Pronoun resolution
        # - Multi-operation chaining
        # - Data-grounded response generation
        try:
            response = self.orchestrator.process_query(
                user_query=question,
                mode="qa"
            )
            
            logger.info(
                f"Question answered successfully in "
                f"{response.metadata.get('execution_time', 0):.2f}s"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return Response(
                answer=f"I encountered an error while processing your question: {str(e)}",
                metadata={
                    "error": str(e),
                    "question": question
                }
            )
    
    def _is_clarification_request(self, question: str) -> bool:
        """
        Detect if the question is a request for clarification.
        
        Args:
            question: User's question
            
        Returns:
            True if this is a clarification request
        """
        clarification_keywords = [
            "explain", "clarify", "what do you mean", "can you elaborate",
            "tell me more", "more details", "how did you", "why did you",
            "what does that mean", "break down", "in detail"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in clarification_keywords)
    
    def _handle_clarification(self, question: str) -> Response:
        """
        Handle clarification requests with detailed explanations.
        
        This method provides more detailed explanations by:
        - Retrieving the last response from conversation history
        - Using the LLM to generate a more detailed explanation
        - Including additional data evidence
        
        Args:
            question: Clarification request
            
        Returns:
            Response with detailed explanation
        """
        logger.info("Handling clarification request")
        
        # Get conversation history
        history = self.orchestrator.get_conversation_history()
        
        if len(history) < 2:
            # No previous conversation to clarify
            return Response(
                answer="I don't have any previous responses to clarify. Please ask a question about your data first.",
                metadata={"clarification_request": True}
            )
        
        # Get the last assistant response
        last_assistant_message = None
        for msg in reversed(history):
            if msg.role == "assistant":
                last_assistant_message = msg
                break
        
        if not last_assistant_message:
            return Response(
                answer="I don't have a previous response to clarify.",
                metadata={"clarification_request": True}
            )
        
        # Create a clarification prompt
        clarification_prompt = f"""The user asked for clarification about your previous response.

Previous response: {last_assistant_message.content}

User's clarification request: {question}

Please provide a more detailed explanation with additional context and data evidence."""
        
        try:
            # Use the orchestrator's LLM provider for clarification
            llm_response = self.orchestrator.llm_provider.generate_with_cache(
                prompt=clarification_prompt,
                temperature=0.7
            )
            
            # Add clarification to conversation context
            clarification_message = Message(
                role="assistant",
                content=llm_response.content,
                timestamp=datetime.now(),
                metadata={
                    "clarification": True,
                    "tokens_used": llm_response.tokens_used
                }
            )
            self.orchestrator.context_manager.add_message(clarification_message)
            
            return Response(
                answer=llm_response.content,
                metadata={
                    "clarification": True,
                    "tokens_used": llm_response.tokens_used
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating clarification: {str(e)}")
            return Response(
                answer=f"I encountered an error while generating a clarification: {str(e)}",
                metadata={
                    "error": str(e),
                    "clarification_request": True
                }
            )
    
    def get_conversation_history(self) -> list:
        """
        Get the conversation history.
        
        Returns:
            List of Message objects representing the conversation
        """
        return self.orchestrator.get_conversation_history()
    
    def reset_conversation(self) -> None:
        """
        Reset the conversation history.
        
        This clears all conversation context and starts fresh.
        """
        self.orchestrator.reset_context()
        logger.info("Conversation history reset")
