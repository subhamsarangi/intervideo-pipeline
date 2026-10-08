"""
Interview Control Agent

Decides interview flow:
- When to move to next question
- When to probe deeper with follow-ups
- When to skip or stop interview
- Detects fatigue and time pressure

Uses GPT-4 for lightweight decision-making on answer quality + session state.
"""

import os
import asyncio
import logging
from typing import Optional, List
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel

from src.models.interview_agent import (
    AnswerRecord,
    AnswerQuality,
    InterviewSessionState,
    AgentDecision,
    DecisionReason,
    InterviewMetrics,
)

logger = logging.getLogger(__name__)


# GPT-structured output models
class QualityAssessment(BaseModel):
    """LLM's assessment of answer quality"""
    quality: AnswerQuality
    reasoning: str  # Why this quality level
    confidence: float  # 0-1


class NextDecision(BaseModel):
    """LLM's decision on what to do next"""
    action: DecisionReason
    follow_up: Optional[str] = None  # If action is probe_deeper
    reasoning: str
    confidence: float


class InterviewControlAgent:
    """Controls interview flow based on answers and session state"""
    
    def __init__(self):
        """Initialize agent with OpenAI client"""
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = instructor.from_openai(self.client)
        
        logger.info("Interview Control Agent initialized")
    
    async def assess_answer_quality(
        self,
        question: str,
        question_level: str,
        answer: str,
    ) -> QualityAssessment:
        """
        Judge quality of candidate's answer.
        
        Args:
            question: The question asked
            question_level: Level (junior/mid/senior)
            answer: Candidate's response
        
        Returns:
            Quality assessment with reasoning
        """
        
        system_prompt = """You are an expert interviewer assessing candidate answers.
        
Rate answer quality on this scale:
- EXCELLENT: Directly answers all parts, detailed, relevant, no hesitation
- GOOD: Answers with minor gaps or slight misunderstanding  
- FAIR: Partially addresses question, needs probing for completeness
- POOR: Doesn't meaningfully address the question
- OFF_TOPIC: Talks around it or completely misunderstands

Consider the question level (junior/mid/senior) when assessing.
Be fair but strict."""
        
        user_prompt = f"""QUESTION (Level: {question_level}):
{question}

CANDIDATE ANSWER:
{answer}

Assess the quality of this answer."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=QualityAssessment,
                temperature=0.3,
            )
            
            logger.info(f"Answer quality assessed: {response.quality.value}")
            return response
            
        except Exception as e:
            logger.error(f"Error assessing answer quality: {e}")
            # Default to FAIR if error
            return QualityAssessment(
                quality=AnswerQuality.FAIR,
                reasoning="Assessment error - defaulting to FAIR",
                confidence=0.5,
            )
    
    async def decide_next_action(
        self,
        session_state: InterviewSessionState,
        last_answer_quality: AnswerQuality,
        last_answer_text: str,
        question_asked: str,
    ) -> AgentDecision:
        """
        Decide what to do next (continue, probe, skip, exit).
        
        Args:
            session_state: Current session state
            last_answer_quality: Quality of last answer
            last_answer_text: Text of last answer
            question_asked: Text of question just answered
        
        Returns:
            Decision on next action
        """
        
        # Calculate metrics
        elapsed_seconds = (datetime.now() - session_state.start_time).total_seconds()
        remaining_seconds = session_state.planned_duration_seconds - elapsed_seconds
        avg_quality = session_state.avg_answer_quality
        
        # Build context
        system_prompt = """You are an interview flow manager. Decide whether to:
- CONTINUE_NORMAL: Answer was good, move to next question
- PROBE_DEEPER: Answer was incomplete, ask a follow-up
- SKIP_QUESTION: Candidate struggled too much, move to next
- TIME_LIMIT: Running low on time, wrap up soon
- FATIGUE: Candidate appears tired or disengaged
- ALL_ASKED: All questions done
- CANDIDATE_EXIT: User wants to stop

Rules:
- If answer is EXCELLENT or GOOD: usually CONTINUE_NORMAL
- If answer is FAIR: PROBE_DEEPER with 1-2 follow-ups
- If answer is POOR or OFF_TOPIC: SKIP_QUESTION
- If remaining time < 2 minutes or <25% of questions left: TIME_LIMIT
- If last 3 answers average POOR/OFF_TOPIC: likely FATIGUE
- Prioritize candidate comfort and engagement"""
        
        user_prompt = f"""SESSION STATE:
- Questions asked: {session_state.questions_asked}/{session_state.total_questions}
- Time elapsed: {elapsed_seconds//60}m {int(elapsed_seconds%60)}s
- Time remaining: {remaining_seconds//60}m {int(remaining_seconds%60)}s
- Average answer quality: {avg_quality:.2f}
- Last answer quality: {last_answer_quality.value}

LAST Q&A:
Q: {question_asked[:200]}...
A: {last_answer_text[:300]}...

What should we do next?"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=NextDecision,
                temperature=0.3,
            )
            
            decision = AgentDecision(
                action=response.action,
                follow_up=response.follow_up,
                reasoning=response.reasoning,
                confidence=response.confidence,
                based_on_answer=last_answer_text[:100],
            )
            
            logger.info(f"Next action decided: {decision.action.value} (confidence: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"Error deciding next action: {e}")
            # Default: continue
            return AgentDecision(
                action=DecisionReason.CONTINUE_NORMAL,
                reasoning="Decision error - defaulting to continue",
                confidence=0.3,
            )
    
    async def record_answer(
        self,
        session_state: InterviewSessionState,
        question_id: int,
        question_text: str,
        question_level: str,
        question_type: str,
        answer_text: str,
        duration_seconds: int,
    ) -> AnswerRecord:
        """
        Record an answer and assess quality.
        
        Args:
            session_state: Session to update
            question_id: ID of question
            question_text: Full question text
            question_level: Level (junior/mid/senior)
            question_type: Type (technical/behavioral/design)
            answer_text: Candidate's answer
            duration_seconds: Time spent on this question
        
        Returns:
            Updated AnswerRecord with quality assessment
        """
        
        # Assess quality
        quality_assessment = await self.assess_answer_quality(
            question_text,
            question_level,
            answer_text,
        )
        
        # Create record
        record = AnswerRecord(
            question_id=question_id,
            question_text=question_text,
            question_level=question_level,
            question_type=question_type,
            asked_at=datetime.now(),
            answer_text=answer_text,
            answered_at=datetime.now(),
            duration_seconds=duration_seconds,
            quality=quality_assessment.quality,
            feedback=quality_assessment.reasoning,
        )
        
        # Update session
        session_state.answers.append(record)
        session_state.questions_asked += 1
        
        # Update average quality (convert enum to numeric)
        quality_scores = {
            AnswerQuality.EXCELLENT: 1.0,
            AnswerQuality.GOOD: 0.8,
            AnswerQuality.FAIR: 0.6,
            AnswerQuality.POOR: 0.3,
            AnswerQuality.OFF_TOPIC: 0.0,
        }
        scores = [quality_scores[a.quality] for a in session_state.answers]
        session_state.avg_answer_quality = sum(scores) / len(scores) if scores else 0.5
        
        logger.info(f"Answer recorded: {record.quality.value} (avg quality: {session_state.avg_answer_quality:.2f})")
        
        return record
    
    async def generate_follow_up(
        self,
        original_question: str,
        answer: str,
    ) -> str:
        """
        Generate a follow-up question based on answer gaps.
        
        Args:
            original_question: Original question asked
            answer: Candidate's answer
        
        Returns:
            Follow-up question
        """
        
        prompt = f"""Based on this answer, generate ONE short follow-up question to probe deeper:

ORIGINAL Q: {original_question}

ANSWER: {answer}

Generate a natural follow-up (1 sentence) that probes a gap or asks for clarification."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=100,
            )
            
            follow_up = response.choices[0].message.content
            logger.info(f"Follow-up generated: {follow_up[:80]}...")
            return follow_up
            
        except Exception as e:
            logger.error(f"Error generating follow-up: {e}")
            return "Can you tell me more about that?"
    
    def should_exit_interview(self, session_state: InterviewSessionState) -> bool:
        """Check if interview should end"""
        if not session_state.is_active:
            return True
        
        if session_state.exit_reason:
            return True
        
        return False
    
    async def generate_summary(
        self,
        session_state: InterviewSessionState,
    ) -> InterviewMetrics:
        """
        Generate summary metrics for interview.
        
        Args:
            session_state: Completed session
        
        Returns:
            Summary metrics
        """
        
        session_state.end_time = datetime.now()
        session_state.total_duration_seconds = int(
            (session_state.end_time - session_state.start_time).total_seconds()
        )
        
        # Quality distribution
        quality_dist = {}
        for q in AnswerQuality:
            count = len([a for a in session_state.answers if a.quality == q])
            quality_dist[q] = count
        
        # By level
        questions_by_level = {}
        performance_by_level = {}
        for level in ["junior", "mid", "senior"]:
            level_answers = [a for a in session_state.answers if a.question_level == level]
            questions_by_level[level] = len(level_answers)
            
            if level_answers:
                quality_scores = {
                    AnswerQuality.EXCELLENT: 1.0,
                    AnswerQuality.GOOD: 0.8,
                    AnswerQuality.FAIR: 0.6,
                    AnswerQuality.POOR: 0.3,
                    AnswerQuality.OFF_TOPIC: 0.0,
                }
                scores = [quality_scores[a.quality] for a in level_answers]
                performance_by_level[level] = sum(scores) / len(scores)
            else:
                performance_by_level[level] = 0.0
        
        # Detect fatigue
        last_3 = session_state.answers[-3:] if len(session_state.answers) >= 3 else session_state.answers
        poor_answers = len([a for a in last_3 if a.quality in [AnswerQuality.POOR, AnswerQuality.OFF_TOPIC]])
        fatigue = poor_answers >= 2
        
        return InterviewMetrics(
            total_questions=session_state.total_questions,
            questions_asked=session_state.questions_asked,
            questions_skipped=session_state.questions_skipped,
            quality_distribution=quality_dist,
            avg_answer_quality=session_state.avg_answer_quality,
            total_duration_minutes=session_state.total_duration_seconds / 60 if session_state.total_duration_seconds else 0,
            avg_question_duration_seconds=int(sum(a.duration_seconds or 0 for a in session_state.answers) / len(session_state.answers)) if session_state.answers else 0,
            fatigue_detected=fatigue,
            reason_for_exit=session_state.exit_reason or DecisionReason.ALL_ASKED,
            questions_by_level=questions_by_level,
            performance_by_level=performance_by_level,
        )
