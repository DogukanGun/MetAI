"""
AI Agent for Meeting Analysis

Combines emotion recognition results with knowledge base retrieval
to provide intelligent meeting analysis and insights.

Version: 2.0 - Fixed OpenAI import for Streamlit
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Force module reload marker
__version__ = "2.0"
_FORCE_RELOAD = True

# Configure logging first
logging.basicConfig(level=logging.INFO, force=True)

# OpenAI - Force import and provide fallback
OPENAI_AVAILABLE = False
OpenAI = None

# Try multiple import strategies
import sys
logging.info(f"Python executable: {sys.executable}")
logging.info(f"Python path: {sys.path[:3]}")

try:
    logging.info("Attempting to import OpenAI library...")
    from openai import OpenAI as OpenAIClient
    OpenAI = OpenAIClient
    OPENAI_AVAILABLE = True
    logging.info("✓ OpenAI library imported successfully")
    logging.info(f"✓ OpenAI class: {OpenAI}")
except ImportError as e:
    OPENAI_AVAILABLE = False
    logging.error(f"✗ OpenAI ImportError: {e}")
    logging.error(f"   Python executable: {sys.executable}")
    logging.error("   Install with: pip install openai")
    
    # Try to get more details
    try:
        import importlib.util
        spec = importlib.util.find_spec("openai")
        if spec:
            logging.error(f"   OpenAI module found at: {spec.origin}")
        else:
            logging.error("   OpenAI module not found in path")
    except Exception as e2:
        logging.error(f"   Could not check module: {e2}")
        
except Exception as e:
    OPENAI_AVAILABLE = False
    logging.error(f"✗ Unexpected error importing OpenAI: {e}")
    import traceback
    logging.error(traceback.format_exc())

# Log the import status immediately
logging.info(f"AI Agent module loaded - OPENAI_AVAILABLE: {OPENAI_AVAILABLE}, OpenAI: {OpenAI}")


logger = logging.getLogger(__name__)


class MeetingAnalysisAgent:
    """
    AI Agent for comprehensive meeting analysis.
    
    Combines:
    - Emotion recognition results
    - Knowledge base retrieval
    - LLM reasoning
    """
    
    def __init__(
        self,
        query_engine=None,
        model: str = None,
        temperature: float = None,
        provider: str = "cloud"
    ):
        """
        Initialize the AI agent.
        
        Args:
            query_engine: QueryEngine instance for knowledge base access
            model: Model name (default from env based on provider)
            temperature: LLM temperature (default from env)
            provider: "cloud" for OpenAI or "local" for local LLM (default: "cloud")
        """
        self.query_engine = query_engine
        self.provider = provider.lower()
        
        # Get configuration from environment based on provider
        if self.provider == "local":
            self.api_key = os.getenv("LOCAL_LLM_API_KEY", "ollama")
            self.model = model or os.getenv("LOCAL_LLM_MODEL", "llama3")
            self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        else:  # cloud/openai
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.temperature = temperature if temperature is not None else float(os.getenv("AGENT_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("AGENT_MAX_TOKENS", "2000"))
        self.kb_top_k = int(os.getenv("AGENT_KB_TOP_K", "5"))
        self.kb_similarity_threshold = float(os.getenv("AGENT_KB_SIMILARITY_THRESHOLD", "0.5"))
        
        # Initialize OpenAI client with appropriate settings
        self.client = None
        
        # Debug logging
        logger.info(f"OpenAI available: {OPENAI_AVAILABLE}")
        logger.info(f"OpenAI object: {OpenAI}")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"API key set: {bool(self.api_key)}")
        logger.info(f"API key value: {self.api_key}")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Model: {self.model}")
        
        # Try to create client - force creation for local provider even if OPENAI_AVAILABLE is False
        if OpenAI and self.api_key:
            try:
                logger.info(f"Creating OpenAI client with base_url={self.base_url}")
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info(f"✓ AI Agent client initialized successfully with {self.provider} provider using {self.model}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize OpenAI client: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.client = None
        elif not OpenAI:
            logger.error(f"✗ OpenAI class is None - import failed. OPENAI_AVAILABLE={OPENAI_AVAILABLE}")
            logger.error("Try: pip install --upgrade openai")
        elif not self.api_key:
            logger.error(f"✗ API key not set for provider '{self.provider}'")
        else:
            logger.warning(f"✗ Agent will operate in limited mode. OPENAI_AVAILABLE={OPENAI_AVAILABLE}, OpenAI={OpenAI}, api_key={bool(self.api_key)}")
    
    def analyze_meeting(
        self,
        emotion_results: Dict[str, Any],
        video_metadata: Dict[str, Any],
        transcription: Optional[str] = None,
        context_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive meeting analysis.
        
        Args:
            emotion_results: Results from emotion recognition
            video_metadata: Video metadata (duration, filename, etc.)
            transcription: Video transcription text
            context_query: Optional query to retrieve relevant knowledge base context
            
        Returns:
            Analysis results with insights and recommendations
        """
        logger.info("Starting meeting analysis with AI agent")
        
        # Extract key information
        analysis = {
            "summary": "",
            "key_insights": [],
            "emotional_dynamics": {},
            "recommendations": [],
            "knowledge_base_context": [],
            "detailed_analysis": "",
            "agent_available": OPENAI_AVAILABLE and bool(self.api_key)
        }
        
        # Get knowledge base context if available
        dominant_emotion = emotion_results.get('overall_prediction', {}).get('predicted_emotion', None)
        if self.query_engine and context_query:
            kb_context = self._retrieve_knowledge_context(
                query=context_query,
                transcription=transcription,
                emotion=dominant_emotion
            )
            analysis["knowledge_base_context"] = kb_context
        else:
            kb_context = []
        
        # Generate analysis
        if OPENAI_AVAILABLE and self.api_key:
            try:
                analysis_result = self._generate_llm_analysis(
                    emotion_results, video_metadata, transcription, kb_context
                )
                analysis.update(analysis_result)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                analysis["error"] = str(e)
                # Fallback to rule-based analysis
                analysis.update(self._fallback_analysis(emotion_results, video_metadata))
        else:
            # Use rule-based analysis
            analysis.update(self._fallback_analysis(emotion_results, video_metadata))
        
        logger.info("Meeting analysis completed")
        return analysis
    
    def _retrieve_knowledge_context(
        self,
        query: str,
        transcription: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from knowledge base with emotion-specific queries.
        
        Args:
            query: Base query string
            transcription: Video transcription to augment query
            emotion: Detected emotion to tailor queries
            
        Returns:
            List of relevant context chunks with citations
        """
        if not self.query_engine:
            return []
        
        try:
            all_context = []
            queries_made = []
            
            # Build multiple targeted queries based on emotion
            queries = [query]
            
            # Add emotion-specific queries
            if emotion:
                emotion_queries = [
                    f"How to handle {emotion} emotions in meetings",
                    f"{emotion} emotion workplace best practices",
                    f"dealing with {emotion} feelings team communication"
                ]
                queries.extend(emotion_queries[:2])  # Add top 2 emotion queries
            
            # Add transcription-based query
            if transcription and len(transcription) > 50:
                # Extract key phrases (simple approach - first 100 words)
                words = transcription.lower().split()[:100]
                trans_query = f"meeting about {' '.join(words[:20])}"
                queries.append(trans_query)
            
            # Execute each query
            for q in queries[:3]:  # Limit to 3 queries to avoid too many calls
                try:
                    text_chunks, _ = self.query_engine.query(
                        query_text=q,
                        top_k=3,  # Get fewer per query
                        include_images=False,
                        similarity_threshold=self.kb_similarity_threshold
                    )
                    
                    if text_chunks:
                        queries_made.append(q)
                        for chunk in text_chunks:
                            # Create citation string
                            citation = f"Document {chunk.document_id[:8]}"
                            if chunk.page_number:
                                citation += f", Page {chunk.page_number}"
                            
                            context_item = {
                                "content": chunk.metadata.get('content', ''),
                                "similarity_score": chunk.similarity_score,
                                "document_id": chunk.document_id,
                                "page_number": chunk.page_number,
                                "citation": citation,
                                "query_used": q
                            }
                            
                            # Avoid duplicates
                            if not any(c['document_id'] == chunk.document_id and 
                                     c['content'] == context_item['content'] 
                                     for c in all_context):
                                all_context.append(context_item)
                
                except Exception as e:
                    logger.warning(f"Query '{q}' failed: {e}")
                    continue
            
            # Sort by relevance
            all_context.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Limit to top results
            all_context = all_context[:self.kb_top_k]
            
            logger.info(f"Retrieved {len(all_context)} unique knowledge base chunks from {len(queries_made)} queries")
            return all_context
        
        except Exception as e:
            logger.error(f"Knowledge base retrieval failed: {e}")
            return []
    
    def _generate_llm_analysis(
        self,
        emotion_results: Dict[str, Any],
        video_metadata: Dict[str, Any],
        transcription: Optional[str],
        kb_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate analysis using LLM.
        
        Args:
            emotion_results: Emotion recognition results
            video_metadata: Video metadata
            transcription: Video transcription
            kb_context: Knowledge base context
            
        Returns:
            Analysis dictionary
        """
        # Build prompt
        prompt = self._build_analysis_prompt(
            emotion_results, video_metadata, transcription, kb_context
        )
        
        # Call OpenAI using new client format
        if not self.client:
            raise Exception("OpenAI client not initialized")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert meeting analyst specializing in emotional intelligence and workplace dynamics. 
You analyze video meetings by combining emotion recognition data, transcriptions, and relevant background knowledge 
to provide actionable insights. Your analysis should be professional, empathetic, and focused on improving team dynamics."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        analysis_text = response.choices[0].message.content
        
        # Parse the structured response
        parsed_result = self._parse_llm_response(analysis_text)
        
        # Add raw LLM output for transparency
        parsed_result['raw_llm_response'] = analysis_text
        parsed_result['llm_model'] = self.model
        parsed_result['llm_prompt'] = prompt
        
        return parsed_result
    
    def _build_analysis_prompt(
        self,
        emotion_results: Dict[str, Any],
        video_metadata: Dict[str, Any],
        transcription: Optional[str],
        kb_context: List[Dict[str, Any]]
    ) -> str:
        """Build the analysis prompt for the LLM."""
        
        prompt_parts = [
            "# Meeting Video Analysis Request\n",
            f"## Video Information",
            f"- Filename: {video_metadata.get('filename', 'Unknown')}",
            f"- Duration: {video_metadata.get('duration', 0):.1f} seconds",
            f"- Date: {video_metadata.get('upload_date', 'Unknown')}\n"
        ]
        
        # Add emotion analysis (robust with .get() methods)
        if 'overall_prediction' in emotion_results and emotion_results['overall_prediction']:
            pred = emotion_results['overall_prediction']
            emotion = pred.get('predicted_emotion', 'neutral')
            confidence = pred.get('confidence', 0.0)
            
            prompt_parts.extend([
                "## Emotional Analysis",
                f"- Dominant Emotion: {emotion.title()}",
                f"- Confidence: {confidence:.1%}",
            ])
            
            # Add emotion distribution if available
            all_confs = pred.get('all_confidences', pred.get('probabilities', {}))
            if all_confs:
                prompt_parts.append("- Emotion Distribution:")
                for emotion_name, conf in all_confs.items():
                    prompt_parts.append(f"  - {emotion_name.title()}: {conf:.1%}")
            prompt_parts.append("")
        
        # Add temporal emotion data (robust)
        if 'temporal_predictions' in emotion_results and emotion_results['temporal_predictions']:
            prompt_parts.extend([
                "## Emotional Timeline",
                "Key moments during the meeting:"
            ])
            for i, temp in enumerate(emotion_results['temporal_predictions'][:10], 1):
                timestamp = temp.get('timestamp', 0)
                emotion = temp.get('emotion', temp.get('predicted_emotion', 'neutral'))
                confidence = temp.get('confidence', 0.0)
                prompt_parts.append(
                    f"{i}. At {timestamp:.0f}s: {emotion.title()} ({confidence:.0%})"
                )
            prompt_parts.append("")
        
        # Add mental health analysis (robust)
        if 'mental_health_analysis' in emotion_results and emotion_results['mental_health_analysis']:
            mh = emotion_results['mental_health_analysis']
            prompt_parts.extend([
                "## Participant Well-being Indicators",
                f"- Overall Score: {mh.get('mental_health_score', 0):.1f}/100",
                f"- Status: {mh.get('status', 'Unknown')}",
                f"- Dominant Emotion: {mh.get('dominant_emotion', 'neutral').title()}",
                f"- Positive Emotions: {mh.get('positive_percentage', 0):.1f}%",
                f"- Negative Emotions: {mh.get('negative_percentage', 0):.1f}%\n"
            ])
        
        # Add transcription
        if transcription:
            prompt_parts.extend([
                "## Meeting Transcription",
                f"{transcription[:1000]}..." if len(transcription) > 1000 else transcription,
                ""
            ])
        
        # Add knowledge base context with citations
        if kb_context:
            prompt_parts.extend([
                "## Relevant Background Knowledge",
                "The following information from the knowledge base may be relevant.",
                "**IMPORTANT**: When referencing this information in your analysis, cite the source using [Source: Document X, Page Y] format."
            ])
            for i, ctx in enumerate(kb_context, 1):
                citation = ctx.get('citation', f"Document {i}")
                prompt_parts.append(
                    f"\n**Source {i}** [{citation}]:\n{ctx['content'][:400]}..."
                )
                prompt_parts.append(f"Query used: '{ctx.get('query_used', 'N/A')}'")
                prompt_parts.append("")
        
        # Add instructions
        prompt_parts.extend([
            "\n## Analysis Request",
            "Based on the above data, provide a comprehensive meeting analysis with the following structure:",
            "",
            "**CRITICAL INSTRUCTIONS**:",
            "1. Actively use the knowledge base context in your analysis",
            "2. When making statements based on knowledge base, cite the source: [Source: Document X, Page Y]",
            "3. Connect emotional findings to knowledge base recommendations",
            "4. Reference relevant best practices, guidelines, or research from the knowledge base",
            "5. Make your analysis evidence-based by grounding it in the provided context",
            "",
            "### Executive Summary",
            "A brief 2-3 sentence overview of the meeting dynamics.",
            "If relevant, reference knowledge base findings here.",
            "",
            "### Key Insights",
            "List 3-5 important observations about:",
            "- Emotional patterns and their implications (cite relevant knowledge if available)",
            "- Team dynamics and engagement",
            "- Communication effectiveness",
            "- Any concerning patterns",
            "**Use knowledge base citations when applicable**",
            "",
            "### Emotional Dynamics",
            "Analyze:",
            "- Overall emotional tone of the meeting",
            "- Significant emotional shifts and what might have caused them",
            "- Participant well-being indicators",
            "**Compare findings to best practices from knowledge base if available**",
            "",
            "### Recommendations",
            "Provide 3-5 actionable recommendations for:",
            "- Improving meeting effectiveness",
            "- Addressing emotional concerns (reference knowledge base strategies)",
            "- Enhancing team collaboration",
            "- Follow-up actions",
            "**Ground recommendations in knowledge base evidence when possible**",
            "",
            "### Detailed Analysis",
            "A deeper dive into specific moments or patterns that stood out.",
            "**Include knowledge base references and citations to support your analysis.**",
            "",
            "Format your response clearly with these exact section headers."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured response from LLM."""
        
        analysis = {
            "summary": "",
            "key_insights": [],
            "emotional_dynamics": {},
            "recommendations": [],
            "detailed_analysis": ""
        }
        
        try:
            # Simple parsing by sections
            sections = {
                "Executive Summary": "summary",
                "Key Insights": "key_insights",
                "Emotional Dynamics": "emotional_dynamics",
                "Recommendations": "recommendations",
                "Detailed Analysis": "detailed_analysis"
            }
            
            current_section = None
            current_content = []
            
            for line in response_text.split('\n'):
                line = line.strip()
                
                # Check if it's a section header
                for header, key in sections.items():
                    if header.lower() in line.lower() and (line.startswith('#') or line.startswith('**')):
                        # Save previous section
                        if current_section:
                            self._save_section_content(analysis, current_section, current_content)
                        # Start new section
                        current_section = key
                        current_content = []
                        break
                else:
                    # Add to current section
                    if current_section and line:
                        current_content.append(line)
            
            # Save last section
            if current_section:
                self._save_section_content(analysis, current_section, current_content)
        
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: use entire response as detailed analysis
            analysis["detailed_analysis"] = response_text
        
        return analysis
    
    def _save_section_content(self, analysis: Dict, section_key: str, content: List[str]):
        """Save parsed section content."""
        text = '\n'.join(content).strip()
        
        if section_key in ["key_insights", "recommendations"]:
            # Extract bullet points
            items = []
            for line in content:
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    items.append(line.lstrip('-•* ').strip())
                elif line and not line.startswith('#'):
                    items.append(line)
            analysis[section_key] = [item for item in items if item]
        elif section_key == "emotional_dynamics":
            analysis[section_key] = {"analysis": text}
        else:
            analysis[section_key] = text
    
    def _fallback_analysis(
        self,
        emotion_results: Dict[str, Any],
        video_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when LLM is not available.
        
        Args:
            emotion_results: Emotion recognition results
            video_metadata: Video metadata
            
        Returns:
            Basic analysis dictionary
        """
        logger.info("Using rule-based fallback analysis")
        
        analysis = {
            "summary": "Analysis completed using rule-based system (OpenAI API not configured).",
            "key_insights": [],
            "emotional_dynamics": {},
            "recommendations": [],
            "detailed_analysis": ""
        }
        
        # Extract dominant emotion (robust)
        if 'overall_prediction' in emotion_results and emotion_results['overall_prediction']:
            pred = emotion_results['overall_prediction']
            emotion = pred.get('predicted_emotion', 'neutral')
            confidence = pred.get('confidence', 0.0)
            
            analysis["summary"] = (
                f"The meeting shows predominantly {emotion} emotion with {confidence:.0%} confidence. "
                f"Duration: {video_metadata.get('duration', 0):.0f} seconds."
            )
            
            # Generate insights based on emotion
            if emotion == 'happy':
                analysis["key_insights"].append("Positive and engaged atmosphere detected")
                analysis["recommendations"].append("Maintain this positive energy in future meetings")
            elif emotion == 'neutral':
                analysis["key_insights"].append("Professional and focused meeting tone")
                analysis["recommendations"].append("Consider adding energizing activities to boost engagement")
            elif emotion in ['sad', 'angry', 'fear']:
                analysis["key_insights"].append(f"Detected {emotion} emotion - may indicate concerns")
                analysis["recommendations"].append("Follow up with participants to address any issues")
        
        # Mental health insights (robust)
        if 'mental_health_analysis' in emotion_results and emotion_results['mental_health_analysis']:
            mh = emotion_results['mental_health_analysis']
            score = mh.get('mental_health_score', 50)
            
            if score >= 70:
                analysis["key_insights"].append("Positive well-being indicators")
            elif score < 50:
                analysis["key_insights"].append("Lower well-being scores detected - may need attention")
                analysis["recommendations"].append("Consider wellness check-ins with team members")
            
            analysis["emotional_dynamics"]["well_being_score"] = f"{score:.0f}/100"
            analysis["emotional_dynamics"]["status"] = mh.get('status', 'Unknown')
        
        # Temporal analysis
        if 'temporal_predictions' in emotion_results:
            temporal = emotion_results['temporal_predictions']
            if len(temporal) > 0:
                analysis["key_insights"].append(f"Analyzed {len(temporal)} time points throughout the meeting")
        
        return analysis
