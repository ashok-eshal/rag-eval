"""Evaluation service for evaluating LLM responses"""

import re
import json
import streamlit as st
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from src.config.constants import DEFAULT_METRICS


def create_evaluation_prompt(question: str, response: str, ground_truth: str, 
                           retrieved_chunks: List[str], metrics: Dict[str, Dict[str, str]], 
                           chat_history: str = "") -> str:
    """Create detailed evaluation prompt for Claude"""
    
    chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    metrics_description = "\n\n".join([
        f"{i+1}. **{metric['name']}**: {metric['description']}"
        for i, metric in enumerate(metrics.values())
    ])
    
    chat_history_section = ""
    if chat_history and chat_history.strip():
        chat_history_section = f"\n## Chat History\n{chat_history}\n"
    
    prompt = f"""You are an expert evaluator tasked with assessing the quality of LLM responses in a RAG (Retrieval-Augmented Generation) system.

## Task Context
- **Question**: {question}
- **LLM Response**: {response}
- **Ground Truth**: {ground_truth}
{chat_history_section}
## Retrieved Context Chunks
{chunks_text}

## Evaluation Metrics
Please evaluate the response based on the following metrics:

{metrics_description}

## Evaluation Instructions
1. For each metric, provide:
   - A score from 0 to 10 (where 0 is completely failing and 10 is perfect)
   - A brief explanation for your score
   - Specific examples from the response that support your evaluation

2. Consider the following in your evaluation:
   - How well the response aligns with the ground truth
   - Whether the response uses information from the retrieved chunks appropriately
   - The overall quality and usefulness of the response
   - Any issues with factual accuracy, relevance, or appropriateness
   - For engagement metric, consider how well the response maintains conversation flow with the chat history

3. Be objective and specific in your evaluation. Point to concrete examples in the response.

## Output Format
Please structure your evaluation as follows:

```json
{{
    "overall_score": <weighted average of all scores>,
    "metrics": {{
        "answer_relevancy": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "task_completion": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "correctness": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "hallucination": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "contextual_relevancy": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "responsible_metrics": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "task_specific": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }},
        "engagement": {{
            "score": <0-10>,
            "explanation": "<detailed explanation>"
        }}
    }},
    "summary": "<brief overall assessment of the response quality>",
    "recommendations": "<suggestions for improvement>"
}}
```

Ensure your evaluation is thorough, fair, and provides actionable insights."""
    
    return prompt


def evaluate_response(question: str, response: str, ground_truth: str, 
                     retrieved_chunks: List[str], eval_api_key: str,
                     metrics: Dict[str, Dict[str, str]], chat_history: str = "",
                     eval_provider: str = "Claude", eval_model: str = "claude-sonnet-4-20250514", 
                     base_url: str = None) -> Dict[str, Any]:
    """Evaluate response using Claude, OpenAI, or Custom Models"""
    try:
        if eval_provider == "Claude":
            if not eval_api_key or not eval_api_key.startswith('sk-'):
                raise ValueError("Invalid Claude API key format. Key should start with 'sk-'")
            
            evaluator = ChatAnthropic(
                anthropic_api_key=eval_api_key,
                model_name=eval_model,
                temperature=0.1,
                max_retries=2,
                timeout=360.0,
                max_tokens=4096
            )
        elif eval_provider == "Custom Models":
            # Custom models from session state
            model_info = st.session_state.get('selected_evaluation_model', None)
            
            if not model_info:
                raise ValueError("No custom evaluation model selected")
            
            # Get temperature from model info, default to 0.1 for evaluation
            model_temperature = model_info.get('temperature', 0.1)
            model_base_url = base_url or model_info.get('base_url', 'https://api.deepinfra.com/v1/openai')
            
            # Use API key from selected model if available, otherwise use provided key
            custom_api_key = model_info.get('api_key', eval_api_key)
            
            if not custom_api_key:
                raise ValueError("API key is required for custom evaluation models")
            
            # Use OpenAI-compatible API for custom models
            models_fixed_temp = ['gpt-5', 'o1-preview', 'o1-mini', 'o1']
            is_fixed_temp_model = any(model in eval_model.lower() for model in models_fixed_temp)
            
            if is_fixed_temp_model:
                evaluator = ChatOpenAI(
                    model_name=eval_model,
                    openai_api_key=custom_api_key,
                    base_url=model_base_url,
                    temperature=1.0,
                    max_retries=2,
                    timeout=360.0,
                    max_completion_tokens=4096
                )
            else:
                evaluator = ChatOpenAI(
                    model_name=eval_model,
                    openai_api_key=custom_api_key,
                    base_url=model_base_url,
                    temperature=model_temperature,
                    max_retries=2,
                    timeout=360.0,
                    max_tokens=4096
                )
        else:  # OpenAI
            if not eval_api_key:
                raise ValueError("OpenAI API key is required")
            
            # Some newer OpenAI models (like GPT-5, O1 series) use max_completion_tokens instead of max_tokens
            # Models that require max_completion_tokens AND have fixed temperature (must use default temperature=1)
            models_fixed_temp_and_completion_tokens = ['gpt-5', 'o1-preview', 'o1-mini', 'o1']
            
            # Check if model is GPT-5 or O1 series
            is_gpt5_or_o1 = any(model in eval_model.lower() for model in models_fixed_temp_and_completion_tokens)
            
            if is_gpt5_or_o1:
                # GPT-5 and O1 models: must use default temperature (1.0) - set explicitly to avoid ChatOpenAI default of 0.7
                evaluator = ChatOpenAI(
                    model_name=eval_model,
                    openai_api_key=eval_api_key,
                    temperature=1.0,  # GPT-5 requires default temperature of 1.0
                    max_retries=2,
                    timeout=360.0,
                    max_completion_tokens=4096
                )
            else:
                # Other models: use temperature=0.1 for consistent evaluation, use max_tokens
                evaluator = ChatOpenAI(
                    model_name=eval_model,
                    openai_api_key=eval_api_key,
                    temperature=0.1,
                    max_retries=2,
                    timeout=360.0,
                    max_tokens=4096
                )
        
        prompt = create_evaluation_prompt(question, response, ground_truth, retrieved_chunks, metrics, chat_history)
        
        try:
            response_message = evaluator.invoke(prompt)
            evaluation = response_message.content
        except Exception as api_error:
            error_msg = str(api_error)
            if "connection" in error_msg.lower():
                provider_name = "Claude" if eval_provider == "Claude" else "OpenAI"
                raise ConnectionError(
                    f"Failed to connect to {provider_name} API. Please check:\n"
                    f"1. Your API key is valid\n"
                    f"2. Your network connection\n"
                    f"3. {provider_name} API service status"
                )
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                provider_name = "Claude" if eval_provider == "Claude" else "OpenAI"
                raise ValueError(f"Invalid {provider_name} API key. Please check your API key.")
            elif "rate" in error_msg.lower():
                raise ValueError("Rate limit exceeded. Please wait a moment and try again.")
            else:
                raise api_error
        
        try:
            # Try multiple patterns to extract JSON
            json_patterns = [
                r'```json\s*\n(.*?)\n```',  # With potential whitespace after json
                r'```json\n(.*?)```',       # Without newline before closing
                r'```\n(.*?)\n```',          # Generic code block
                r'\{.*\}',                   # Raw JSON object
            ]
            
            evaluation_dict = None
            
            for pattern in json_patterns:
                json_match = re.search(pattern, evaluation, re.DOTALL)
                if json_match:
                    try:
                        if pattern == r'\{.*\}':
                            # For raw JSON, use the whole match
                            json_str = json_match.group(0)
                        else:
                            # For code blocks, use the captured group
                            json_str = json_match.group(1)
                        
                        # Clean up the JSON string
                        json_str = json_str.strip()
                        
                        # Try to parse the JSON
                        evaluation_dict = json.loads(json_str)
                        break  # Successfully parsed, exit loop
                    except json.JSONDecodeError:
                        continue  # Try next pattern
            
            # If no pattern worked, try parsing the entire response as JSON
            if evaluation_dict is None:
                try:
                    evaluation_dict = json.loads(evaluation.strip())
                except json.JSONDecodeError:
                    # Last resort: create a minimal valid response
                    st.warning("Could not parse Claude's response. Creating default evaluation.")
                    evaluation_dict = {
                        "overall_score": 5,
                        "metrics": {},
                        "summary": "Failed to parse complete evaluation",
                        "recommendations": f"Raw response (first 500 chars): {evaluation[:500]}"
                    }
            
            # Ensure all expected metrics are present
            for metric_key in metrics.keys():
                if metric_key not in evaluation_dict.get('metrics', {}):
                    evaluation_dict['metrics'][metric_key] = {
                        "score": 5,  # Default neutral score
                        "explanation": "Metric not evaluated or parsing failed"
                    }
            
            # Ensure overall_score exists
            if 'overall_score' not in evaluation_dict:
                # Calculate average from available metric scores
                scores = [m.get('score', 5) for m in evaluation_dict.get('metrics', {}).values()]
                evaluation_dict['overall_score'] = sum(scores) / len(scores) if scores else 5
            
            # Ensure summary and recommendations exist
            if 'summary' not in evaluation_dict:
                evaluation_dict['summary'] = "Evaluation completed with parsing issues"
            if 'recommendations' not in evaluation_dict:
                evaluation_dict['recommendations'] = "Review the raw response for complete details"
            
            return evaluation_dict
            
        except Exception as parse_error:
            st.error(f"Unexpected error during parsing: {str(parse_error)}")
            st.error(f"Response length: {len(evaluation)} characters")
            
            # Return a valid structure even on complete failure
            return {
                "overall_score": 0,
                "metrics": {key: {"score": 0, "explanation": f"Parse error: {str(parse_error)}"} 
                          for key in metrics.keys()},
                "summary": f"Error parsing evaluation response: {str(parse_error)}",
                "recommendations": "Check the logs for the raw response"
            }
            
    except Exception as e:
        error_type = type(e).__name__
        if eval_provider == "Claude":
            provider_name = "Claude"
        elif eval_provider == "Custom Models":
            provider_name = "Custom Models"
        else:
            provider_name = "OpenAI"
        st.error(f"Error during {provider_name} evaluation ({error_type}): {str(e)}")
        
        if isinstance(e, ConnectionError):
            st.info(f"💡 Tip: Ensure your {provider_name} API key is valid and you have internet connectivity.")
        elif isinstance(e, ValueError) and "api key" in str(e).lower():
            if eval_provider == "Claude":
                st.info("💡 Tip: Get your Claude API key from https://console.anthropic.com/")
            elif eval_provider == "Custom Models":
                st.info("💡 Tip: Check your API key in the Configuration tab")
            else:
                st.info("💡 Tip: Get your OpenAI API key from https://platform.openai.com/api-keys")
        
        return {
            "overall_score": 0,
            "metrics": {key: {"score": 0, "explanation": f"Evaluation error: {str(e)}"} 
                      for key in metrics.keys()},
            "summary": f"Error during evaluation: {str(e)}",
            "recommendations": ""
        }
