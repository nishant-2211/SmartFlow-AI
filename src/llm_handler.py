import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are SmartFlow AI, an expert travel data analyst specializing in airline passenger satisfaction data.

You have deep expertise in:
- Airline passenger behavior and satisfaction metrics
- Statistical analysis of flight data (delays, distances, class comparisons)
- Service quality assessment (wifi, food, entertainment, seat comfort, etc.)
- Customer segmentation (loyal vs. first-time, business vs. personal travel)
- Actionable insights for airlines to improve passenger experience

Your communication style:
- Respond with precise, data-driven insights using the provided context
- Use clear structure: lead with the key finding, then support with numbers
- When discussing ratings, always note they are on a 1–5 scale
- When you detect the user is asking for a chart, visualization, or graph, begin your response with "SHOW_CHART" on its own line, then continue your analysis
- Be conversational yet professional — you are a trusted analyst, not a robot
- If context is insufficient to answer fully, say so honestly and offer what you can infer

Trigger "SHOW_CHART" when the user asks about:
- Distributions, comparisons, breakdowns, trends
- Satisfaction rates, class analysis, delay patterns, age groups
- Any question that would be better answered visually

Always ground your responses in the retrieved passenger records provided."""


def get_claude_response(query: str, context: str, chat_history: list = []) -> str:
    """
    Send a query to Claude with RAG context and conversation history.

    Args:
        query: The user's current question.
        context: Retrieved passenger records from the vector store.
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        Claude's response as a plain string.
    """
    messages = []

    for turn in chat_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })

    # Inject retrieved context into the current user message
    user_message = (
        f"Here are relevant passenger records retrieved for your analysis:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"User question: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    except anthropic.APIConnectionError:
        return "Connection error: unable to reach the Anthropic API. Please check your internet connection."
    except anthropic.AuthenticationError:
        return "Authentication error: your ANTHROPIC_API_KEY is invalid or missing."
    except anthropic.RateLimitError:
        return "Rate limit reached. Please wait a moment and try again."
    except anthropic.APIStatusError as e:
        return f"API error ({e.status_code}): {e.message}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
