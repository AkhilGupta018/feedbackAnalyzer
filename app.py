import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Define the output schema using Pydantic
class FeedbackAnalysis(BaseModel):
    summary: str = Field(description="A brief summary of the customer feedback")
    sentiment: str = Field(
        description="Overall sentiment: positive, negative, neutral, or mixed"
    )
    themes: List[str] = Field(description="Main themes identified in the feedback")
    pain_points: List[str] = Field(
        description="Specific pain points or issues mentioned"
    )
    feature_requests: List[str] = Field(
        description="Any feature requests or suggestions"
    )
    priority_level: str = Field(description="Priority level: low, medium, or high")
    suggested_actions: List[str] = Field(
        description="Recommended actions to address the feedback"
    )


# Initialize the parser
parser = PydanticOutputParser(pydantic_object=FeedbackAnalysis)

# Create the prompt template
prompt_template = """You are an expert customer feedback analyzer. Analyze the following customer feedback and extract structured insights.

Customer Feedback:
{feedback}

Please analyze this feedback and provide a comprehensive analysis in the following JSON format:

{format_instructions}

Be thorough and accurate in your analysis. Consider both explicit and implicit meanings in the feedback.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


def analyze_feedback(feedback_text):
    """Analyze customer feedback using Gemini and LangChain"""
    try:

        # Create the chain
        chain = prompt | llm | parser

        # Run analysis
        result = chain.invoke({"feedback": feedback_text})

        return result
    except Exception as e:
        raise Exception(f"Error analyzing feedback: {str(e)}")


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Customer Feedback Analyzer", page_icon="📊", layout="wide"
    )

    # Header
    st.title("📊 Customer Feedback Analyzer")
    st.markdown("### Powered by Gemini AI & LangChain")
    st.markdown("---")

    # Main content area - Single column layout
    st.header("📝 Enter Customer Feedback")

    # Sample feedback examples
    sample_feedback = {
        "Positive": "I love this product! The checkout process is so smooth and fast. The UI is beautiful and intuitive. Keep up the great work!",
        "Negative": "The app keeps crashing during checkout. It's extremely frustrating and I've lost my cart twice now. The loading time is also unbearably slow.",
        "Mixed": "Great product overall, but the mobile app needs work. Desktop experience is fantastic, but the app is buggy and slow. Would love to see push notifications added.",
        "Feature Request": "It would be amazing if you could add a dark mode. Also, bulk editing would save me so much time. Love everything else though!",
    }

    # Dropdown for sample feedback
    sample_choice = st.selectbox(
        "Try a sample feedback (or write your own below):",
        ["Custom"] + list(sample_feedback.keys()),
    )

    # Text area for feedback
    if sample_choice == "Custom":
        feedback_text = st.text_area(
            "Customer Feedback",
            height=200,
            placeholder="Enter customer feedback here...",
            help="Paste or type the customer feedback you want to analyze",
        )
    else:
        feedback_text = st.text_area(
            "Customer Feedback",
            value=sample_feedback[sample_choice],
            height=200,
            help="You can edit this sample or use it as-is",
        )

    # Analyze button
    analyze_button = st.button(
        "🔍 Analyze Feedback", type="primary", use_container_width=True
    )

    # Analysis Results Section - Shows below the input
    if analyze_button:
        if not feedback_text.strip():
            st.error("⚠️ Please enter some feedback to analyze")
        else:
            st.markdown("---")
            st.header("📈 Analysis Results")

            with st.spinner("🤖 Analyzing feedback..."):
                try:
                    # Analyze the feedback
                    result = analyze_feedback(feedback_text)

                    # Display results in an organized way
                    st.success("✅ Analysis Complete!")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Summary Card
                    st.markdown("### 📋 Summary")
                    st.info(result.summary)
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Create two columns for Sentiment and Priority
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 💭 Sentiment")
                        sentiment_colors = {
                            "positive": ("🟢", "success"),
                            "negative": ("🔴", "error"),
                            "neutral": ("🟡", "warning"),
                            "mixed": ("🟠", "info"),
                        }
                        emoji, color = sentiment_colors.get(
                            result.sentiment.lower(), ("⚪", "info")
                        )

                        if color == "success":
                            st.success(f"{emoji} **{result.sentiment.title()}**")
                        elif color == "error":
                            st.error(f"{emoji} **{result.sentiment.title()}**")
                        elif color == "warning":
                            st.warning(f"{emoji} **{result.sentiment.title()}**")
                        else:
                            st.info(f"{emoji} **{result.sentiment.title()}**")

                    with col2:
                        st.markdown("### ⚡ Priority Level")
                        priority_colors = {
                            "high": ("🔴", "error"),
                            "medium": ("🟡", "warning"),
                            "low": ("🟢", "success"),
                        }
                        emoji, color = priority_colors.get(
                            result.priority_level.lower(), ("⚪", "info")
                        )

                        if color == "success":
                            st.success(f"{emoji} **{result.priority_level.title()}**")
                        elif color == "error":
                            st.error(f"{emoji} **{result.priority_level.title()}**")
                        elif color == "warning":
                            st.warning(f"{emoji} **{result.priority_level.title()}**")
                        else:
                            st.info(f"{emoji} **{result.priority_level.title()}**")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Create 3 columns
                    left, middle, right = st.columns(3, border=True)

                    # THEMES
                    if result.themes:
                        with left:
                            st.markdown("### 🏷️ Themes")
                            for theme in result.themes:
                                st.markdown(f"- {theme}")

                    # PAIN POINTS
                    if result.pain_points:
                        with middle:
                            st.markdown("### ⚠️ Pain Points")
                            for pain in result.pain_points:
                                st.markdown(f"- {pain}")

                    # FEATURE REQUESTS
                    if result.feature_requests:
                        with right:
                            st.markdown("### 💡 Feature Requests")
                            for feature in result.feature_requests:
                                st.markdown(f"- {feature}")

                    # Suggested Actions
                    if result.suggested_actions:
                        st.markdown("### ✅ Suggested Actions")
                        for i, action in enumerate(result.suggested_actions, 1):
                            st.markdown(f"{i}. {action}")
                        st.markdown("<br>", unsafe_allow_html=True)

                    # Raw JSON output (collapsible)
                    with st.expander("🔍 View Raw JSON Output"):
                        st.json(result.model_dump())

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info(
                        "💡 Tip: Make sure your API key is valid and you have internet connection"
                    )


if __name__ == "__main__":
    main()
