import streamlit as st

from src.config.session import initialize_session_state
from src.ui.select_models_tab import render_select_models_tab
from src.ui.config_tab import render_config_tab
from src.ui.rag_tab import render_rag_tab
from src.ui.eval_data_tab import render_eval_data_tab
from src.ui.custom_metrics_tab import render_custom_metrics_tab
from src.ui.run_eval_tab import render_run_eval_tab
from src.ui.results_tab import render_results_tab
from src.ui.rag_gen_tab import render_rag_gen_tab

st.set_page_config(
    page_title="LLM Response Evaluation System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# Streamlit UI
st.title("🤖 LLM Response Evaluation System")
st.markdown("### Evaluate LLM responses using RAG pipeline with multiple providers")

# Main content area with tabs
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔌 Select Models",
    "⚙️ Configuration", 
    "📚 RAG Documents", 
    "📋 Evaluation Data",
    "➕ Custom Metrics",
    "🔍 Run Evaluation", 
    "📊 Results",
    "🤖 RAG Generation"
])

with tab0:
    render_select_models_tab()

with tab1:
    render_config_tab()

with tab2:
    render_rag_tab()

with tab3:
    render_eval_data_tab()

with tab4:
    render_custom_metrics_tab()

with tab5:
    render_run_eval_tab()

with tab6:
    render_results_tab()

with tab7:
    render_rag_gen_tab()

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
st.markdown("""
- **Select Models**: Add custom embedding and inference models in the Select Models tab (e.g., DeepInfra models)
- **Configuration**: Set all API keys and Milvus settings in the Configuration tab first
- **Embeddings**: Choose between OpenAI (API-based), open-source models (local/free), or custom models
  - OpenAI models require API key but offer high quality
  - HuggingFace/SentenceTransformers run locally for free
  - Ollama requires Ollama server running locally
  - Custom models from DeepInfra or other providers can be added with ease
- **Collections**: Each embedding model needs its own collection due to dimension differences
- **Sample Template**: Download the sample Excel template to ensure correct column formatting
- **Chat History**: Include conversation context in 'Chat history' column for better engagement evaluation
- **Testing**: Start with a small number of questions to test the setup
- **Custom Metrics**: Add domain-specific evaluation criteria as needed
- **Performance**: For faster processing with open-source models, consider using GPU
""")
