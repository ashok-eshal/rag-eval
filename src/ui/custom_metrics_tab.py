"""Custom Metrics Tab UI"""

import streamlit as st


def render_custom_metrics_tab():
    """Render the Custom Metrics tab"""
    st.header("Custom Evaluation Metrics")
    st.markdown("Add custom metrics to evaluate specific aspects of your use case")
    
    if st.session_state.custom_metrics:
        st.subheader("Existing Custom Metrics")
        for i, metric in enumerate(st.session_state.custom_metrics):
            with st.expander(f"{metric['name']}"):
                st.write(f"**ID:** {metric['id']}")
                st.write(f"**Description:** {metric['description']}")
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.custom_metrics.pop(i)
                    st.rerun()
    
    st.subheader("Add New Metric")
    with st.form("add_custom_metric"):
        metric_name = st.text_input("Metric Name", placeholder="e.g., Technical Accuracy")
        metric_description = st.text_area(
            "Metric Description", 
            placeholder="Detailed description of what this metric evaluates and how it should be scored...",
            height=150
        )
        
        if st.form_submit_button("Add Metric"):
            if metric_name and metric_description:
                metric_id = metric_name.lower().replace(" ", "_").replace("-", "_")
                
                existing_ids = [m['id'] for m in st.session_state.custom_metrics]
                if metric_id in existing_ids:
                    st.error(f"A metric with similar name already exists")
                else:
                    st.session_state.custom_metrics.append({
                        "id": metric_id,
                        "name": metric_name,
                        "description": metric_description
                    })
                    
                    st.success(f"✅ Added custom metric: {metric_name}")
                    st.rerun()
            else:
                st.error("Please provide both metric name and description")
