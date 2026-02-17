import streamlit as st
import os
import torch
from PIL import Image
import io
from adaptive_enhancer import AdaptiveEnhancer  # Changed
from quality_checker import QualityChecker
from ai_analyzer import AIAnalyzer

# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

st.set_page_config(page_title="AI Photo Enhancer", layout="wide")
st.title("🎨 AI Photo Quality Enhancer")

# Show device info
device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.caption(f"⚡ Running on: {device}")

st.markdown("*Powered by Real-ESRGAN + Multi-AI Analysis*")

# Sidebar
st.sidebar.header("⚙️ AI Settings")
auto_mode = st.sidebar.checkbox("🤖 AI Auto-Enhancement", value=True, 
                                 help="Let AI decide best enhancements")
manual_denoise = st.sidebar.checkbox("Manual Denoise", value=False, disabled=auto_mode)
manual_sharpen = st.sidebar.checkbox("Manual Sharpen", value=False, disabled=auto_mode)
manual_brighten = st.sidebar.checkbox("Manual Brighten", value=False, disabled=auto_mode)

# Multiple input methods
st.subheader("📤 Upload or Paste Image")
tab1, tab2 = st.tabs(["📁 Upload File", "📋 Paste from Clipboard"])

input_path = None
uploaded_file = None

with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'webp'], key="file_upload")
    if uploaded_file:
        input_path = f"uploads/{uploaded_file.name}"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

with tab2:
    st.markdown("**Paste image from clipboard:**")
    pasted_image = st.file_uploader("Ctrl+V to paste", type=['jpg', 'jpeg', 'png', 'webp'], key="paste_upload")
    if pasted_image:
        input_path = f"uploads/pasted_{pasted_image.name}"
        with open(input_path, "wb") as f:
            f.write(pasted_image.getbuffer())
        uploaded_file = pasted_image

if input_path:
    # Display original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(input_path, use_container_width=True)
        
        # Quality assessment
        with st.spinner("🔍 Analyzing image quality..."):
            quality = QualityChecker.overall_score(input_path)
        
        st.metric("Quality Score", f"{quality['overall']}/100")
        st.write(f"**Blur:** {quality['blur']}")
        st.write(f"**Resolution:** {quality['resolution']}")
        st.write(f"**Brightness:** {quality['brightness']}")
        
        # AI Analysis
        st.markdown("---")
        st.subheader("🤖 AI Deep Analysis")
        
        with st.spinner("🧠 AI is analyzing your image..."):
            analyzer = AIAnalyzer()
            ai_result = analyzer.analyze_content(input_path)
        
        # Display AI insights
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("AI Confidence", f"{ai_result['overall_confidence']}%")
            st.write(f"**Scene Type:** {ai_result['scene_type']}")
        with col_b:
            st.metric("Predicted Improvement", f"+{ai_result['predicted_improvement']}%")
            st.write(f"**Mode:** {ai_result['enhancement_mode']}")
        
        if ai_result['face_count'] > 0:
            st.success(f"✅ {ai_result['face_count']} face(s) detected")
        
        if ai_result['has_text']:
            st.info(f"📄 Text detected ({ai_result['text_length']} chars)")
        
        # AI Recommendations
        if ai_result['recommendations']:
            st.markdown("**🎯 AI Recommendations:**")
            for rec in ai_result['recommendations']:
                st.write(f"• {rec}")
        
        # Issues detected
        if ai_result['issues']:
            st.warning("**⚠️ Issues Detected:**")
            for issue in ai_result['issues']:
                st.write(f"• {issue}")
    
    with col2:
        st.subheader("✨ Enhanced Image")
        
if st.button("🚀 AI Enhance Now", type="primary", use_container_width=True):
    output_path = f"outputs/enhanced_{os.path.basename(input_path)}"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🤖 Initializing AI models...")
        progress_bar.progress(10)
        
        enhancer = AdaptiveEnhancer()  # Changed
        progress_bar.progress(30)
        
        status_text.text("🧠 AI is selecting best algorithm...")
        progress_bar.progress(50)
        
        # AI Auto Enhancement with algorithm selection
        result, algorithm, img_class, steps = enhancer.auto_enhance(input_path, output_path, ai_result)
        progress_bar.progress(90)
        
        status_text.text("✅ Enhancement complete!")
        progress_bar.progress(100)
        
        st.success(f"🎉 Enhancement Complete!")
        
        # Show AI's decisions
        st.info(f"**Image Type:** {img_class['type']} ({img_class['confidence']}% confidence)")
        st.info(f"**Algorithm Used:** {algorithm}")
        
        st.image(output_path, use_container_width=True)
        
        # New quality score
        new_quality = QualityChecker.overall_score(output_path)
        
        # Show improvements
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("New Quality", 
                     f"{new_quality['overall']}/100",
                     delta=f"+{round(new_quality['overall'] - quality['overall'], 2)}")
        with col_y:
            st.metric("Resolution", new_quality['resolution'])
        with col_z:
            st.metric("Sharpness", new_quality['blur'])
        
        # Download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Enhanced Image",
                data=file,
                file_name=f"ai_enhanced_{os.path.basename(input_path)}",
                mime="image/png",
                use_container_width=True
            )
        
        # Show AI processing steps
        with st.expander("🔍 View AI Processing Pipeline"):
            st.write(f"**Image Classification:** {img_class['type']}")
            st.write(f"**Algorithm Selected:** {algorithm}")
            st.write("**Processing Steps:**")
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()
# Footer
st.markdown("---")
st.markdown("""
**🤖 AI Technologies Used:**
- Real-ESRGAN (Deep Learning Super-Resolution)
- Haar Cascade (Face Detection)
- OCR (Text Detection)  
- CLAHE (Adaptive Brightness)
- Scene Classification AI
- Smart Enhancement Prediction
- Intelligent Scale Detection
""")