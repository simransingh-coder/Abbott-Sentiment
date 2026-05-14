"""
STREAMLIT SENTIMENT ANALYSIS APP
=================================
Multi-language sentiment analysis tool with Claude API integration.

Features:
- Upload Excel/CSV files with comments in any language
- Automatic language detection
- Sentiment analysis (Positive/Negative/Neutral)
- Topic extraction
- Visual dashboards
- Downloadable results

Deploy:
    streamlit run sentiment_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E2761;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #1E2761 0%, #667eea 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 12px rgba(30, 39, 97, 0.4);
    }
</style>
""", unsafe_allow_html=True)


class MultilingualSentimentAnalyzer:
    """Sentiment analyzer with multilingual support"""
    
    def __init__(self):
        self.supported_languages = [
            "English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Hindi", "Chinese", "Japanese", "Korean",
            "Arabic", "Russian", "Dutch", "Polish", "Turkish",
            "Indonesian", "Thai", "Vietnamese", "Greek", "Hebrew"
        ]
    
    def detect_languages(self, comments: pd.DataFrame, text_column: str) -> dict:
        """Detect languages present in the comments"""
        # In production, this would use actual language detection
        # For now, we'll use Claude to detect
        
        sample_comments = comments[text_column].head(20).tolist()
        
        prompt = f"""Analyze these comments and identify which languages are present.

COMMENTS:
{json.dumps(sample_comments, indent=2)}

Respond with JSON:
{{
  "languages_detected": ["English", "Spanish", "Hindi"],
  "primary_language": "English",
  "is_multilingual": true
}}
"""
        
        # Mock response for demo
        return {
            "languages_detected": ["English"],
            "primary_language": "English",
            "is_multilingual": False
        }
    
    def create_sentiment_prompt(self, comments: list, metadata: dict = None) -> str:
        """Create Claude prompt for sentiment analysis with multilingual support"""
        
        brand = metadata.get('brand', 'Brand') if metadata else 'Brand'
        post_type = metadata.get('post_type', 'Posts') if metadata else 'Posts'
        
        prompt = f"""Analyze the sentiment of these comments. The comments may be in MULTIPLE LANGUAGES.

IMPORTANT: 
- Understand and analyze comments in their original language
- DO NOT translate - analyze sentiment directly in the source language
- Sentiment categories: Positive, Negative, Neutral (universal across languages)
- Topics should be in English for consistency in reporting

CONTEXT:
- Brand: {brand}
- Post Type: {post_type}
- Total Comments: {len(comments)}

COMMENTS (may contain multiple languages):
{json.dumps(comments[:300], indent=2, ensure_ascii=False)}

ANALYSIS REQUIRED:

1. For EACH comment:
   - Detect the language (if not obvious, default to "Unknown")
   - Determine sentiment: Positive, Negative, or Neutral
   - Identify the main topic in English (e.g., "Product Quality", "Price", "Taste", "Service")
   - Note: Analyze sentiment in original language, but report topic in English

2. Summary Statistics:
   - Total comments analyzed
   - Sentiment breakdown (counts and percentages)
   - Top topics per sentiment category
   - Languages detected
   - Key insights

SENTIMENT GUIDELINES (Universal across languages):
- Positive: Praise, satisfaction, recommendations, positive emotions (😊❤️👍🎉)
- Negative: Complaints, disappointment, criticism, negative emotions (😡😢👎)
- Neutral: Questions, neutral statements, factual information

TOPIC CATEGORIES (report in English):
- Product Quality
- Price/Value
- Taste/Flavor
- Convenience
- Ingredients/Nutrition
- Availability/Stock
- Customer Service
- Packaging
- Shipping/Delivery
- Brand Trust
- Comparison to competitors
- General Feedback

Respond with ONLY valid JSON (no markdown):
{{
  "comments": [
    {{
      "index": 0, 
      "language": "English",
      "sentiment": "Positive", 
      "topic": "Product Quality",
      "original_text_preview": "First 50 chars of comment..."
    }},
    {{
      "index": 1,
      "language": "Spanish", 
      "sentiment": "Negative", 
      "topic": "Price"
    }}
  ],
  "summary": {{
    "total_analyzed": {len(comments)},
    "languages_detected": ["English", "Spanish", "Hindi"],
    "positive_count": 0,
    "negative_count": 0,
    "neutral_count": 0,
    "positive_percentage": 0.0,
    "negative_percentage": 0.0,
    "neutral_percentage": 0.0,
    "positive_topics": [
      {{"topic": "Product Quality", "count": 0}},
      {{"topic": "Taste", "count": 0}}
    ],
    "negative_topics": [
      {{"topic": "Price", "count": 0}},
      {{"topic": "Availability", "count": 0}}
    ],
    "neutral_topics": [
      {{"topic": "General Feedback", "count": 0}}
    ],
    "language_distribution": {{
      "English": 150,
      "Spanish": 80,
      "Hindi": 70
    }},
    "key_insights": [
      "Insight 1 about multilingual feedback",
      "Insight 2 about sentiment trends",
      "Insight 3 about topic patterns"
    ]
  }}
}}

CRITICAL: Analyze ALL {len(comments)} comments, handling each language naturally."""
        
        return prompt
    
    async def analyze_with_claude(self, prompt: str) -> dict:
        """Call Claude API for sentiment analysis"""
        
        try:
            response = await fetch("https://api.anthropic.com/v1/messages", {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 8192,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            })
            
            data = await response.json()
            content = data['content'][0]['text']
            
            # Remove markdown code blocks if present
            content = content.replace('```json', '').replace('```', '').strip()
            
            return json.loads(content)
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None


def load_data(uploaded_file) -> pd.DataFrame:
    """Load data from uploaded Excel or CSV file"""
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            # Try different encodings for multilingual support
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        else:
            st.error("Unsupported file format. Please upload Excel or CSV.")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def display_sentiment_dashboard(results: dict):
    """Display interactive sentiment analysis dashboard"""
    
    summary = results.get('summary', {})
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary.get('total_analyzed', 0)}</h3>
            <p>Total Comments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);">
            <h3>{summary.get('positive_count', 0)} ({summary.get('positive_percentage', 0):.1f}%)</h3>
            <p>Positive</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f44336 0%, #e57373 100%);">
            <h3>{summary.get('negative_count', 0)} ({summary.get('negative_percentage', 0):.1f}%)</h3>
            <p>Negative</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #9e9e9e 0%, #bdbdbd 100%);">
            <h3>{summary.get('neutral_count', 0)} ({summary.get('neutral_percentage', 0):.1f}%)</h3>
            <p>Neutral</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution Pie Chart
        st.subheader("📊 Sentiment Distribution")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [
                summary.get('positive_count', 0),
                summary.get('negative_count', 0),
                summary.get('neutral_count', 0)
            ]
        })
        
        fig = px.pie(
            sentiment_data, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#4caf50',
                'Negative': '#f44336',
                'Neutral': '#9e9e9e'
            },
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Language Distribution
        st.subheader("🌍 Language Distribution")
        
        lang_dist = summary.get('language_distribution', {})
        
        if lang_dist:
            lang_df = pd.DataFrame(list(lang_dist.items()), columns=['Language', 'Count'])
            lang_df = lang_df.sort_values('Count', ascending=False)
            
            fig = px.bar(
                lang_df,
                x='Language',
                y='Count',
                color='Count',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Language",
                yaxis_title="Number of Comments"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Language distribution not available")
    
    # Topics Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Top Topics by Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ Positive Topics")
        positive_topics = summary.get('positive_topics', [])
        for topic in positive_topics[:5]:
            st.markdown(f"- **{topic.get('topic')}**: {topic.get('count')} mentions")
    
    with col2:
        st.markdown("### ❌ Negative Topics")
        negative_topics = summary.get('negative_topics', [])
        for topic in negative_topics[:5]:
            st.markdown(f"- **{topic.get('topic')}**: {topic.get('count')} mentions")
    
    with col3:
        st.markdown("### ⚪ Neutral Topics")
        neutral_topics = summary.get('neutral_topics', [])
        for topic in neutral_topics[:5]:
            st.markdown(f"- **{topic.get('topic')}**: {topic.get('count')} mentions")
    
    # Key Insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 Key Insights")
    
    insights = summary.get('key_insights', [])
    for i, insight in enumerate(insights, 1):
        st.markdown(f"{i}. {insight}")


def create_detailed_dataframe(results: dict, original_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Create detailed DataFrame with sentiment results"""
    
    comments_analysis = results.get('comments', [])
    
    # Create mapping
    sentiment_map = {}
    topic_map = {}
    language_map = {}
    
    for item in comments_analysis:
        idx = item.get('index', 0)
        sentiment_map[idx] = item.get('sentiment', 'Unknown')
        topic_map[idx] = item.get('topic', 'Unknown')
        language_map[idx] = item.get('language', 'Unknown')
    
    # Add to dataframe
    result_df = original_df.copy()
    result_df['AI_Sentiment'] = result_df.index.map(lambda x: sentiment_map.get(x, 'Unknown'))
    result_df['AI_Topic'] = result_df.index.map(lambda x: topic_map.get(x, 'Unknown'))
    result_df['AI_Language'] = result_df.index.map(lambda x: language_map.get(x, 'Unknown'))
    
    return result_df


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">😊 Multilingual Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze comments in any language with AI-powered sentiment detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1E2761/FFFFFF?text=Sentiment+AI", use_container_width=True)
        
        st.markdown("## 📋 Supported Features")
        st.markdown("""
        ✅ Multiple file formats (Excel, CSV)  
        ✅ 20+ languages supported  
        ✅ Automatic language detection  
        ✅ Topic extraction  
        ✅ Visual dashboards  
        ✅ Downloadable results  
        """)
        
        st.markdown("---")
        
        st.markdown("## 🌍 Supported Languages")
        analyzer = MultilingualSentimentAnalyzer()
        
        with st.expander("View all languages"):
            for lang in analyzer.supported_languages:
                st.markdown(f"• {lang}")
        
        st.markdown("---")
        
        st.markdown("## 📊 How It Works")
        st.markdown("""
        1. **Upload** your comments file
        2. **Select** the text column
        3. **Analyze** with Claude AI
        4. **View** interactive dashboard
        5. **Download** results
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Step 1: Upload Your Comments File")
        
        uploaded_file = st.file_uploader(
            "Choose an Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload a file containing comments in any language"
        )
        
        if uploaded_file:
            # Load data
            df = load_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File loaded successfully! Found {len(df)} rows.")
                
                # Show preview
                with st.expander("📋 Preview Data"):
                    st.dataframe(df.head(10))
                
                st.markdown("### Step 2: Configure Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Select text column
                    text_column = st.selectbox(
                        "Select the column containing comments",
                        options=df.columns.tolist(),
                        help="Choose the column with the text you want to analyze"
                    )
                
                with col2:
                    # Optional: Select grouping columns
                    group_columns = st.multiselect(
                        "Select grouping columns (optional)",
                        options=[col for col in df.columns if col != text_column],
                        help="Group analysis by brand, post type, etc."
                    )
                
                # Additional metadata
                st.markdown("### Step 3: Add Context (Optional)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    brand_name = st.text_input("Brand Name", value="Brand")
                
                with col2:
                    post_type = st.text_input("Post Type", value="All Posts")
                
                # Analyze button
                st.markdown("### Step 4: Run Analysis")
                
                if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
                    
                    with st.spinner("🤖 Claude is analyzing your comments..."):
                        
                        # Create analyzer
                        analyzer = MultilingualSentimentAnalyzer()
                        
                        # Prepare comments
                        comments = df[text_column].fillna('').tolist()
                        
                        # Create prompt
                        metadata = {
                            'brand': brand_name,
                            'post_type': post_type
                        }
                        
                        prompt = analyzer.create_sentiment_prompt(comments, metadata)
                        
                        # Show prompt in expander
                        with st.expander("🔍 View Claude Prompt"):
                            st.code(prompt, language="text")
                        
                        st.info("""
                        **For this demo:** Copy the prompt above and paste it into Claude to get the sentiment analysis.
                        
                        In production, this would automatically call the Claude API.
                        """)
                        
                        # Instructions for manual use
                        st.markdown("""
                        ### 📝 Instructions:
                        
                        1. **Copy the prompt** from the expander above
                        2. **Open Claude** (claude.ai or app)
                        3. **Paste and send** the prompt
                        4. **Copy Claude's response** (the entire JSON)
                        5. **Paste the response** in the text area below
                        6. **Click "Process Results"**
                        """)
                        
                        # Text area for Claude's response
                        claude_response = st.text_area(
                            "Paste Claude's JSON response here:",
                            height=300,
                            placeholder='{"comments": [...], "summary": {...}}'
                        )
                        
                        if st.button("Process Results", use_container_width=True):
                            if claude_response:
                                try:
                                    # Parse response
                                    response = claude_response.replace('```json', '').replace('```', '').strip()
                                    results = json.loads(response)
                                    
                                    # Store in session state
                                    st.session_state['results'] = results
                                    st.session_state['original_df'] = df
                                    st.session_state['text_column'] = text_column
                                    
                                    st.success("✅ Results processed successfully! Go to the 'Results' tab to view.")
                                    
                                except Exception as e:
                                    st.error(f"Error parsing response: {str(e)}")
                            else:
                                st.warning("Please paste Claude's response first.")
    
    with tab2:
        st.markdown("## 📊 Analysis Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Display dashboard
            display_sentiment_dashboard(results)
            
            # Detailed results
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("📋 Detailed Results")
            
            if 'original_df' in st.session_state:
                detailed_df = create_detailed_dataframe(
                    results,
                    st.session_state['original_df'],
                    st.session_state['text_column']
                )
                
                # Filter options
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_filter = st.multiselect(
                        "Filter by Sentiment",
                        options=['Positive', 'Negative', 'Neutral'],
                        default=['Positive', 'Negative', 'Neutral']
                    )
                
                with col2:
                    if 'AI_Language' in detailed_df.columns:
                        language_filter = st.multiselect(
                            "Filter by Language",
                            options=detailed_df['AI_Language'].unique().tolist(),
                            default=detailed_df['AI_Language'].unique().tolist()
                        )
                
                # Apply filters
                filtered_df = detailed_df[detailed_df['AI_Sentiment'].isin(sentiment_filter)]
                
                if 'AI_Language' in detailed_df.columns:
                    filtered_df = filtered_df[filtered_df['AI_Language'].isin(language_filter)]
                
                st.dataframe(filtered_df, use_container_width=True, height=400)
                
                # Download buttons
                st.markdown("### 💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Excel download
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='Sentiment Analysis')
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # JSON download
                    json_str = json.dumps(results, indent=2)
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        else:
            st.info("👆 Upload and analyze a file first to see results here!")
    
    with tab3:
        st.markdown("## ℹ️ About This Tool")
        
        st.markdown("""
        ### 🎯 Purpose
        
        This tool helps you analyze sentiment in customer comments, reviews, and social media posts 
        across **multiple languages** without requiring translation.
        
        ### ✨ Key Features
        
        - **Multilingual Support**: Analyzes comments in 20+ languages
        - **No Translation Needed**: Direct sentiment analysis in source language
        - **Topic Extraction**: Identifies what people are talking about
        - **Visual Insights**: Interactive charts and dashboards
        - **Export Results**: Download analyzed data in Excel or JSON
        
        ### 🤖 How It Works
        
        1. **Upload**: You upload a file with comments in any language(s)
        2. **AI Analysis**: Claude analyzes each comment:
           - Detects the language
           - Determines sentiment (Positive/Negative/Neutral)
           - Identifies the topic being discussed
        3. **Visualization**: Results are displayed in an interactive dashboard
        4. **Export**: Download the complete analysis
        
        ### 🌍 Supported Languages
        
        English, Spanish, French, German, Italian, Portuguese, Hindi, Chinese, Japanese, Korean,
        Arabic, Russian, Dutch, Polish, Turkish, Indonesian, Thai, Vietnamese, Greek, Hebrew, 
        and many more!
        
        ### 📊 Use Cases
        
        - Social media comment analysis
        - Product review sentiment tracking
        - Customer feedback analysis
        - Brand health monitoring
        - Competitor analysis
        - Campaign performance measurement
        
        ### 🔒 Privacy & Security
        
        - Your data is processed securely
        - No data is stored permanently
        - Analysis happens in real-time
        - You control data access and downloads
        
        ### 💡 Tips for Best Results
        
        1. **Clean Data**: Remove spam or irrelevant comments first
        2. **Context Matters**: Provide brand/post type for better accuracy
        3. **Review Results**: Spot-check a sample to verify accuracy
        4. **Iterate**: Refine based on what you learn
        
        ### 📞 Support
        
        Need help? Have questions? Want to customize this tool?
        
        Contact your development team for:
        - Custom theme/topic categories
        - API integration
        - Batch processing
        - Scheduled automation
        """)
        
        st.markdown("---")
        st.markdown("**Version 1.0** | Built with ❤️ using Streamlit & Claude AI")


if __name__ == "__main__":
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    
    main()
