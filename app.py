import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from langdetect import detect, DetectorFactory
import sqlite3
from datetime import datetime, timedelta

# --- Database configuration ---
DB_FILE = "sentiment.db"

def initialize_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            sentiment TEXT,
            count INTEGER
        )
    ''')
    conn.commit()
    conn.close()


initialize_db()

# Ensure consistent language detection results

# Ensure consistent language detection results
DetectorFactory.seed = 0

#database


# --- Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Sentiment & NLP Dashboard")

# --- Initialize Session State ---
if 'analysis_results_df' not in st.session_state:
    st.session_state.analysis_results_df = pd.DataFrame()
if 'sentiment_distribution_data' not in st.session_state:
    st.session_state.sentiment_distribution_data = None
if 'processed_texts_for_pdf' not in st.session_state:
    st.session_state.processed_texts_for_pdf = []
if 'last_analyzed_text_full_results' not in st.session_state:
    st.session_state.last_analyzed_text_full_results = {} # To store all results for single *selected* text
if 'detailed_analysis_target_text' not in st.session_state:
    st.session_state.detailed_analysis_target_text = ""

# --- Model Loading (Cached) ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis", top_k=None)

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_ner_model():
    return pipeline("ner", grouped_entities=True)

sentiment_analyzer = load_sentiment_model()
emotion_analyzer = load_emotion_model()
summarizer = load_summarization_model()
ner_recognizer = load_ner_model()

# --- Helper Functions ---

def analyze_sentiment(text):
    if not text.strip():
        return {'label': 'N/A', 'score': 0.0}
    result = sentiment_analyzer(text)[0]
    return result

def analyze_emotion(text):
    if not text.strip():
        return []
    results = emotion_analyzer(text)
    return results[0] if results and isinstance(results[0], list) else []


def summarize_text(text, max_length, min_length):
    if not text.strip():
        return "No text to summarize."
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization error: {e}. Try adjusting length or using a shorter text.")
        return "Error during summarization."

def recognize_entities(text):
    if not text.strip():
        return []
    entities = ner_recognizer(text)
    return entities

def get_label_name(label):
    if label == 'LABEL_0':
        return 'Negative'
    elif label == 'LABEL_1':
        return 'Neutral'
    elif label == 'LABEL_2':
        return 'Positive'
    else:
        # For emotion models, the label might already be the emotion name
        return label.replace('_', ' ').title() if isinstance(label, str) else 'N/A'

def get_sentiment_color(label):
    if label == 'Positive':
        return 'green'
    elif label == 'Negative':
        return 'red'
    elif label == 'Neutral':
        return 'orange'
    else:
        return 'gray' # For N/A or emotion labels


def explain_sentiment(text, sentiment_label):
    st.subheader("Explanation of Sentiment")
    st.write(f"The text was classified as **{sentiment_label}**.")
    st.write("Below are some words that might have contributed to this sentiment:")

    words = text.split()
    positive_keywords = ["great", "excellent", "love", "happy", "good", "amazing", "fantastic", "wonderful", "best", "super", "perfect", "enjoy", "like", "favorite"]
    negative_keywords = ["bad", "terrible", "hate", "awful", "poor", "worst", "disappointing", "sad", "frustrating", "issue", "problem", "ugly", "don't like", "dislike"]
    neutral_keywords = ["and", "the", "a", "is", "it", "this", "that", "there", "what", "can", "but", "also", "if", "then", "when", "how", "where"] # Expanded neutral keywords

    display_text = []
    for word in words:
        cleaned_word = word.lower().strip(".,!?;:\"'").replace("’", "'")
        # Handle multi-word negative phrases for better explanation
        if sentiment_label == 'Negative' and ("don't like" in cleaned_word or "dislike" in cleaned_word):
             display_text.append(f"<span style='background-color: lightcoral; padding: 2px; border-radius: 3px;'>{word}</span>")
        elif sentiment_label == 'Positive' and cleaned_word in positive_keywords:
            display_text.append(f"<span style='background-color: lightgreen; padding: 2px; border-radius: 3px;'>{word}</span>")
        elif sentiment_label == 'Negative' and cleaned_word in negative_keywords:
            display_text.append(f"<span style='background-color: lightcoral; padding: 2px; border-radius: 3px;'>{word}</span>")
        elif sentiment_label == 'Neutral' and cleaned_word in neutral_keywords:
             display_text.append(f"<span style='background-color: lightyellow; padding: 2px; border-radius: 3px;'>{word}</span>")
        else:
            display_text.append(word)
    st.markdown(" ".join(display_text), unsafe_allow_html=True)
    st.caption("Note: This is a simplified explanation. Advanced models use more sophisticated techniques for keyword influence.")


# --- PDF Generator Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NLP Analysis Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        # Ensure proper encoding for special characters if any
        self.multi_cell(0, 5, body.encode('latin-1', 'replace').decode('latin-1'))
        self.ln()

    def add_plot(self, plot_path, width=150):
        try:
            self.image(plot_path, x=self.get_x(), y=self.get_y(), w=width)
            self.ln(width * (9/16) + 10) # Adjust for aspect ratio and add some space
        except RuntimeError as e:
            st.error(f"Error adding plot to PDF: {e}. Ensure the image file exists and is valid.")
            self.ln(20) # Add some space even if image fails


# --- Streamlit UI ---

def landing_page():
    st.title("Welcome to the Advanced NLP Dashboard! 📊")
    st.write("Understand the emotional tone, key entities, and concise summaries of your text data with powerful AI.")

    st.markdown("---")
    st.subheader("Your AI-Powered Insights Await!")

    st.write("#### Sentiment Trends (Stored Analysis)")

    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=7))
    end_date = st.date_input("End Date", datetime.today())

    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        conn = sqlite3.connect(DB_FILE)
        query = """
            SELECT date, sentiment, SUM(count) as total_count
            FROM sentiment_log
            WHERE date BETWEEN ? AND ?
            GROUP BY date, sentiment
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        conn.close()

        if df.empty:
            st.info("No sentiment data available for the selected range.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                pie_data = df.groupby('sentiment')['total_count'].sum().reset_index()
                fig_pie = px.pie(pie_data, values='total_count', names='sentiment',
                                 title='Sentiment Distribution',
                                 color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'orange'})
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.write("##### Sentiment Over Time")

                # Ensure 'date' is parsed as datetime
                df['date'] = pd.to_datetime(df['date'])

                # Pivot by date and sentiment
                pivot_df = df.pivot(index='date', columns='sentiment', values='total_count').fillna(0)

                # Ensure all expected sentiment columns exist
                for sentiment in ['Positive', 'Negative', 'Neutral']:
                    if sentiment not in pivot_df.columns:
                        pivot_df[sentiment] = 0

                # Reset index and reshape for Plotly
                pivot_df = pivot_df.reset_index()
                melted_df = pivot_df.melt(id_vars='date', var_name='Sentiment', value_name='Count')

                # Debug output: Show the melted_df
                st.write("Debug: Data for Sentiment Over Time plot:")
                st.dataframe(melted_df)

                if melted_df.empty:
                    st.warning("No data available to plot sentiment trend over time. Please check your database or date range.")
                else:
                    # Plot
                    fig_line = px.line(
                        melted_df,
                        x='date',
                        y='Count',
                        color='Sentiment',
                        title='Sentiment Trend Over Time',
                        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("""
    This dashboard provides a comprehensive view of sentiment patterns and other NLP insights in your text data.
    You can analyze customer reviews, social media posts, survey responses, and more.
    **Get started by using the sidebar to input text or upload files for real-time NLP analysis!**
    """)
    st.markdown("---")

#database
def analysis_page():

    st.title("NLP Analysis Module 💡")
    st.write("Input text directly or upload files for sentiment, emotion, summarization, and entity recognition.")

    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ("Single Text/Multiple Lines Analysis", "Batch File Analysis (CSV/TXT)"),
        horizontal=True
    )

    # Clear previous analysis results if mode changes or new analysis starts
    if st.button("Clear Previous Analysis"):
        st.session_state.analysis_results_df = pd.DataFrame()
        st.session_state.sentiment_distribution_data = None
        st.session_state.processed_texts_for_pdf = []
        st.session_state.last_analyzed_text_full_results = {}
        st.session_state.detailed_analysis_target_text = ""
        st.rerun() # Rerun to clear the display

    st.markdown("---")

    if analysis_mode == "Single Text/Multiple Lines Analysis":
        # Add the custom delimiter input
        custom_delimiter = st.text_input(
            "Optional: Enter a custom delimiter to split texts (e.g., '###', '---END---'). Leave empty to split by newlines.",
            value=""
        )

        user_input_area = st.text_area(
            "Enter text(s) for analysis:", height=200,
            placeholder=f"Type your text(s) here.\n\nUse newlines to separate texts, or use your custom delimiter '{custom_delimiter}' if specified.\n\nExample: This product is amazing!\nAnother example: I am very disappointed with the service."
        )

        # Split input into individual texts based on delimiter
        if custom_delimiter:
            input_texts_raw = user_input_area.split(custom_delimiter)
        else:
            input_texts_raw = user_input_area.split('\n')

        input_texts = [text.strip() for text in input_texts_raw if text.strip()] # Filter out empty lines

        if st.button("Perform All Analyses"):
            if input_texts:
                st.subheader("Analysis Results:")

                with st.spinner("Performing analyses..."):
                    all_sentiment_results = []
                    detailed_analysis_options = []

                    for i, text in enumerate(input_texts):
                        sentiment_result = analyze_sentiment(text)
                        sentiment_label = get_label_name(sentiment_result['label'])
                        sentiment_confidence = sentiment_result['score'] * 100
                        all_sentiment_results.append({
                            'Original Text': text,
                            'Sentiment': sentiment_label,
                            'Confidence': sentiment_confidence
                        })
                        detailed_analysis_options.append(f"Text {i + 1}: {text[:100]}...")  # Truncate for dropdown

                    # Update session state with all sentiment results AFTER the loop
                    st.session_state.analysis_results_df = pd.DataFrame(all_sentiment_results)

                    # Now you can safely do sentiment counts on the updated DataFrame
                    sentiment_counts = st.session_state.analysis_results_df['Sentiment'].value_counts()

                    # Save sentiment counts to DB
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    today_str = datetime.today().strftime('%Y-%m-%d')

                    for sentiment, count in sentiment_counts.items():
                        c.execute(
                            "INSERT INTO sentiment_log (date, sentiment, count) VALUES (?, ?, ?)",
                            (today_str, sentiment, int(count))
                        )
                    conn.commit()
                    conn.close()

                    st.success("Analysis complete and results saved to database!")

                    sentiment_distribution_data = st.session_state.analysis_results_df[
                        'Sentiment'].value_counts().reset_index()
                    sentiment_distribution_data.columns = ['Sentiment', 'Count']
                    sentiment_distribution_data['Color'] = sentiment_distribution_data['Sentiment'].apply(
                        get_sentiment_color)
                    st.session_state.sentiment_distribution_data = sentiment_distribution_data

                    st.write("#### Overall Sentiment Analysis for All Inputs:")
                    st.dataframe(st.session_state.analysis_results_df)

                    st.markdown("---")

                    # Allow selection for detailed analysis
                    if len(input_texts) > 0: # Ensure there's at least one text
                        if len(input_texts) > 1:
                            st.subheader("Detailed NLP Analysis (Select a Text):")
                            selected_index = st.selectbox(
                                "Choose a text for detailed emotion, NER, and summarization:",
                                options=range(len(input_texts)),
                                format_func=lambda x: detailed_analysis_options[x]
                            )
                            st.session_state.detailed_analysis_target_text = input_texts[selected_index]
                            st.info(f"Detailed analysis performed on: \"{st.session_state.detailed_analysis_target_text[:200]}...\"") # Truncate for display
                        else:
                            st.session_state.detailed_analysis_target_text = input_texts[0]
                            st.info(f"Detailed analysis performed on the input text: \"{st.session_state.detailed_analysis_target_text[:200]}...\"") # Truncate for display

                        # --- Perform and display detailed NLP for the selected text ---
                        target_text = st.session_state.detailed_analysis_target_text

                        # Get sentiment for the selected text (it's already calculated, just for consistency)
                        # Find the sentiment result for the target_text by matching the exact text
                        selected_sentiment_result = next((item for item in all_sentiment_results if item['Original Text'] == target_text), None)
                        selected_sentiment_label = selected_sentiment_result['Sentiment'] if selected_sentiment_result else 'N/A'
                        selected_sentiment_confidence = selected_sentiment_result['Confidence'] if selected_sentiment_result else 0.0

                        st.markdown(f"**Sentiment of selected text:** <span style='color:{get_sentiment_color(selected_sentiment_label)}; font-size:20px;'>**{selected_sentiment_label}**</span>", unsafe_allow_html=True)
                        st.write(f"**Confidence:** `{selected_sentiment_confidence:.2f}%`")
                        st.write("---")
                        explain_sentiment(target_text, selected_sentiment_label)
                        st.write("---")

                        # 2. Emotion Detection
                        st.subheader("Emotion Detection:")
                        emotion_results = analyze_emotion(target_text)
                        if emotion_results:
                            emotion_df = pd.DataFrame(emotion_results)
                            emotion_df['score'] = emotion_df['score'] * 100 # Convert to percentage
                            # Sort by score and display top 5 (or all if less than 5)
                            emotion_df = emotion_df.sort_values(by='score', ascending=False).head(5).reset_index(drop=True)
                            st.write("Top Emotions Detected:")
                            for idx, row in emotion_df.iterrows():
                                st.write(f"- **{get_label_name(row['label'])}:** `{row['score']:.2f}%`")

                            # Plot emotion distribution (Pie chart)
                            fig_emotion = px.pie(emotion_df, values='score', names='label',
                                                 title='Emotion Distribution for Selected Text',
                                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                            st.plotly_chart(fig_emotion, use_container_width=True)
                        else:
                            st.info("No distinct emotions detected or text is too short for emotion analysis.")
                        st.write("---")

                        # 3. Named Entity Recognition
                        st.subheader("Named Entity Recognition (NER):")
                        entities = recognize_entities(target_text)
                        if entities:
                            ner_data = []
                            for entity in entities:
                                ner_data.append({
                                    "Entity": entity['word'],
                                    "Type": entity['entity_group'],
                                    "Score": f"{entity['score']:.2f}%"
                                })
                            st.dataframe(pd.DataFrame(ner_data))
                        else:
                            st.info("No named entities detected.")
                        st.write("---")

                        # 4. Text Summarization
                        st.subheader("Text Summarization:")
                        summary_min_length = st.slider("Minimum summary length (words)", 10, 50, 30, key="summary_min_length_single")
                        summary_max_length = st.slider("Maximum summary length (words)", 50, 200, 100, key="summary_max_length_single")
                        if summary_min_length >= summary_max_length:
                            st.warning("Minimum length must be less than maximum length.")
                        else:
                            summary_text = summarize_text(target_text, summary_max_length, summary_min_length)
                            st.info(summary_text)
                        st.write("---")

                        # Store full results for the *selected* text for PDF
                        st.session_state.last_analyzed_text_full_results = {
                            'text': target_text,
                            'sentiment': {'label': selected_sentiment_label, 'confidence': selected_sentiment_confidence},
                            'emotions': emotion_results,
                            'entities': entities,
                            'summary': summary_text
                        }
                    else:
                        st.warning("No valid text lines found for detailed analysis after splitting. Please enter text.")

            else:
                st.warning("Please enter some text to analyze.")

    elif analysis_mode == "Batch File Analysis (CSV/TXT)":
        uploaded_file = st.file_uploader("Upload a CSV or TXT file:", type=["csv", "txt"])

        text_column = None
        texts_to_analyze = []

        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.write("CSV Columns:")
                st.write(df.columns.tolist())
                text_column = st.selectbox("Select the column containing text data:", df.columns)

                if text_column:
                    texts_to_analyze = df[text_column].astype(str).tolist()
                else:
                    st.warning("Please select a text column.")

            elif uploaded_file.type == "text/plain":
                texts_to_analyze = uploaded_file.read().decode("utf-8").splitlines()
                texts_to_analyze = [line.strip() for line in texts_to_analyze if line.strip()] # Remove empty lines

            if st.button("Analyze File"):
                if texts_to_analyze:
                    st.subheader("Batch Analysis Results:")
                    progress_bar = st.progress(0)
                    total_texts = len(texts_to_analyze)
                    results = []
                    for i, text in enumerate(texts_to_analyze):
                        if text.strip():
                            sentiment_analysis_result = analyze_sentiment(text)
                            results.append({
                                'Original Text': text,
                                'Sentiment': get_label_name(sentiment_analysis_result['label']),
                                'Confidence': sentiment_analysis_result['score'] * 100
                            })
                        else:
                            results.append({
                                'Original Text': text,
                                'Sentiment': 'N/A',
                                'Confidence': 0.0
                            })
                        progress_bar.progress((i + 1) / total_texts)

                    # Update session state for batch results
                    st.session_state.analysis_results_df = pd.DataFrame(results)
                    st.dataframe(st.session_state.analysis_results_df)

                    sentiment_distribution_for_batch = st.session_state.analysis_results_df['Sentiment'].value_counts().reset_index()
                    sentiment_distribution_for_batch.columns = ['Sentiment', 'Count']
                    sentiment_distribution_for_batch['Color'] = sentiment_distribution_for_batch['Sentiment'].apply(get_sentiment_color)
                    st.session_state.sentiment_distribution_data = sentiment_distribution_for_batch

                    st.session_state.processed_texts_for_pdf = st.session_state.analysis_results_df[['Original Text', 'Sentiment', 'Confidence']].to_dict(orient='records')
                    st.session_state.last_analyzed_text_full_results = {} # Clear single text results
                    st.session_state.detailed_analysis_target_text = ""


                else:
                    st.warning("No valid text found in the uploaded file for analysis.")
        else:
            st.info("Upload a CSV or TXT file to begin batch analysis.")


    # --- Visualization Components (after analysis, if data exists in session state) ---
    if not st.session_state.analysis_results_df.empty and st.session_state.sentiment_distribution_data is not None:
        st.markdown("---")
        st.subheader("Sentiment Distribution 📈")

        # Bar Chart for Sentiment Distribution
        fig_bar = px.bar(st.session_state.sentiment_distribution_data, x='Sentiment', y='Count',
                         title='Overall Sentiment Distribution',
                         color='Sentiment',
                         color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'orange', 'N/A':'gray'},
                         text='Count')
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie Chart for Sentiment Distribution
        fig_pie_analysis = px.pie(st.session_state.sentiment_distribution_data, values='Count', names='Sentiment',
                                  title='Proportion of Sentiments',
                                  color='Sentiment',
                                  color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'orange', 'N/A':'gray'})
        st.plotly_chart(fig_pie_analysis, use_container_width=True)

        st.markdown("---")
        st.subheader("Comparative Analysis (If applicable)")
        if len(st.session_state.analysis_results_df) > 1: # Only show if more than one text was analyzed
            st.write("##### Top 5 Most Positive Texts:")
            positive_texts = st.session_state.analysis_results_df[st.session_state.analysis_results_df['Sentiment'] == 'Positive'].sort_values(by='Confidence', ascending=False).head(5)
            if not positive_texts.empty:
                for i, row in positive_texts.iterrows():
                    st.markdown(f"- **Confidence {row['Confidence']:.2f}%**: {row['Original Text']}")
            else:
                st.info("No positive texts found to display.")

            st.write("##### Top 5 Most Negative Texts:")
            negative_texts = st.session_state.analysis_results_df[st.session_state.analysis_results_df['Sentiment'] == 'Negative'].sort_values(by='Confidence', ascending=False).head(5)
            if not negative_texts.empty:
                for i, row in negative_texts.iterrows():
                    st.markdown(f"- **Confidence {row['Confidence']:.2f}%**: {row['Original Text']}")
            else:
                st.info("No negative texts found to display.")
        else:
            st.info("Comparative analysis is more relevant for batch processing or multiple texts.")

        # --- Export Options ---
        st.markdown("---")
        st.subheader("Download Results")
        if not st.session_state.analysis_results_df.empty:
            csv_export = st.session_state.analysis_results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv_export,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

            json_export = st.session_state.analysis_results_df.to_json(orient='records', indent=4)
            st.download_button(
                label="Download as JSON",
                data=json_export,
                file_name="sentiment_results.json",
                mime="application/json",
            )

            # PDF Download
            if st.button("Download as PDF Report"):
                pdf = PDF()
                pdf.add_page()
                pdf.chapter_title("NLP Analysis Report")

                # Add plots for sentiment distribution
                pdf.chapter_body("Summary of Sentiment Distribution:")
                bar_chart_path = "bar_chart.png"
                pie_chart_path = "pie_chart.png"

                plt.figure(figsize=(8, 5))
                colors = [get_sentiment_color(s) for s in st.session_state.sentiment_distribution_data['Sentiment']]
                plt.bar(st.session_state.sentiment_distribution_data['Sentiment'], st.session_state.sentiment_distribution_data['Count'], color=colors)
                plt.title('Overall Sentiment Distribution')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                for index, row in st.session_state.sentiment_distribution_data.iterrows():
                    plt.text(index, row['Count'], str(row['Count']), ha='center', va='bottom')
                plt.tight_layout()
                plt.savefig(bar_chart_path)
                plt.close()

                plt.figure(figsize=(8, 8))
                plt.pie(st.session_state.sentiment_distribution_data['Count'], labels=st.session_state.sentiment_distribution_data['Sentiment'],
                        autopct='%1.1f%%', colors=colors, startangle=90)
                plt.title('Proportion of Sentiments')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(pie_chart_path)
                plt.close()

                pdf.add_plot(bar_chart_path)
                pdf.add_plot(pie_chart_path)

                # Add detailed single text analysis results to PDF if available
                if st.session_state.last_analyzed_text_full_results and analysis_mode == "Single Text/Multiple Lines Analysis":
                    full_results = st.session_state.last_analyzed_text_full_results
                    pdf.add_page() # New page for detailed single analysis
                    pdf.chapter_title(f"Detailed Analysis of Selected Text:")
                    pdf.chapter_body(f"Original Text: {full_results['text']}\n")
                    pdf.chapter_body(f"Sentiment: {full_results['sentiment']['label']} (Confidence: {full_results['sentiment']['confidence']:.2f}%)\n")

                    if full_results['emotions']:
                        pdf.chapter_body("\nDetected Emotions:")
                        for emo in full_results['emotions']:
                            pdf.chapter_body(f"- {get_label_name(emo['label'])}: {emo['score']:.2f}%\n")

                    if full_results['entities']:
                        pdf.chapter_body("\nNamed Entities:")
                        for entity in full_results['entities']:
                            pdf.chapter_body(f"- {entity['word']} ({entity['entity_group']})\n")

                    pdf.chapter_body(f"\nSummary:\n{full_results['summary']}\n")
                    pdf.chapter_body("\nNote on explanations: For detailed keyword explanations and interactive charts, please refer to the web application.")


                # Add detailed batch text analysis results to PDF (for file upload or multiple lines)
                if st.session_state.processed_texts_for_pdf:
                    if analysis_mode == "Batch File Analysis (CSV/TXT)":
                        pdf.add_page() # New page for detailed batch analysis
                        pdf.chapter_title("Detailed Batch File Analysis:")
                        for item in st.session_state.processed_texts_for_pdf:
                            text_content = item.get('Original Text', 'N/A')
                            sentiment = item.get('Sentiment', 'N/A')
                            confidence = item.get('Confidence', 'N/A')
                            pdf.chapter_body(f"Text: {text_content[:200]}...\nSentiment: {sentiment} (Confidence: {confidence:.2f}%)\n") # Truncate long texts
                            pdf.ln(2) # Small line break for readability
                    elif analysis_mode == "Single Text/Multiple Lines Analysis" and len(st.session_state.processed_texts_for_pdf) > 1:
                        pdf.add_page()
                        pdf.chapter_title("Sentiment Analysis for Multiple Input Lines:")
                        for item in st.session_state.processed_texts_for_pdf:
                            text_content = item.get('Original Text', 'N/A')
                            sentiment = item.get('Sentiment', 'N/A')
                            confidence = item.get('Confidence', 'N/A')
                            pdf.chapter_body(f"Text: {text_content[:200]}...\nSentiment: {sentiment} (Confidence: {confidence:.2f}%)\n") # Truncate long texts
                            pdf.ln(2) # Small line break for readability

                pdf_output_bytes = bytes(pdf.output(dest='S')) # Explicitly convert bytearray to bytes

                # Clean up temporary plot files
                if os.path.exists(bar_chart_path):
                    os.remove(bar_chart_path)
                if os.path.exists(pie_chart_path):
                    os.remove(pie_chart_path)

                st.download_button(
                    label="Click here to download PDF",
                    data=pdf_output_bytes,
                    file_name="nlp_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("Run an analysis first to enable download options.")
    else:
        st.info("Perform an analysis above to see sentiment distribution and download options.")


# --- Main App Logic (Sidebar Navigation) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Analyze Content")) # Changed to "Analyze Content"

if page == "Home":
    landing_page()
elif page == "Analyze Content": # Updated page name
    analysis_page()
    