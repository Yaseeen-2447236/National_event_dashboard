import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageEnhance
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

# Generate synthetic dataset
def generate_dataset():
    np.random.seed(42)
    tracks = ['AI & ML', 'Data Science', 'Cybersecurity', 'IoT']
    colleges = ['IIT Bombay', 'IIT Delhi', 'IIT Madras', 'NIT Trichy', 'BITS Pilani']
    states = ['Andhra Pradesh', 'Tamil Nadu', 'Maharashtra', 'Karnataka']
    genders = ['Male', 'Female', 'Other']
    age_groups = ['18-22', '23-27', '28-32', '33+']
    reviews = [
        "The session was highly engaging and informative.",
        "Great insights shared by the speakers.",
        "The content was well-structured and easy to follow.",
        "I learned a lot about the latest trends in the field.",
        "The presentation could have been more interactive.",
        "Excellent delivery and practical examples.",
        "The session exceeded my expectations.",
        "Good session, but the Q&A could have been longer.",
        "Very detailed and insightful presentation.",
        "Highly relevant topics covered.",
        "The speakers were very knowledgeable.",   
        "Very well-organized event.",
        "The networking opportunities were great.",
        "Loved the way the concepts were explained."
    ]
    
    data = []
    for i in range(400):
        participant = {
            'Participant_ID': f'P{i+1}',
            'Day': np.random.choice(['Day 1', 'Day 2', 'Day 3', 'Day 4']),
            'Track': np.random.choice(tracks),
            'College': np.random.choice(colleges),
            'State': np.random.choice(states),
            'Gender': np.random.choice(genders),
            'Age_Group': np.random.choice(age_groups),
            'Feedback': np.random.choice(reviews),
            'Rating': np.random.randint(1, 6),
            'Attendance': np.random.choice(['Present', 'Absent'])
        }
        data.append(participant)
    df = pd.DataFrame(data)
    df.to_excel("poster_presentation_data.xlsx", index=False)
    return df

# Load dataset
df = generate_dataset()

# Streamlit UI
st.title("National Poster Presentation Event Dashboard")

# Sidebar Filters
track_filter = st.sidebar.multiselect("Select Track", df['Track'].unique(), default=df['Track'].unique())
state_filter = st.sidebar.multiselect("Select State", df['State'].unique(), default=df['State'].unique())

# Dynamically update the college dropdown based on selected states
if state_filter:
    filtered_colleges = df[df['State'].isin(state_filter)]['College'].unique()
else:
    filtered_colleges = df['College'].unique()

college_filter = st.sidebar.multiselect("Select College", options=filtered_colleges, default=filtered_colleges)

# Apply all filters to the dataset
filtered_df = df[
    (df['Track'].isin(track_filter)) &
    (df['State'].isin(state_filter)) &
    (df['College'].isin(college_filter))
]

# Sidebar Button for Additional Insights
show_additional_insights = st.sidebar.button("Show Additional Insights")

# Participation Trends with Chart Type Selection
st.subheader("Participation Trends")
chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Line Plot", "Box Plot", "Heatmap"], key="chart_type")

if chart_type == "Bar Chart":
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    sns.countplot(data=filtered_df, x='Track', ax=ax[0, 0], palette='coolwarm')
    ax[0, 0].set_title("Track-wise Participation")
    sns.countplot(data=filtered_df, x='Day', ax=ax[0, 1], palette='viridis')
    ax[0, 1].set_title("Day-wise Participation")
    sns.countplot(data=filtered_df, x='College', ax=ax[0, 2], palette='pastel')
    ax[0, 2].set_title("College-wise Participation")
    sns.countplot(data=filtered_df, x='State', ax=ax[1, 0], palette='muted')
    ax[1, 0].set_title("State-wise Participation")
    sns.countplot(data=filtered_df, x='Gender', ax=ax[1, 1], palette='Set2')
    ax[1, 1].set_title("Gender-wise Participation")
    sns.countplot(data=filtered_df, x='Age_Group', ax=ax[1, 2], palette='Set3')
    ax[1, 2].set_title("Age Group-wise Participation")
    st.pyplot(fig)

elif chart_type == "Pie Chart":
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    filtered_df['Track'].value_counts().plot.pie(ax=ax[0, 0], autopct='%1.1f%%', colors=sns.color_palette('coolwarm'))
    ax[0, 0].set_title("Track-wise Participation")
    ax[0, 0].set_ylabel("")
    filtered_df['Day'].value_counts().plot.pie(ax=ax[0, 1], autopct='%1.1f%%', colors=sns.color_palette('viridis'))
    ax[0, 1].set_title("Day-wise Participation")
    ax[0, 1].set_ylabel("")
    filtered_df['College'].value_counts().plot.pie(ax=ax[0, 2], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    ax[0, 2].set_title("College-wise Participation")
    ax[0, 2].set_ylabel("")
    filtered_df['State'].value_counts().plot.pie(ax=ax[1, 0], autopct='%1.1f%%', colors=sns.color_palette('muted'))
    ax[1, 0].set_title("State-wise Participation")
    ax[1, 0].set_ylabel("")
    filtered_df['Gender'].value_counts().plot.pie(ax=ax[1, 1], autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    ax[1, 1].set_title("Gender-wise Participation")
    ax[1, 1].set_ylabel("")
    filtered_df['Age_Group'].value_counts().plot.pie(ax=ax[1, 2], autopct='%1.1f%%', colors=sns.color_palette('Set3'))
    ax[1, 2].set_title("Age Group-wise Participation")
    ax[1, 2].set_ylabel("")
    st.pyplot(fig)

elif chart_type == "Line Plot":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=filtered_df, x='Day', y='Rating', hue='Track', marker='o', ax=ax)
    ax.set_title("Day-wise Rating Trends by Track")
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

elif chart_type == "Box Plot":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x='Track', y='Rating', palette='coolwarm', ax=ax)
    ax.set_title("Rating Distribution by Track")
    ax.set_xlabel("Track")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

elif chart_type == "Heatmap":
    st.write("Heatmap of Ratings by Track and Day")
    heatmap_data = filtered_df.pivot_table(values='Rating', index='Track', columns='Day', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Average Ratings by Track and Day")
    st.pyplot(fig)

# Rating Distribution
st.subheader("Rating Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=filtered_df, x='Rating', bins=5, kde=True, color='skyblue')
ax.set_title("Distribution of Ratings")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Word Cloud for Feedback
st.subheader("Track-wise Feedback Word Cloud")
selected_track = st.selectbox("Select Track for Word Cloud", df['Track'].unique(), key="wordcloud_track")
feedback_text = " ".join(df[df['Track'] == selected_track]['Feedback'])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(feedback_text)
st.image(wordcloud.to_array(), use_column_width=True)

# Text Similarity Analysis
st.subheader("Feedback Similarity Analysis")
tfidf = TfidfVectorizer()
feedback_matrix = tfidf.fit_transform(df[df['Track'] == selected_track]['Feedback'])
similarity_matrix = cosine_similarity(feedback_matrix, feedback_matrix)
st.write("Feedback Similarity Matrix:")
st.dataframe(similarity_matrix)

# Reviews Section with Text Processing and Dropdowns
st.subheader("Participant Reviews with Text Processing")

# Dropdown to select track for reviews
selected_review_track = st.selectbox("Select Track for Reviews", df['Track'].unique(), key="text_processing_track")

# Dropdown for selecting text processing method
text_processing_method = st.selectbox("Select Text Processing Method", ["None", "Stemming", "Lemmatization"], key="text_processing_method")

# Dropdown to select the number of reviews to display
num_reviews = st.selectbox("Number of Reviews to Display", [5, 10, 20, 50], key="num_reviews")

# Dropdown to sort reviews by relevance or rating
sort_by = st.selectbox("Sort Reviews By", ["Most Relevant", "Highest Rating", "Lowest Rating"], key="sort_reviews")

# Input box for text matching
search_text = st.text_input("Enter text to search in reviews", "", key="search_text")

# Filter reviews based on selected track
reviews_df = df[df['Track'] == selected_review_track][['Participant_ID', 'Feedback', 'Rating']]

# Filter reviews based on matching text
if search_text:
    reviews_df = reviews_df[reviews_df['Feedback'].str.contains(search_text, case=False, na=False)]

# Sort reviews based on the selected sorting method
if sort_by == "Highest Rating":
    reviews_df = reviews_df.sort_values(by="Rating", ascending=False)
elif sort_by == "Lowest Rating":
    reviews_df = reviews_df.sort_values(by="Rating", ascending=True)

# Limit the number of reviews to display
reviews_df = reviews_df.head(num_reviews)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply text processing
def process_text(text, method):
    tokens = word_tokenize(text)
    if method == "Stemming":
        return " ".join([stemmer.stem(token) for token in tokens])
    elif method == "Lemmatization":
        return " ".join([lemmatizer.lemmatize(token) for token in tokens])
    return text

# Highlight searched words in feedback
def highlight_text(text, word):
    if word:
        return text.replace(word, f"<span class='highlight-word'>{word}</span>")
    return text

# Display the filtered reviews with highlighted text
if not reviews_df.empty:
    st.write(f"Reviews for {selected_review_track}:")
    for _, row in reviews_df.iterrows():
        processed_feedback = process_text(row['Feedback'], text_processing_method)
        highlighted_feedback = highlight_text(processed_feedback, search_text)
        st.markdown(f"""
            <div class="review-container">
                <p><strong>Participant ID:</strong> {row['Participant_ID']}</p>
                <p><strong>Feedback:</strong> {highlighted_feedback}</p>
                <p><strong>Rating:</strong> {'⭐' * row['Rating']}</p>
            </div>
            <hr>
        """, unsafe_allow_html=True)
else:
    st.write("No reviews available for the selected track.")

# Add custom CSS for colorful styling and word highlighting
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: #333;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: #ecf0f1;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    .highlight-word {
        background-color: #ffff00; /* Yellow highlight */
        color: #d35400; /* Orange text */
        font-weight: bold;
    }
    .review-container {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Day-wise Image Gallery with Dropdown
st.subheader("Day-wise Image Gallery")
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
if uploaded_files:
    day_filter = st.selectbox("Select Day for Image Gallery", ['Day 1', 'Day 2', 'Day 3', 'Day 4'], key="day_filter")
    cols = st.columns(3)  # Display images in a grid with 3 columns
    for idx, file in enumerate(uploaded_files):
        img = Image.open(file)
        # Simulate day-based filtering (replace with actual logic if available)
        if idx % 4 == ['Day 1', 'Day 2', 'Day 3', 'Day 4'].index(day_filter):
            with cols[idx % 3]:  # Align images in columns
                st.image(img, caption=f"{file.name} - {day_filter}", use_column_width=True)

# Custom Image Processing Component
st.subheader("Custom Image Processing for Track-related Photos")
if uploaded_files:
    track_filter = st.selectbox("Select Track for Image Processing", df['Track'].unique(), key="track_filter")
    enhancement_type = st.selectbox("Select Enhancement Type", ["Brightness", "Contrast", "Sharpness"], key="enhancement_type")
    enhancement_factor = st.slider("Select Enhancement Factor", 0.5, 3.0, 1.0, key="enhancement_factor")
    cols = st.columns(3)  # Display enhanced images in a grid with 3 columns
    for idx, file in enumerate(uploaded_files):
        img = Image.open(file)
        enhancer = None
        if enhancement_type == "Brightness":
            enhancer = ImageEnhance.Brightness(img)
        elif enhancement_type == "Contrast":
            enhancer = ImageEnhance.Contrast(img)
        elif enhancement_type == "Sharpness":
            enhancer = ImageEnhance.Sharpness(img)
        enhanced_img = enhancer.enhance(enhancement_factor)
        # Simulate track-based filtering (replace with actual logic if available)
        if idx % len(df['Track'].unique()) == list(df['Track'].unique()).index(track_filter):
            with cols[idx % 3]:  # Align enhanced images in columns
                st.image(enhanced_img, caption=f"Enhanced {file.name} - {track_filter}", use_column_width=True)

# Additional Insights Section at the End
st.subheader("Additional Insights")

# Chart 1: Average Rating by Gender
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=filtered_df, x='Gender', y='Rating', palette='coolwarm', ax=ax)
ax.set_title("Average Rating by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# Chart 2: Attendance Distribution by Track
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=filtered_df, x='Track', hue='Attendance', palette='Set2', ax=ax)
ax.set_title("Attendance Distribution by Track")
ax.set_xlabel("Track")
ax.set_ylabel("Count")
st.pyplot(fig)

# Chart 3: State-wise Average Rating
fig, ax = plt.subplots(figsize=(8, 5))
state_avg_rating = filtered_df.groupby('State')['Rating'].mean().reset_index()
sns.barplot(data=state_avg_rating, x='State', y='Rating', palette='muted', ax=ax)
ax.set_title("State-wise Average Rating")
ax.set_xlabel("State")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# Chart 4: Age Group-wise Participation
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=filtered_df, x='Age_Group', palette='pastel', ax=ax)
ax.set_title("Age Group-wise Participation")
ax.set_xlabel("Age Group")
ax.set_ylabel("Count")
st.pyplot(fig)

# Chart 5: Day-wise Average Rating
fig, ax = plt.subplots(figsize=(8, 5))
day_avg_rating = filtered_df.groupby('Day')['Rating'].mean().reset_index()
sns.lineplot(data=day_avg_rating, x='Day', y='Rating', marker='o', color='orange', ax=ax)
ax.set_title("Day-wise Average Rating")
ax.set_xlabel("Day")
ax.set_ylabel("Average Rating")
st.pyplot(fig)

st.success("Dashboard Development Complete!")
