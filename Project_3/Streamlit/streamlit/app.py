# Import the libraries 
import streamlit as st 										# This helps us to structure the app
from streamlit_option_menu import option_menu 				# This lets us use a navigation bar 
from pathlib import Path 									# This lets us use the file path 
import re 													# To remove the punctuations and numbers from the text 
import matplotlib.pyplot as plt 							# To plot the bar charts
from sklearn.feature_extraction.text import CountVectorizer # To vectorize our text for getting the top words
import pandas as pd 										# To use dataframes
import plotly.express as px 								# To plot the wordcloud 
from wordcloud import WordCloud 							# To plot the wordcloud 
import praw 												# To get live data from reddit
import nltk 												# Get the Natural Language Toolkit
from nltk.corpus import stopwords 							# Get the stopwords
from nltk.stem import WordNetLemmatizer 					# To lemmatize text
import pickle 												# To get our trained model
from sklearn.feature_extraction.text import TfidfVectorizer # To vectorize our text
from sklearn.preprocessing import LabelEncoder 				# To convert the 0 and 1's back to the label names
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For sentiment analysis
import streamlit.components.v1 as components 				# To display the map from tableau 

# Import dataset
data_path = Path(__file__).parent / 'data/merged_df.csv'
df = pd.read_csv(data_path,lineterminator='\n')

# Import model
model_filepath = Path(__file__).parent / 'models/multinomial_naive_bayes_model.pkl'
model = pickle.load(open(model_filepath, 'rb')) 

# Stopwords
nltk.download('all')
stopwords = nltk.corpus.stopwords.words('english')

# Functions 
# Function to get data from reddit live
@st.cache_data(ttl=3600) # time to live (ttl). Streamlit invalidates any cached values after 1 hour (3600 seconds) and runs the cached function again.
def fetch_reddit_data(subreddit_name, limit=1000):
    # Initialize Reddit API client
    # Reddit PRAW Read-only instance
	reddit_read_only = praw.Reddit(client_id="g107SqPxa4_nJV7P4UIpgg",          # your client id
								client_secret="CgVmy57wFTM8hOLqGpu2UFUtPHHiGA", # your client secret
                            	user_agent="Scraper")                           # your user agent
	subreddit = reddit_read_only.subreddit(subreddit_name)
	top_posts = subreddit.hot(limit=limit)
	post_data = []
	for post in top_posts:
		post_data.append({
			"title": post.title,
			"body": post.selftext
			})
	df = pd.DataFrame(post_data)
	df['subreddit'] = subreddit_name.lower()  # Add subreddit column
	return df

# Function to get word type
def get_wordnet_pos(treebank_tag):
    # Map POS tag to first character lemmatize() accepts
    tag = treebank_tag[0].upper()
    tag_dict = {
        'N': 'n',  # Noun
        'V': 'v',  # Verb
        'R': 'r',  # Adverb
        'J': 'a'   # Adjective
    }
    return tag_dict.get(tag, 'n')  # Default to noun

# Function to lemmatrize text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text and determine the POS tag for each token
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    # Lemmatize each token based on its POS tag
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)) for token, pos_tag in pos_tags]
    # Join the lemmatized tokens into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

# Function to clean text data, handling NaN and non-string types
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuations
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
	# Remove all numbers from the text
    text = re.sub(r'\d+', '', text)
    # Lemmatize words
    text = lemmatize_text(text)
    # Remove stopwords
    tokens = text.split()
    text = ' '.join([word for word in tokens if word not in stopwords])
    return text

# Function to get top n words or n-grams
def get_top_n_words(corpus, n=1, max_features=10):
    vec = CountVectorizer(ngram_range=(n, n), max_features=max_features)
    bag_of_words = vec.fit_transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=False)
    return words_freq

# Function to plot horizontal bar chart
def plot_barh_chart(words_freq, color):
    words, frequencies = zip(*words_freq)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words, frequencies, color=color, alpha=1)
    ax.set_xlabel('Frequency', color='white')  # Set x-axis label color to white
    ax.set_ylabel('Words', color='white')  # Set y-axis label color to white
    ax.set_xticklabels(ax.get_xticks(), color='white')
    ax.set_yticklabels(words, color='white')  # Set y-axis tick label color to white
    ax.set_facecolor('none')  # Make the figure background transparent
    fig.patch.set_alpha(0)  # Make the figure background transparent
    st.pyplot(fig)

# Function to plot the word cloud
def plot_word_cloud(data, n=2, max_ngrams=10, color='Reds'):    
    # Create a dictionary of word frequencies
    word_frequencies = {item[0]: int(item[1]) for item in data}
    # Generate the word cloud using generate_from_frequencies
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA', colormap=color).generate_from_frequencies(word_frequencies)
    # Create a transparent figure
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    # Display the word cloud with transparency
    ax.imshow(wordcloud, interpolation='bilinear', alpha=0.9)
    ax.axis('off')
    # Set the figure background to transparent
    fig.patch.set_alpha(0.0)
    # Pass the figure object explicitly to st.pyplot
    st.pyplot(fig)

# Page configurations
st.set_page_config(
	page_title='Vegan or Meat lover?',
	page_icon='🥩',
	layout='wide',
	initial_sidebar_state='expanded'
	)

# This creates a navigation bar for the user to choose from
selected = option_menu(
	menu_title=None,
	options=["Insights", "Existing Businesses", "Prospective Businesses"],
	icons=['eyeglasses', 'bar-chart', 'pin-map'],
	default_index = 0,
	orientation='horizontal',
	# Define the styles
	styles = {
	'nav-link-selected': {
		'background-color': 'red',  
		'color': 'white'  
    	}
	}
	)

# Get the top 1000 posts from the hot section of the respective reddits
vegan_df = fetch_reddit_data('Vegan')
meat_df = fetch_reddit_data('Meat')

if selected == 'Insights':
	st.title('Vegan or Meat lover?')
	# This code determines the color and thickness of the line seperator
	style = "<div style='background-color:red; padding:2px'></div>"
	st.markdown(style, unsafe_allow_html = True)

	# Tab to switch between live and historical data
	live, historical = st.tabs(['Live', 'Historical'])

	with live:
		# Header
		st.header('Top Keywords')
		# Slider for number of words per keyword
		n_words_live = st.slider('Select the number of words per keyword', 1, 3, 1, key='n_words_live')
		# Slider for number of top keywords to be shown
		top_n_words_live = st.slider('Select the number of keywords to be shown', 6, 20, 10, key='top_n_words_live')

		# Create columns to separate meat and vegan
		left_column, right_column  = st.columns(2)
		# Left 
		left_column.subheader('Vegan')
		# Right
		right_column.subheader('Meat')
		
		# Get the top 1000 posts from the hot section of the respective reddits 
		#vegan_df = fetch_reddit_data('Vegan')
		#meat_df = fetch_reddit_data('Meat')

		# Merge the 2 dataframes
		merged_new_df = pd.concat([vegan_df, meat_df], ignore_index=True)
		# Drop rows where both 'title' and 'body' are empty as certain posts contains pictures
		merged_new_df.dropna(subset=['title', 'body'], how='all', inplace=True)

		# Merge the 'title' and 'body' columns into a single text column
		merged_new_df['text'] = merged_new_df['title'].apply(''.join) + ' ' + merged_new_df['body'].apply(''.join)

		# Apply the cleaning function to the 'text' column of merged dataframe (Remove punctuation / stopwords / lemmatise)
		merged_new_df['text'] = merged_new_df['text'].apply(clean_text)

		# Drop the 'title' and 'body' columns
		merged_new_df.drop(columns=["title", "body"], inplace=True)

		# Get top words or n-grams for 'meat'
		meat_text = merged_new_df[merged_new_df['subreddit'] == 'meat']['text']
		top_meat_words = get_top_n_words(meat_text, n=n_words_live, max_features=top_n_words_live)

		# Get top words or n-grams for 'vegan'
		vegan_text = merged_new_df[merged_new_df['subreddit'] == 'vegan']['text']
		top_vegan_words = get_top_n_words(vegan_text, n=n_words_live, max_features=top_n_words_live)

		# Plot for vegan in the left column (color: green)
		with left_column:
			st.write("Top Words for Vegan")
			# Generate and display the word cloud for "vegan"
			plot_word_cloud(top_vegan_words, n=n_words_live, max_ngrams=top_n_words_live, color='Greens')  # Call the function with the "vegan" text
			# Plot the bar chart
			plot_barh_chart(top_vegan_words, color='green')

		# Plot for meat in the right column (color: red)
		with right_column:
			st.write("Top Words for Meat")
			# Generate and display the word cloud for "meat"
			plot_word_cloud(top_meat_words,  n=n_words_live, max_ngrams=top_n_words_live, color='Reds')  # Call the function with the "meat" text
			# Plot the bar chart
			plot_barh_chart(top_meat_words, color='red')

	with historical:
		# Header
		st.header('Top Keywords')
		# Slider for number of words per keyword
		n_words = st.slider('Select the number of words per keyword', 1, 3, 1, key='n_words')
		# Slider for number of top keywords to be shown
		top_n_words = st.slider('Select the number of keywords to be shown', 6, 20, 10, key='top_n_words')

		# Create columns to separate meat and vegan
		left_column, right_column  = st.columns(2)
		# Left 
		left_column.subheader('Vegan')
		# Right
		right_column.subheader('Meat')

		# Get top words or n-grams for 'meat'
		meat_text = df[df['subreddit'] == 'meat']['text']
		top_meat_words = get_top_n_words(meat_text, n=n_words, max_features=top_n_words)

		# Get top words or n-grams for 'vegan'
		vegan_text = df[df['subreddit'] == 'vegan']['text']
		top_vegan_words = get_top_n_words(vegan_text, n=n_words, max_features=top_n_words)

		# Plot for vegan in the left column (color: green)
		with left_column:
			st.write("Top Words for Vegan")
			# Generate and display the word cloud for "vegan"
			plot_word_cloud(top_vegan_words, n=n_words, max_ngrams=top_n_words, color='Greens')  # Call the function with the "vegan" text
			# Plot the bar chart
			plot_barh_chart(top_vegan_words, color='green')
			# Write down key points
			st.subheader('Analysis (Vegan🌿)')
			st.write('The r/vegan comunity are concerned about the :green[animal welfare] as a :green[plant-based diet] keeps farm animals from :green[slaughter]. They even post websites like the [Vegan Hacktivists](https://veganhacktivists.org/) to show how people can support the vegan movement. This is why the top words that appear are: animal, vegan, plant base, animal right, plant based diet, stop eat meat, animal right activist and plant base milk.')

		# Plot for meat in the right column (color: red)
		with right_column:
			st.write("Top Words for Meat")
			# Generate and display the word cloud for "meat"
			plot_word_cloud(top_meat_words,  n=n_words, max_ngrams=top_n_words, color='Reds')  # Call the function with the "meat" text
			# Plot the bar chart
			plot_barh_chart(top_meat_words, color='red')
			# Write down key points 
			st.subheader('Analysis (Meat🥩)')
			st.write('The r/meat comunity are concerned about the :red[freshness of meats], :red[quality of meats] and :red[methods of cooking]. The most popular meats discussed are :red[beef] and :red[pork] and they are against the wastage of food. This is why the top words that appear are: beef, pork, reverse sear, sous vide, dry age, best way cook and still safe eat.')


if selected == 'Existing Businesses':
	st.title('Vegan or Meat lover?')
	style = "<div style='background-color:red; padding:2px'></div>"
	st.markdown(style, unsafe_allow_html = True)

	st.header('Customer Analysis')
	st.write("Please upload your CSV file with only 1 column named 'text' where you place your customer's feedback")

	# Get the data from the customer
	data = st.file_uploader('Upload CSV file')

	# If the user has not uploaded any data, we can opt to show him a sample.
	button = False
	if st.button('See how our model works on a sample dataset'):
		button = True			
		# Import sample dataset
		# Get the path to the CSV file using pathlib
		data_file_path = Path('data') / 'Afterglow_reviews_50.csv'
		# Read the CSV file into a DataFrame
		data = pd.read_csv(data_file_path)

		# This works too
		# data = pd.read_csv('data/Afterglow_reviews_50.csv')

	# Check if data is uploaded
	if data is not None:
		if button != True:
			# Read the CSV file into a DataFrame
			data = pd.read_csv(data)
		st.success('Submitted')

		# Check if the file they upload is in the right format
		if 'text' in data.columns:
		    st.write(":green[Accepted file format. Column 'text' exists in the uploaded CSV file.]")
		else:
		    st.write(":red[Please upload the CSV file with a column named 'text' containing your customer's feedback.]")

    	# Apply the cleaning function to the 'text' column of the DataFrame (Remove punctuation / stopwords / lemmatise)
		data['text'] = data['text'].apply(clean_text)

		# # Load the pre-trained TF-IDF vectorizer
		# vec_filepath = Path(__file__).parent / 'models/vectorizer.pkl'
		# with open(vec_filepath, 'rb') as file:
		#     vectorizer = pickle.load(file)

		# Assuming you have your training data in a DataFrame called 'df'
		# Fit the vectorizer on the training data to ensure consistent features
		vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
		X_train = vectorizer.fit_transform(df['text'])
		# Now, 'X_train' contains the TF-IDF features for the training text data
		# Assuming you have your test data in a DataFrame called 'test_df'
		# Vectorize the text data using the loaded TF-IDF vectorizer
		X_test = vectorizer.transform(data['text'])

		# # Import vectorizer
		# vec_filepath = Path(__file__).parent / 'models/vectorizer.pkl'
		# # Assuming you have a pre-trained TF-IDF vectorizer saved as vectorizer.pkl
		# vectorizer = pickle.load(open(vec_filepath, 'rb')) 
		# # Fit the vectorizer on the training data to ensure consistent features
		# vectorizer.fit(df['text'])
		# # Vectorize the text data using the loaded TF-IDF vectorizer
		# X_test = vectorizer.transform(data['text'])

		# Encode the 'subreddit' column
		le = LabelEncoder()
		y = le.fit_transform(df['subreddit'])

		# Make predictions
		predictions = model.predict(X_test)
		# Convert numerical predictions back to labels
		predicted_labels = le.inverse_transform(predictions)
		# Add the predicted labels to the DataFrame
		data['predicted_label'] = predicted_labels

		# Create columns to separate diet preferences and top keywords
		left_column, mid_column, right_column  = st.columns(3)
		# Left 
		left_column.subheader('Customer diet preferences')
		# Middle
		mid_column.subheader('Top keywords')
		# Right
		right_column.subheader('Sentiment Analysis')

		with left_column:
			# Check the majority class
			if data['predicted_label'].value_counts().index[0] == 'meat':
				label = ['meat', 'vegan']
				colors = ['Red', 'Green']
			else:
				label = ['vegan', 'meat']
				colors = ['Green', 'Red']

			# Display a pie chart of predicted labels with a transparent background
			fig, ax = plt.subplots()
			wedges, texts, autotexts = ax.pie(data['predicted_label'].value_counts(), labels=label,
			                                  autopct='%1.1f%%', textprops={'color': 'white'}, colors=colors)
			ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
			# Modify the appearance of labels
			for text in texts + autotexts:
			    text.set(color='white')
			fig.set_facecolor('none')  # Make the figure background transparent
			st.pyplot(fig)
			# Display the majority class
			majority_class = data['predicted_label'].mode().values[0] # Mode returns the label that appears the most
			st.write(f"The majority of your customers have :green[{majority_class}] preferences.")

		with mid_column:
			# Display the top keywords
			# Slider for number of words per keyword
			n_words_cust = st.slider('Select the number of words per keyword', 1, 3, 1, key='n_words')
			# Slider for number of top keywords to be shown
			top_n_words_cust = st.slider('Select the number of keywords to be shown', 6, 20, 10, key='top_n_words')

			# Get the top words or n-grams for the whole dataset
			top_words_cust = get_top_n_words(data['text'], n=n_words_cust, max_features=top_n_words_cust)

			# Plot the word cloud and barh chart by the majority class color
			if majority_class == "vegan":
				color_cust1 = "Greens"
				color_cust2 = "green"
			else:
				color_cust1 = "Reds"
				color_cust2 = "red"

			# Generate and display the word cloud for the whole dataset
			plot_word_cloud(top_words_cust, n=n_words_cust, max_ngrams=top_n_words_cust, color=color_cust1)  # Call the function with the "vegan" text
			# Plot the bar chart
			plot_barh_chart(top_words_cust, color=color_cust2)
			st.write("You can use include the :green[top keywords] for your :green[advertising campaigns] so people searching for those food will be led to your restaurant's website.")

		with right_column:
			# Sentiment analysis
			# Create a sentiment analyzer
			analyzer = SentimentIntensityAnalyzer()
			# Function to analyze sentiment for a text
			def analyze_sentiment(text):
			    sentiment_scores = analyzer.polarity_scores(text)
			    # Interpret the compound score
			    if sentiment_scores['compound'] >= 0.05:
			        return 'Positive'
			    elif sentiment_scores['compound'] <= -0.05:
			        return 'Negative'
			    else:
			        return 'Neutral'
			# Apply sentiment analysis to each row in the 'text' column
			data['sentiment_score'] = data['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
			data['sentiment_label'] = data['text'].apply(analyze_sentiment)

			# Get the count of each sentiment label
			sentiment_counts = data['sentiment_label'].value_counts()

			# Create a figure and axis for the bar chart
			fig, ax = plt.subplots()
			# Plot the sentiment counts as a bar chart with a transparent background
			ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'black'], alpha=1)
			# Add labels and title with white text
			ax.set_xlabel('Sentiments', color='white')
			ax.set_ylabel('Count', color='white')
			# Set x-axis and y-axis label colors to white
			ax.xaxis.label.set_color('white')
			ax.yaxis.label.set_color('white')
			# Set x-axis and y-axis tick colors to white
			ax.tick_params(axis='x', colors='white')
			ax.tick_params(axis='y', colors='white')
			ax.set_facecolor('none')  # Make the figure background transparent
			fig.patch.set_alpha(0)  # Make the figure background transparent
			# Display the bar chart using Streamlit
			st.pyplot(fig)

			# Check if there are more positive or negative reviews
			if 'Positive' in sentiment_counts and 'Negative' in sentiment_counts:
			    if sentiment_counts['Positive'] > sentiment_counts['Negative']:
			        st.markdown("""The majority of reviews are :green[positive]. Good job!  
			        				See what are some :green[top food items] mentioned in the reviews by :green[toggling] the number of words per keyword.""")
			        st.balloons()
			    elif sentiment_counts['Positive'] < sentiment_counts['Negative']:
			        st.write("The majority of reviews are :red[negative]. Time to change things up. See what are some hot food items in the live analysis for some ideas")
			        st.balloons()
			    else:
			        st.write("There is an equal number of positive and negative reviews.")
			        st.balloons()
		
if selected == "Prospective Businesses":
	st.title('Vegan or Meat lover?')
	style = "<div style='background-color:red; padding:2px'></div>"
	st.markdown(style, unsafe_allow_html = True)

	st.header('What kind of restaurant would you like to open?')
	option = st.selectbox(
    ' ',
    ('Choose an option', 'Vegan', 'Meat')) # Choose an option is the default value

	if option != 'Choose an option': 
		st.write('You selected:', option)
		if option == 'Vegan':
			# Show them a map of the vegan restaurants in Singapore
			st.header('Map of vegan restaurants in Singapore')
			# Link to tableau to show the restaurant locations in Singapore
			components.html(
			    "<div class='tableauPlaceholder' id='viz1695873506236' style='position: relative'><noscript><a href='#'><img alt='Vegan Restaurants in Singapore- The redder the spot, the higher is the competition (more restaurants in the area). ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project_16955750624040&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Project_16955750624040&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;Project_16955750624040&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1695873506236');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script> ",
			    height=800, width=1270
			)
			st.write("There are more restaurants in the central part of Singapore as it is the area with the highest footfall for working adults. Since the prices do not vary that much based on location, our recommendation is to open a vegan restaurant in a :green[residential area] where :green[rent is lower] and there are less competitors.")

			# Create columns to separate diet preferences and top keywords
			left_column, right_column  = st.columns(2)
			# Titles
			left_column.header('Suppliers in Singapore')
			right_column.header('Sample Menu')

			with left_column:
				# Show them a list of vegan suppliers
				st.markdown(
				"""
				Let us help you build your network! Here are some top ethical vegan suppliers in Singapore:
				- [LIM THIAM CHWEE FOOD SUPPLIER PTE LTD](http://www.ltcfood.com.sg/)
					- LTC specialises in imports, distribution of all fresh vegetables, fruits and wet provisions (e.g. tofu, noodles, etc) for Singapore’s food and beverage industry.
				- [BAN CHOON MARKETING PTE LTD](http://www.banchoon.com.sg/)
					- Ban Choon Marketing specializes in delivering a diverse selection of fresh produce, including a core focus on berries, premium salads, and Japanese fruits and vegetables. 
				- [LIM KIAN SENG FOOD SUPPLIER PTE LTD](https://limkianseng.com.sg/)
					- Lim Kian Seng Food Supplier Pte Ltd has been in this business for over 40 years. They specialise in vegetable & fruit wholesale and supply to hotels & restaurant as well.
				"""
				)

			with right_column:
				# Show them a sample menu of hot food items
				# Sample menu data
				st.markdown(
					"""
					Plant-Based Delights:
					- Lays Chaat Explosion: Experience the playful fusion of Lays with a burst of delightful Indian chaat spices. A snack like no other. ($12)
					- Hummus Bowl: Experience a Mediterranean journey of flavors with our signature Hummus Bowl. A delightful union of creamy hummus and vibrant, fresh vegetables, this bowl is a celebration of wholesome goodness. ($16)
					- Chilli Garlic Noodles Extravaganza: Delve into the aromatic and spicy world of our Chilli Garlic Noodles. A dish that'll awaken your taste buds. ($20)
					- Vegan Cheese Pizza Delight: Savor the gooey, melty goodness of our Vegan Cheese Pizza, topped with an array of colorful, fresh vegetables. ($22)					

					Beverages and deserts:
					- Creamy Oat Milk Elixir: Enjoy a creamy, dairy-free alternative with our Oat Milk. A perfect addition to your favorite beverages. ($6)
					- Almond Milk Bliss: Experience the nutty richness of our Almond Milk, a delightful companion to your morning coffee or tea. ($6)
					- Marble Cake Symphony: Indulge in the harmonious blend of flavors in our marble cake, a sweet symphony that's entirely plant-based. ($18)
					""")

			# Show them what are some hot food to give them ideas for their menu
			# Display photos from instagram hashtag
			st.header('Hot vegan food in Singapore')
			components.html("<div class='sk-ww-instagram-hashtag-feed' data-embed-id='201472'></div><script src='https://widgets.sociablekit.com/instagram-hashtag-feed/widget.js' async defer></script>",
				height=885, width=1270)


		elif option == 'Meat':
			# Show them a map of the meat restaurants in Singapore
			st.header('Map of meat restaurants in Singapore') 
			# Link to tableau to show the restaurant locations in Singapore
			components.html(
			    "<div class='tableauPlaceholder' id='viz1695873582190' style='position: relative'><noscript><a href='#'><img alt='Meat Restaurants in Singapore- The redder the spot, the higher is the competition (more restaurants in the area). ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;ProjectMeat&#47;Sheet2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ProjectMeat&#47;Sheet2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;ProjectMeat&#47;Sheet2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1695873582190');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script> ",
			    height=800, width=1270
			)
			st.write("There are more restaurants in the central part of Singapore as it is the area with the highest footfall for working adults. Since the prices do not vary that much based on location, our recommendation is to open a meat restaurant in a :green[residential area] where :green[rent is lower] and there are less competitors.")

			# Create columns to separate diet preferences and top keywords
			left_column, right_column  = st.columns(2)
			# Titles
			left_column.header('Suppliers in Singapore')
			right_column.header('Sample Menu')

			with left_column:
				# Show them a list of vegan suppliers
				st.markdown(
				"""
				Let us help you build your network! Here are some top meat suppliers in Singapore:
				- [HLRB FOOD PTE LTD](https://www.agri-biz.com/companies/hlrb-food-pte-ltd)
					- Specialised in the supply of fresh and frozen poultry products as well as provide safe and quality products to customers at competitive prices. Supply to wholesalers, food manufacturers, restaurants, food stalls and wet markets.
				- [HEN TICK FOODS PTE LTD](http://www.hentick.com.sg/)
					- A main importer/exporter/wholesaler and distributor of chilled, fresh frozen foods and manufacturer of RTC, RTE food products in Singapore for over 40 years with fully equipped cold stores, food processing and packaging facilities.
				- [SONG FISH DEALER PTE LTD ](https://songfish.com.sg/)
					- Offers a wide range of fresh and frozen seafood products in the local market. They cater to hotels, restaurants, caterers, and ship chandlers, and also provide wholesale and retail poultry options.
				"""
				)

			with right_column:
				# Show them a sample menu of hot food items
				# Sample menu data
				st.markdown(
					"""
					Beef Selections:
					- Succulent Beef: Tender and juicy beef cut to perfection. ($15)
					- Ground Beef Symphony: A harmonious blend of flavors in every bite. ($12)
					- Beef Ribs Extravaganza: Fall-off-the-bone ribs, slow-cooked to perfection. ($18)
					- Sirloin Tip Delight: A delightful and lean sirloin cut for the connoisseur in you. ($20)
					- Beef Fat Bliss: A flavor enhancer to elevate your culinary creations. ($10)
					
					Pork Delights:
					- Braised Pork Perfection: Our signature pork offering, a true taste of excellence. ($14)
					- Pork Shoulder Paradise: Slow-cooked pork shoulder, tender and bursting with flavor. ($10)
					- Uncured Bacon Feast: A bacon lover's dream—crisp, flavorful, and unforgettable. ($12)
					- Pork Chop Elegance: Thick, juicy pork chops for a hearty and satisfying meal. ($16)

					Specialties:
					- Sous Vide Steak: Immerse yourself in the sous vide experience. ($20)
					- New York Strip Classic: Savor the exquisite flavor of our carefully grilled New York Strip. A true classic for the discerning palate. ($25)
					- Brisket Bonanza: A smoky and tender beef brisket that melts in your mouth. ($25)
					""")

			# Show them what are some hot food to give them ideas for their menu
			# Display photos from instagram hashtag
			st.header('Hot food in Singapore')
			components.html("<div class='sk-ww-instagram-hashtag-feed' data-embed-id='201479'></div><script src='https://widgets.sociablekit.com/instagram-hashtag-feed/widget.js' async defer></script>",
				height=885, width=1270)