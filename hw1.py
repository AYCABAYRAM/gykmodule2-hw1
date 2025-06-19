import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

corpus = [
    "Artificial Intelligence is revolutionizing every industry.",
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks to mimic the human brain.",
    "AI applications include speech recognition and computer vision.",
    "Natural language processing enables AI to understand human language.",
    "AI is widely used in healthcare for diagnosis and treatment recommendations.",
    "Self-driving cars are powered by artificial intelligence.",
    "AI helps automate repetitive tasks in businesses.",
    "Ethics in AI is a growing area of concern.",
    "AI continues to evolve with more advanced algorithms."
]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(doc):
    # Lowercase + tokenization
    tokens = nltk.word_tokenize(doc.lower())
    # Noktalama ve stopword temizliği + lemmatization
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in string.punctuation and token not in stop_words
    ]
    return ' '.join(cleaned)

preprocessed_corpus = [preprocess(doc) for doc in corpus]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_corpus)

print("Feature (kelime) isimleri:")
print(vectorizer.get_feature_names_out())
print("\nTF-IDF Array:")
print(X.toarray())