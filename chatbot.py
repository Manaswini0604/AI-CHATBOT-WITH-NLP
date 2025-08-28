import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download("punkt_tab")
nltk.download('averaged_perceptron_tagger_eng')
# ---- Ensure NLTK resources are available ----
def ensure_nltk_resources():
    resources = [
        'punkt', 
        'wordnet', 
        'omw-1.4', 
        'averaged_perceptron_tagger', 
        'stopwords'
    ]
    for r in resources:
        try:
            if r == 'punkt':
                nltk.data.find(f'tokenizers/{r}')
            else:
                nltk.data.find(f'corpora/{r}')
        except LookupError:
            nltk.download(r, quiet=True)

ensure_nltk_resources()

# ---- Helper: POS mapping for lemmatizer ----
def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer()

# ---- Tokenizer for TF-IDF ----
def tokenize_and_lemmatize(text):
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    cleaned = []
    for token, tag in pos_tags:
        if all(ch in string.punctuation for ch in token) or token.isnumeric():
            continue
        pos = _get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, pos=pos)
        if len(lemma) > 1:
            cleaned.append(lemma)
    return cleaned

# ---- FAQ knowledge base ----
faq_data = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I do for you?",
    "what is your name": "I'm PyBot — a small NLP chatbot built with NLTK and TF-IDF.",
    "who created you": "A developer using Python and NLP libraries created me.",
    "how are you": "I'm fine, thanks! How can I help you?",
    "what is nlp": "NLP (Natural Language Processing) is the field of AI that deals with human language.",
    "what is python": "Python is a popular programming language for AI, data science, and more.",
    "tell me a joke": "Why did the programmer quit his job? Because he didn't get arrays.",
    "what is machine learning": "Machine learning teaches systems to learn patterns from data.",
    "bye":"Goodbye! 👋",
}

# ---- Chatbot Class ----
class NLPTfidfChatbot:
    def __init__(self, faq_dict, similarity_threshold=0.25):
        self.questions = list(faq_dict.keys())
        self.answers = list(faq_dict.values())
        self.threshold = similarity_threshold

        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_lemmatize,
            stop_words='english',
            token_pattern=None
        )
        self.tfidf_questions = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_text):
        normalized = re.sub(r'[^\w\s]', '', user_text.lower()).strip()
        for q, a in zip(self.questions, self.answers):
            if normalized == re.sub(r'[^\w\s]', '', q.lower()).strip():
                return a

        user_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, self.tfidf_questions).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score >= self.threshold:
            return self.answers[best_idx]
        else:
            return "I'm sorry — I don't understand. Can you rephrase?"

# ---- Chat Loop ----
def chat_loop():
    bot = NLPTfidfChatbot(faq_data, similarity_threshold=0.25)
    print("PyBot is ready 🤖 — type 'bye' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("bye", "quit", "exit"):
            print("PyBot: Goodbye! 👋")
            break

        response = bot.get_response(user_input)
        print("PyBot:", response)

# ---- Entry point ----
if __name__ == "__main__":
    chat_loop()
