from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define custom transformers for each preprocessing step
class LowerCaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.iloc[:, 0] = X_transformed.iloc[:, 0].str.lower()
        return X_transformed

class CleanLinksTransformer(BaseEstimator, TransformerMixin):
    @staticmethod
    def clean_links(text):
        # Define a regular expression pattern to match URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        # Use the sub() function to replace URLs with an empty string
        return url_pattern.sub('', text)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.iloc[:, 0] = X_transformed.iloc[:, 0].apply(self.clean_links)
        return X_transformed

class RemovePunctuationAndWhitespaceTransformer(BaseEstimator, TransformerMixin):
    @staticmethod
    def remove_punctuation_and_whitespace(text):
        # Use regular expression to remove punctuation and whitespace
        cleaned_text = re.sub(r'[?(),.;:!$%^&*_-]', '', text)
        cleaned_text = re.sub(r'\d+', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.iloc[:, 0] = X_transformed.iloc[:, 0].apply(self.remove_punctuation_and_whitespace)
        return X_transformed

class StopWordRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = []

    def fit(self, X, y=None):
        stop_factory = StopWordRemoverFactory()
        more_stopword = ["yg", "dgn", 'dg', 'kok', 'nya', 'aja', 'lg']
        stopwords = ['ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'amat']
        negation_words = ['tidak', 'bukan', 'jangan', 'tak', 'tiada']
        self.stopwords = stop_factory.get_stop_words() + more_stopword + stopwords
        self.stopwords = [word for word in self.stopwords if word not in negation_words]
        self.remover = stop_factory.create_stop_word_remover(self.stopwords)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.iloc[:, 0] = X_transformed.iloc[:, 0].apply(lambda text: " ".join([word for word in text.split() if word not in self.stopwords]))
        return X_transformed

class PosTaggerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    @staticmethod
    def pos_check(x, flag):
        pos_dic = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
        }
        cnt = 0
        try:
            wiki = TextBlob(x)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_dic[flag]:
                    cnt += 1
        except:
            pass
        return cnt

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['noun_count'] = X_transformed.iloc[:, 0].apply(lambda x: self.pos_check(x, 'noun'))
        X_transformed['verb_count'] = X_transformed.iloc[:, 0].apply(lambda x: self.pos_check(x, 'verb'))
        X_transformed['adj_count'] = X_transformed.iloc[:, 0].apply(lambda x: self.pos_check(x, 'adj'))
        X_transformed['adv_count'] = X_transformed.iloc[:, 0].apply(lambda x: self.pos_check(x, 'adv'))
        X_transformed['pron_count'] = X_transformed.iloc[:, 0].apply(lambda x: self.pos_check(x, 'pron'))
        return X_transformed

class KeywordPresenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keywords, column_name, count_column_name):
        self.keywords = keywords
        self.column_name = column_name
        self.count_column_name = count_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.column_name] = X_transformed.iloc[:, 0].apply(
            lambda x: int(any(re.search(r'\b{}\b'.format(re.escape(word)), x, flags=re.IGNORECASE) for word in self.keywords))
        )
        X_transformed[self.count_column_name] = X_transformed.iloc[:, 0].apply(
            lambda x: sum(1 for word in self.keywords if re.search(r'\b{}\b'.format(re.escape(word)), x, flags=re.IGNORECASE))
        )
        return X_transformed

class BooleanToBinaryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in X_transformed.columns:
            if X_transformed[column].dtype == 'bool':
                X_transformed[column] = X_transformed[column].astype(int)
        return X_transformed

class AdditionalFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['word_count'] = X_transformed.iloc[:, 0].apply(lambda x: len(x.split()))
        X_transformed['letter_count'] = X_transformed.iloc[:, 0].apply(lambda x: len([c for c in x if c.isalpha()]))
        X_transformed['word_density'] = X_transformed['letter_count'] / X_transformed['word_count']
        X_transformed['hashtag_count'] = X_transformed.iloc[:, 0].apply(lambda x: len(re.findall(r'#\w+', x)))
        return X_transformed

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.fitted_ = False

    def fit(self, X, y=None):
        self.scaler.fit(X.select_dtypes(include=['float64', 'int64']))  # Only fit on numeric columns
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            X_transformed = X.copy()
            X_transformed[X.columns] = self.scaler.fit(X.drop(X.columns[0], axis=1))
        X_transformed = X.copy()
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        X_transformed[numeric_columns] = self.scaler.transform(X[numeric_columns])
        return X_transformed

class PositiveNegativeWordsCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, positive_words, negative_words):
        self.positive_words = positive_words
        self.negative_words = negative_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['pos_word'] = X_transformed.iloc[:, 0].apply(lambda x: sum(word in x for word in self.positive_words))
        X_transformed['neg_word'] = X_transformed.iloc[:, 0].apply(lambda x: sum(word in x for word in self.negative_words))
        return X_transformed

#Positive Keywords
pos_word = './utils/positive_keyword.txt'

with open(pos_word, 'r') as file:
    pos_words = [line.strip() for line in file]

#Negative Keywords
neg_word = './utils/negatif_keyword.txt'

with open(neg_word, 'r') as file:
    neg_words = [line.strip() for line in file]


hitam_putih_terms = ['deddy', 'dedy', 'daddy', 'corbuzier', 'hitam', 'hitamputih', 'hitamputiht', 'hitamputihtrans', 'hitamputihtrans7', 'putih', 'trans', 'trans7']
ilc_terms = ['indonesialawyersclub', 'ilc', 'indonesia', 'lawyers', 'club', 'lawyer', 'hukum', 'karni', 'karniilyas', 'ilyas', 'ilctvone']
matanajwa_terms = ['matanajwa', 'mata', 'najwa', 'matanajwametrotv', 'najwashihab', 'shihab', 'tv_matanajwa']
kickandy_terms = ['kickandy', 'kick', 'andy', 'kickandyshow', 'kickandyp', 'kickand', 'kickandymetrotv']

# Create a pipeline with the defined transformers
preprocessing_pipeline = Pipeline([
    ('lowercase', LowerCaseTransformer()),
    ('clean_links', CleanLinksTransformer()),
    ('remove_punctuation_whitespace', RemovePunctuationAndWhitespaceTransformer()),
    ('stopword_removal', StopWordRemoverTransformer()),
    ('additional_features', AdditionalFeatureTransformer()),
    ('pos_tagger', PosTaggerTransformer()),
    ('positive_negative_words_counter', PositiveNegativeWordsCounterTransformer(pos_words, neg_words)),
    ('hitam_putih_keywords', KeywordPresenceTransformer(hitam_putih_terms, 'hitam_putih_term', 'hitam_putih_term_count')),
    ('ilc_keywords', KeywordPresenceTransformer(ilc_terms, 'ilc_terms', 'ilc_terms_count')),
    ('matanajwa_keywords', KeywordPresenceTransformer(matanajwa_terms, 'matanajwa_terms', 'matanajwa_terms_count')),
    ('kickandy_keywords', KeywordPresenceTransformer(kickandy_terms, 'kickandy_terms', 'kickandy_terms_count')),
    ('boolean_to_binary', BooleanToBinaryTransformer()),
    ('min_max_scaler', MinMaxScalerTransformer(feature_range=(0, 1)))
])