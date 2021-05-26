#**********************************************
#*************** Import Libraries *************
#**********************************************
# Basic Packages
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
# NLTK Packages
# import nltk
# nltk.download()
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#**********************************************
#*************** Load Files *******************
#**********************************************
model = joblib.load('models/my_best_model.pkl')
scaler = joblib.load('models/my_best_scaler.pkl')
tfidf = joblib.load('models/my_best_tfidf.pkl')
train_df = pd.read_csv('train_medians.csv')

#**********************************************
#************* Declare Variables **************
#**********************************************
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
#Adds stuff to our stop words list
stop_words.extend(['.',','])


genres = ['action','adventure','animation','biography',
                'comedy','crime','documentary',
                'drama','family','fantasy','film-noir',
                'history','horror','music','musical',
                'mystery','romance','sci-fi','sport',
                'thriller','war','western']

features = ['f_release_year','f_release_month','f_runtime','f_word_count_long','f_imdb_rating','f_num_imdb_votes','f_num_user_reviews','f_num_critic_reviews']

#**********************************************
#**************** Functions *******************
#**********************************************

# Remove Stopwords
def remove_stop(list_of_tokens):

    without_stop = []

    for token in list_of_tokens:
        if token in stop_words: continue
        without_stop.append(token)

    return without_stop

# Stemmer
def stemmer(list_of_tokens):
    '''
    Takes in an input which is a list of tokens, and spits out a list of stemmed tokens.
    '''

    stemmed_tokens_list = []

    for i in list_of_tokens:

        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)

    return stemmed_tokens_list

# Lemmatizer
def lemmatizer(list_of_tokens):

    lemmatized_list = []

    for i in list_of_tokens:
        token = WordNetLemmatizer().lemmatize(i)
        lemmatized_list.append(token)

    return lemmatized_list

# Untokenzier
def untokenizer(token_list):
        '''
        Returns all the tokenized words in the list to one string.
        Used after the pre processing, such as removing stopwords, and lemmatizing.
        '''
        return " ".join(token_list)

# Plot Cleaner
def clean_plot(input_plot):
    tokenized_list = word_tokenize(input_plot)
    removed_stop = remove_stop(tokenized_list)
    stemmed = stemmer(removed_stop)
    lemma = lemmatizer(stemmed)
    return_string = untokenizer(lemma)
    return return_string

#**********************************************
#****************** Flask *********************
#**********************************************
app = Flask(__name__)
app.secret_key = "key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.sqlite3'

#making database
db = SQLAlchemy(app)

class database(db.Model):
      _id = db.Column("id", db.Integer, primary_key=True)
      input_plot=db.Column(db.String)
      results=db.Column(db.String)

      def __init__(self, input_plot, results):
            self.input_plot = input_plot
            self.results = results

#**********************************************
#***************** APP-ROUTES *****************
#**********************************************

#History
@app.route('/history.htm')
def history():
      return render_template("history.htm", values=database.query.all()) 

@app.route('/', methods=["GET","POST"])
def genre_predictor():

    try:
        if request.method == "POST":
            input_plot = request.form['plot']
          
            feature_cols_df = pd.DataFrame([[0]*8 ], columns=features)

            input_for_tfidf = tfidf.transform([clean_plot(input_plot)])
            tfidf_transformed_df = pd.DataFrame(input_for_tfidf.toarray(), columns=tfidf.get_feature_names())

            ready_to_pred = pd.concat([feature_cols_df, tfidf_transformed_df], axis=1)

            for col in features:
                ready_to_pred.at[0,col] = train_df[col].median()
            ready_to_pred.at[0,'f_word_count_long'] = len(input_plot)
            ready_to_pred_final_df = scaler.transform(ready_to_pred)

            pred_genre = model.predict_proba(ready_to_pred_final_df)

            df = pd.DataFrame(pred_genre, columns=genres).T.sort_values(0, ascending=False)
            predicted_list = []
            for index, row in df.iterrows():
                if row.values[0] >= 0.2:
                    temp_list = [int(round(row.values[0]*100,0)), index.capitalize()]
                    predicted_list.append(temp_list)
            print(predicted_list)
            results = "genre"
            db.create_all()
            usr= database(input_plot,results)
            db.session.add(usr)
            db.session.commit()
            return render_template('home.html', predictions=predicted_list, input_plot=input_plot)
        else:
            return render_template('home.html')

    except Exception as e:
        print(e)
        return render_template("home.html", error = e)

if __name__ == "__main__":
    app.debug = True
    db.create_all()
    app.run(host='127.0.0.1', port=8080, debug=True)