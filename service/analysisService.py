from flask_login import current_user
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocessing import preprocessing_pipeline
from database import db
from model.historyModel import History
import utils.classifier as classifier
import joblib
import nltk

class AnalysisService:
    def classification(document):
        document['Predicted Sentiment'] = document.apply(classifier.predict_sentiment, axis=1)
        sentiment_mapping = {
            0 : 'Negative',
            1 : 'Positive'
        }
        document['Sentiment Name'] = document['Predicted Sentiment'].map(sentiment_mapping)
        return document, None
    
    def clustering(document):
        X = document.drop(['Text Tweet','pos_word', 'neg_word', 'hitam_putih_term_count', 'ilc_terms_count', 'matanajwa_terms_count', 'kickandy_terms_count', 'word_count', 'letter_count', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count', 'hashtag_count'], axis=1)
        cluster_mapping = {
            0 : 'IndonesiaLawyersClubTvOne',
            2 : 'HitamPutihTransTV',
            1 : 'KickAndyMetroTV',
            3 : 'MataNajwaMetroTV'
        }
        # Load the saved model
        loaded_kmeans = joblib.load('./model/machine/clustering_silhouette086.pkl')

        # Use the loaded model to predict clusters for the new data
        new_data_clusters = loaded_kmeans.predict(X)

        # Add the predicted clusters to the DataFrame
        document['Predicted Cluster'] = new_data_clusters

        document['Cluster Name'] = document['Predicted Cluster'].map(cluster_mapping)

        return document, None
    
    def preprocess(document):
        df_preprocessed = preprocessing_pipeline.transform(document)
        return df_preprocessed, None
    
    def get_history():
        try:
            # get data by user id and order by created at descending
            history = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).all()
            # print data history
            return history, None
        except Exception as e:
            return None, e
    
    def create_plot(document):
        # Create a pivot table from the original DataFrame
        pivot = document.pivot_table(index='Cluster Name', columns='Sentiment Name', aggfunc='size', fill_value=0)

        # Reset the index
        pivot = pivot.reset_index()

        # Ensure that both 'positive' and 'negative' labels are present for each 'aspek'
        for label in ['Positive', 'Negative']:
            if label not in pivot.columns:
                pivot[label] = 0

        # Melt the DataFrame to long format
        counts = pd.melt(pivot, id_vars='Cluster Name', value_vars=['Positive', 'Negative'], var_name='Sentiment Name', value_name='counts')

        # Create a bar chart of the counts with grouped bars and colored based on 'label'
        ax = sns.barplot(x='counts', y='Cluster Name', hue='Sentiment Name', data=counts, palette={'Positive': 'green', 'Negative': 'red'})

        # Set a minimum length for the x-axis to 10
        ax.set_xlim(0, max(10, counts['counts'].max() + 5))

        # Add a grid
        plt.grid(True)

        # Calculate the total count
        total = counts['counts'].sum()

        # Add the percentage of each bar to the total data on top of each bar
        for i, p in enumerate(ax.patches):
            if i == len(ax.patches) - 2:
                break
            width = p.get_width()
            ax.text(width + 0.2, p.get_y()+p.get_height()/2.,'{:1.1f}'.format(width/total*100) + '%', va="center") 

        # Set the label for the x-axis
        plt.xlabel('Counts')

        # Set the label for the y-axis
        plt.ylabel('Aspek')

        # Set the title of the plot
        plt.title('Counts of Aspek and Label')

        # Save the plot to a file
        plt.savefig('./static/img/plot.png')

        # Clear the plot
        plt.clf()

        export = document[['Text Tweet', 'Predicted Cluster', 'Cluster Name', 'Predicted Sentiment', 'Sentiment Name']]
        # export document to csv
        export.to_csv('./static/csv/result.csv', index=False)

        return None
    

    def plot(document):
        import matplotlib.pyplot as plt

        aspects = [0, 1, 2, 3]
        history = ['', '', '', '']
        aspects_name = ['ILC', 'Kick Andy', 'Hitam Putih', 'Mata Najwa'] 
        positive_sentiments = [document[document['Predicted Sentiment'] == 1].groupby('Predicted Cluster').size().get(aspect, 0) for aspect in aspects]
        negative_sentiments = [document[document['Predicted Sentiment'] == 0].groupby('Predicted Cluster').size().get(aspect, 0) for aspect in aspects]

        total_per_aspect = [p + n for p, n in zip(positive_sentiments, negative_sentiments)]

        bar_width = 0.7
        index = range(len(aspects))

        fig, ax = plt.subplots(figsize=(12,6))
        bar1 = ax.bar(index, positive_sentiments, bar_width, label='Positive', color='skyblue')
        bar2 = ax.bar(index, negative_sentiments, bar_width, label='Negative', color='salmon', bottom=positive_sentiments)

        ax.set_xlabel('Aspects')
        ax.set_ylabel('Sentiment Count')
        ax.set_title('Aspect-based Sentiment Analysis Results')
        ax.set_xticks([i for i in index])
        ax.set_xticklabels(aspects_name)
        
        # Add the legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        #  total data
        total_data = len(document)

        # Add predicted labels
        for i, (p, n, total) in enumerate(zip(positive_sentiments, negative_sentiments, total_per_aspect)):
            percent_positive = p / total_data * 100 if total != 0 else 0
            percent_negative = n / total_data * 100 if total != 0 else 0
            ax.text(i, p / 2, f'{percent_positive:.1f}%', ha='center', va='center', color='black', fontweight='bold')
            ax.text(i, p + n / 2, f'{percent_negative:.1f}%', ha='center', va='center', color='black', fontweight='bold')
            history[i] = f'{percent_positive:.1f}% Positif, {percent_negative:.1f}% Negatif'

        print(history)
        # Save the plot to a file
        plt.savefig('./static/img/plot.png')

        # Clear the plot
        plt.clf()

        export = document[['Text Tweet', 'Predicted Cluster', 'Cluster Name', 'Predicted Sentiment', 'Sentiment Name']]
        # export document to csv
        export.to_csv('./static/csv/result.csv', index=False)

        # insert history to database
        history = History(user_id=current_user.id, data=total_data, aspek_0=history[0], aspek_1=history[1], aspek_2=history[2], aspek_3=history[3])
        db.session.add(history)
        db.session.commit()

        return None