import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

class AnalysisService:
    def predict(document):
        df = pd.DataFrame(columns=['text','aspek', 'label'])
        new_rows = [{'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'rasa', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'bau', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'bau', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'bau', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'bau', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'negative'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'},
            {'text': 'saya suka makanan ini', 'aspek': 'harga', 'label': 'positive'}]

        result = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(result)
        return result, None
    
    def preprocess(document):
        print(document)
        return document, None
    
    def get_history(user_id):
        return None, None
    
    def create_plot(document):
        length = len(document)
        print(length)
        # Create a pivot table from the original DataFrame
        pivot = document.pivot_table(index='aspek', columns='label', aggfunc='size', fill_value=0)

        # Reset the index
        pivot = pivot.reset_index()

        # Ensure that both 'positive' and 'negative' labels are present for each 'aspek'
        for label in ['positive', 'negative']:
            if label not in pivot.columns:
                pivot[label] = 0

        # Melt the DataFrame to long format
        counts = pd.melt(pivot, id_vars='aspek', value_vars=['positive', 'negative'], var_name='label', value_name='counts')

        # Create a bar chart of the counts with grouped bars and colored based on 'label'
        ax = sns.barplot(x='counts', y='aspek', hue='label', data=counts, palette={'positive': 'green', 'negative': 'red'})

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

        # export document to csv
        document.to_csv('./static/csv/result.csv', index=False)
        return None