import pandas as pd
import json
from time import time
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Import API credentials from JSON file
with open('\data\credentials.json') as creds:
    credentials = json.load(creds)

sample_df = pd.read_csv('\data\sample_1k.csv')

def analyse_sentiment_quickstart(credentials, sample_df, model_name='Original tweets'):
    
    # Set API variables
    subscription_key = credentials['api_key']
    endpoint = credentials['endpoint']

    # Create new client
    client = (
        TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(subscription_key)))
    
    # Create function to automate sentiment prediction
    def sentiment_analysis(client, doc):
        documents = [doc]
        response = client.analyze_sentiment(documents=documents)[0]
        for idx, sentence in enumerate(response.sentences):
            return sentence.sentiment
    
    # Create a dataframe to save results
    data = pd.DataFrame(columns=['text', 'label', 'prediction'])

    # Process prediction
    start = time()
    for index, row in sample_df.iterrows():
        data = data.append(
            {
                'text':row['tweet'],
                'label':row['sentiment'],
                'prediction':sentiment_analysis(client, row['tweet'])
            },
            ignore_index=True
        )
    pred_time = time()-start

    # Exclude "neutral" predictions
    data = data[data.prediction != 'neutral']

    # Replace value of predictions columns
    data['prediction'] = data['prediction'].replace(['negative', 'positive'],[0,1])

    print('Nb rows: ', len(data))
    print('Proportion vs Total sample: {}%'.format(len(data)/len(sample_df)*100))

    # Prepare data for evaluation
    y_test = data.label.astype(int)
    y_pred = data.prediction

    # Compute AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    # Compute accuracy: (tp + tn) / (p + n)
    acc = accuracy_score(y_test, y_pred)

    # Create a list to store scores/results
    scores_model = []
    
    # Append scores/results
    scores_model.append(
        {'Model': model_name,
        'Predict_time':'{:0.1f}'.format(pred_time),
        'AUC_Score':'{:0.3f}%'.format(auc_score*100),
        'Accuracy':'{:0.3f}%'.format(acc*100)})
        
    # Save in DF
    model_results = pd.DataFrame.from_records(scores_model)
    
    # Create confusion matrix table
    cm = pd.crosstab(
        index=y_test,
        columns=y_pred,
        values=y_test,
        aggfunc=lambda x:len(x), normalize='index').mul(100)

    # Create classification report
    cr = classification_report(y_test, y_pred)
    print(cr)

    # Plot confusion matrix
    fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    ax = sns.heatmap(cm, annot=True, fmt='.1f', cbar=False, cmap='Blues')
    
    for t in ax.texts: t.set_text(t.get_text() + " %")
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    
    plt.ylabel('True labels', fontweight='bold')
    plt.xlabel('Predicted labels', fontweight='bold')
    
    title = 'Confusion matrix - Original tweets'
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return model_results


# Save to CSV file
model_results = analyse_sentiment_quickstart(credentials, sample_df, model_name='Original tweets')

model_results.to_csv('\data\model_results.csv')