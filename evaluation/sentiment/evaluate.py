# cd to the prod-sentiment-package
# and run the following : python3 -m evaluation.predict_and_evaluate

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Read csv file with both labels and prediction and compute evaluation metrics
df_pandas_labelled = pd.read_csv('./evaluation/postprocessed_data.csv')

# Rename columns if necessary
df_pandas_labelled.rename(columns={"True_Sentiment":"True_Sentiment", "Predcited_Sentiment":"Predicted_Sentiment"},inplace=True)

# Compute the evaluation metrics
accuracy = accuracy_score(df_pandas_labelled["True_Sentiment"], df_pandas_labelled["Predicted_Sentiment"])
accuracy *= 100
classification_report = classification_report(df_pandas_labelled["True_Sentiment"], df_pandas_labelled["Predicted_Sentiment"])

print(f"Accuracy: {accuracy}")
print(classification_report)