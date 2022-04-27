import pandas as pd

df = pd.read_csv('test.csv')

# lab = df[0:51]
# nonlab = df[51:].sample(frac=1)
# df = pd.concat([lab,nonlab]).reset_index(drop=True)
# df.to_csv('test.csv')

print(df['Predicted_Entity_and_Sentiment'][0])
# df = df.drop(df[df.])
#
#
# df.to_csv('posneg.csv')
