import pandas as pd


def _initialise_newdfdict():
    newdfdict = {}
    newdfdict['article_index'] = []
    newdfdict['sentiment_max'] = []
    newdfdict['entity_name'] = []
    newdfdict['entity_type'] = []
    return newdfdict


def _reorganise_column(df, newdfdict, column_name, entity_type):
    for i, entity in enumerate(df[column_name]):
        if not pd.isnull(entity):
            newdfdict['article_index'].append(i)
            newdfdict['sentiment_max'].append(int(df.at[i, 'sentiment (Maxs take)']))
            newdfdict['entity_name'].append(entity)
            newdfdict['entity_type'].append(entity_type)
    return newdfdict


if __name__ == '__main__':
    df = pd.read_csv('labeled_entity_sent.csv')

    newdfdict = _initialise_newdfdict()

    for org_col in ['organisation 1', 'organisation 2', 'organisation 3']:
        newdfdict = _reorganise_column(df, newdfdict, org_col, 'ORG')

    for person_col in ['person 1', 'person 2', 'person 3']:
        newdfdict = _reorganise_column(df, newdfdict, person_col, 'PER')

    for loc_col in ['location 1', 'location 2', 'location 3']:
        newdfdict = _reorganise_column(df, newdfdict, loc_col, 'LOC')

    newdf = pd.DataFrame.from_dict(newdfdict,orient='index') \
                        .transpose() \
                        .sort_values('article_index') \
                        .reset_index(drop=True)

    newdf.to_csv('entity_labeled.csv')
