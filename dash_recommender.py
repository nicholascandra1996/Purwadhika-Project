import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
import dash_table
from dash.dependencies import Input, Output, State

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

movies = pd.read_csv('IMDb movies.csv')
movies = movies[(movies['year']>1970)&(movies['votes']>10000)]
movies.reset_index(drop=True,inplace=True)
movies2 = movies.copy()
movies.drop(['imdb_title_id','original_title','date_published','language','budget','usa_gross_income','worlwide_gross_income','metascore','reviews_from_users','reviews_from_critics'],axis=1,inplace=True)

movies['country'] = movies['country'].apply(lambda x: 'Unknown' if pd.isna(x) else x)
movies['production_company'] = movies['production_company'].apply(lambda x: 'Unknown' if pd.isna(x) else x)
movies['director'] = movies['director'].apply(lambda x: 'Anonymous' if pd.isna(x) else x)
movies['writer'] = movies['writer'].apply(lambda x: 'Anonymous' if pd.isna(x) else x)
movies['actors'] = movies['actors'].apply(lambda x: 'Anonymous' if pd.isna(x) else x)
movies['description'] = movies['description'].apply(lambda x: '' if pd.isna(x) else x)

movies['title'] = movies['title'].apply(lambda x: str(x).lower())
movies2['actors'] = movies2['actors'].apply(lambda x: 'Anonymous' if pd.isna(x) else x)

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
keywords = []
for i in range(len(movies)):
    r = Rake()
    r.extract_keywords_from_text(movies['description'][i])
    key_words_dict_scores = r.get_word_degrees()
    keywords.append(list(key_words_dict_scores.keys()))
movies['keywords'] = keywords

def new_title(cols):
    return str(cols['title']) + ' (' + str(cols['year']) + ')'
movies['new_title'] = movies[['title','year']].apply(new_title,axis=1)

cat = ['genre','country','director','writer','production_company','actors']
for i in cat:
    movies[i] = movies[i].apply(lambda x: x.lower())
    movies[i] = movies[i].apply(lambda x: x.replace(' ',''))
    movies[i] = movies[i].apply(lambda x: x.split(','))
movies['bow_title'] = movies['title'].apply(lambda x: x.lower())
movies['bow_title'] = movies['bow_title'].apply(lambda x: x.replace(':',''))
movies['bow_title'] = movies['bow_title'].apply(lambda x: x.split(' '))

def same_actor(title):
    title = str(title).lower()
    mov = movies[movies['title'] == title].copy()
    new_mov = mov['new_title'].values[0]
    top_actor = mov['actors'].iloc[0][0]
    
    other = []
    for i in range(len(movies)):
        if top_actor in movies['actors'][i]:
            other.append(i)
    
    same = movies.iloc[other,:]
    
    min_votes = same['votes'].mean()
    a = same[same['new_title'] == new_mov].index
    
    same = same.drop(a)
    same = same[same['votes'] >= min_votes]
    same = same.sort_values('avg_vote',ascending=False)
    
    return list(same[:10].index)

def top_actor(title):
    title = str(title).lower()
    idx = movies[movies['title'] == title].index
    mov2 = movies2.iloc[idx,:]
    mov2['actors'] = mov2['actors'].apply(lambda x: x.split(', '))
    top = mov2['actors'].values[0][0]
    return top

bow1 = movies.set_index('title')
bow1 = bow1[['genre','keywords','bow_title']]
def bag1(cols):
    return '' + ' '.join(cols['genre']) + ' ' + ' '.join(cols['keywords']) + ' ' + ' '.join(cols['bow_title'])
bow1['bag_of_words'] = bow1[['genre','keywords','bow_title']].apply(bag1,axis=1)
bow1 = bow1.drop(['genre','keywords','bow_title'],axis=1)

bow2 = movies.set_index('title')
bow2 = bow2[['country','director','writer','production_company','actors']]
def bag2(cols):
    return '' + ' '.join(cols['country']) + ' ' + ' '.join(cols['director']) + ' ' + ' '.join(cols['writer']) + ' ' + ' '.join(cols['production_company']) + ' ' + ' '.join(cols['actors'])
bow2['bag_of_words'] = bow2[['country','director','writer','production_company','actors']].apply(bag2,axis=1)
bow2 = bow2.drop(['country','director','writer','production_company','actors'],axis=1)

count = CountVectorizer()
count_matrix1 = count.fit_transform(bow1['bag_of_words'])
count_matrix2 = count.fit_transform(bow2['bag_of_words'])

cosine_sim1 = cosine_similarity(count_matrix1, count_matrix1)
cosine_sim2 = cosine_similarity(count_matrix2, count_matrix2)

indices = pd.Series(movies['title'])

def similar(title, cosine_sim1 = cosine_sim1):
    title = str(title).lower()
    
    idx = indices[indices == title].index[0]

    score_series = pd.Series(cosine_sim1[idx]).sort_values(ascending = False)
    ss = score_series.copy()
    min_score = ss[(ss.values<0.99)&(ss.values>0.01)].describe()['75%']
    score_series = score_series[score_series.values > min_score]
    sim_idx = list(score_series.index)
    
    sim = movies.iloc[sim_idx,:]
    sim = sim.drop(idx)
    
    country = movies.iloc[idx,:]['country'][0]
    for i in sim.index:
        if country not in sim['country'][i]:
            sim.drop(i,inplace=True)
    
    min_votes = movies.iloc[idx,:]['votes']/5
    min_avg = movies.iloc[idx,:]['avg_vote']-2
    sim = sim[(sim['votes']>=min_votes)&(sim['avg_vote']>=min_avg)]
    
    return list(sim[:10].index)

def parameter1(title, cosine_sim1 = cosine_sim1):
    title = str(title).lower()
    
    idx = indices[indices == title].index[0]

    score_series1 = pd.Series(cosine_sim1[idx]).sort_values(ascending = False)
    s1 = score_series1.copy()
    min_score = s1[(s1.values<0.99)&(s1.values>0.01)].describe()['75%']
    score_series1 = score_series1[score_series1.values > min_score]
    sim_idx = list(score_series1.index)
    
    sim = movies.iloc[sim_idx,:]
    sim = sim.drop(idx)
    
    return sim.index

def parameter2(title, cosine_sim2 = cosine_sim2):
    title = str(title).lower()
    
    idx = indices[indices == title].index[0]

    score_series2 = pd.Series(cosine_sim2[idx]).sort_values(ascending = False)
    s2 = score_series2.copy()
    min_score = s2[(s2.values<0.99)&(s2.values>0.01)].describe()['75%']
    score_series2 = score_series2[score_series2.values > min_score]
    sim_idx = list(score_series2.index)
    
    sim = movies.iloc[sim_idx,:]
    sim = sim.drop(idx)
    
    return sim.index

def recommendations(title):
    a = parameter1(title)
    b = parameter2(title)
    c = set(a).intersection(b)
    c = list(c)
    d = movies.iloc[c,:]
    d = d.sort_values('avg_vote',ascending=False)
    return list(d[:10].index)

rec_col = [1,3,5,6,7,9,12,13,14]

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1("Movie Recommendation Project"),
    html.Div(children=["""
        By: Nicholas Candra.
        Data Science Job Connector JKT Batch 7
        """]),
    html.Br(),
    html.Div(children=[
        html.P('Enter Movie: '),
        dcc.Input(id='movies_input', value='John Wick',type='text')
    ]),
    html.Div([html.Button('Recommend!', id='movies_wanted')]),
    html.Br(),
        dcc.Tabs(children = [
            dcc.Tab(value= 'Tab1', label= 'Top 10 Recommendation', children= [
            html.Div(id='themovie1', children=generate_table(movies2.iloc[recommendations('John Wick'),rec_col]))
                ],className='col-4'),
            dcc.Tab(value= 'Tab2', label= 'Similar Movies', children= [
            html.Div(id='themovie2', children=generate_table(movies2.iloc[similar('John Wick'),rec_col]))
                ],className='col-4'),
            dcc.Tab(value= 'Tab3', label= "Similar Actor", children= [
            html.Div(id='theactor', children = 'Movies with {}:'.format(top_actor('John Wick'))),
            html.Br(),
            html.Div(id='themovie3', children=generate_table(movies2.iloc[same_actor('John Wick'),rec_col]))
                ],className='col-4')
    ],
    content_style= {
        'font_family':'Arial',
        'borderBottom':'1px solid #d6d6d6',
        'borderLeft':'1px solid #d6d6d6',
        'borderRight':'1px solid #d6d6d6',
        'padding':'44px'
    })
],
style={
    'maxWidth':'1200px',
    'margin': '0 auto'
})

@app.callback(
    [Output(component_id = 'themovie1', component_property = 'children'),
    Output(component_id = 'theactor', component_property = 'children'),
    Output(component_id = 'themovie2', component_property = 'children'),
    Output(component_id = 'themovie3', component_property = 'children')],
    [Input(component_id = 'movies_wanted', component_property = 'n_clicks')],
    [State(component_id = 'movies_input', component_property = 'value')]
)

def input_movie(n_clicks, title):
    return [generate_table(movies2.iloc[recommendations(title),rec_col]),
    'Movies with {}:'.format(top_actor(title)),
    generate_table(movies2.iloc[similar(title),rec_col]),
    generate_table(movies2.iloc[same_actor(title),rec_col])]

if __name__ == '__main__':
    app.run_server(debug=True)