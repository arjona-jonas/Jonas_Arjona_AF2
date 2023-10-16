import pandas as pd
import matplotlib.pyplot as plt

# Escolher uma base de dados da sua preferência, através de sites como Kaggle, UCI, etc.
# Escolhi uma base das músicas mais tocadas no Spotify em 2023 
# link: https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023

# utf-8 nao funciona como encoding, usamos ISO-8859-1
spotify_data=pd.read_csv('spotify-2023.csv',encoding='ISO-8859-1')

# Primeira olhada nos dados
print('\nPrimeiras entradas\n',spotify_data.iloc[0:5,0:5],'\n')

# O objetivo é fazer uma análise das músicas presentes na lista de mais ouvidas em 2023 (até agora)
# Para isso, focamos principalmente em fazer agrupamentos com os anos para número de streams ("streams"), número de
# playlists do spotify que aquela música aparece ("in_spotify_playlists") e número de "charts" do spotify que essas
# musicas aparecem ("in_spotify_charts")

# .info já mostra que não há nulos na nossa base
# Temos 953 músicas e 24 variáveis/colunas
print('\nUsando .info')
spotify_data.info()

# Há colunas que não nos interessam aqui (da 9 até a 23)
# Dropamos elas
cols=spotify_data.loc[:,'in_apple_playlists':].columns

spotify_data.drop(columns=cols,axis=1,inplace=True)

print('\nNova lista de colunas\n',spotify_data.columns)

# Além disso, vimos que 'streams' é lido como 'object'
# Convertemos ela e usamos o argumento 'error' para transformar os erros em NaN
# que serão removidos em seguida.
spotify_data['streams']=pd.to_numeric(spotify_data['streams'],errors='coerce')
spotify_data.dropna(axis=0,inplace=True)

print('\nChecando mudancas')
spotify_data.info()

# Renomeamos algumas das colunas para reduzir tamanhos e deixar mais evidente o conteúdo
new_cols_names={
    'artist(s)_name':'artist_name',
    'artist_count':'num_of_artists',
    'released_day':'release_day',
    'released_month':'release_month',
    'released_year':'release_year',
    'in_spotify_playlists':'num_spotify_pl',
    'in_spotify_charts':'num_spotify_charts'}

spotify_data.rename(columns=new_cols_names,inplace=True)

print('\nNovos nomes\n',spotify_data.columns)

# Para melhorar a visualização da informação, criamos uma nova coluna de 'streams' em milhões
spotify_data['streams_em_1kk']=spotify_data['streams']/10**6

# O top 10 músicas mais ouvidas
top_10_tracks=spotify_data[['track_name','artist_name','release_year','streams_em_1kk']].copy()

print('\nQual o top 10 músicas mais tocadas em 2023 (até agora)?\nps: ajustar largura da tela pra ver os resultados')
print(top_10_tracks.sort_values('streams_em_1kk',ascending=False).head(n=10))
print('\nNenhuma música de 2021, 2022 e 2023! Por que será?\n')

# Quantas músicas de cada ano temos na base?
musicas_por_ano=spotify_data.groupby('release_year')['track_name'].count()

# Usamos head para limitar o tamanho da saída
print('Quantas músicas temos por ano?\n',musicas_por_ano.sort_values(ascending=False))

# Finalmente agrupamos por ano e visualizamos algumas estatísticas (soma,contagem,minimo,media,maximo)
# Filtramos também a partir de 2010
resumo_estat=spotify_data.loc[:,['release_year','streams_em_1kk']]

filtro=resumo_estat['release_year']>=2010

resumo_estat=resumo_estat[filtro]

resumo_estat=resumo_estat.groupby(['release_year'])['streams_em_1kk'].agg(['sum','count','min','mean','max']).reset_index()

print('\nEstatísticas sobre streams (por milhão) para cada ano\n',
      resumo_estat.sort_values(by='count',ascending=False).head(n=15))
print('\nApesar dos anos 2020+ liderarem o número de músicas e a soma total de streams, suas estatísticas (em especial o mínimo e a média) são inferiores a dos anos 2019 pra baixo.')
print('\nPor outro lado, vemos que os anos de 2019 pra baixo têm médias maiores que 2020+. Logo, as músicas que persistem muito ouvidas são justamente os grandes hits daqueles anos, enquanto que os anos mais recentes estão cheios de ruído do momento (como lançamentos e afins).')
print('\nPor fim, é de se imaginar que músicas mais antigas tenham as maiores médias, já que elas estão há mais tempo disponíves.')

# Vamos exportar essa tabela agrupada que foi criada
print('\nFizemos o export para csv dessa última tabela\n')
resumo_estat.to_csv('resumo_estat_spotify_ano.csv')

# Partimos então para análises sobre as "playlists" e "charts"

# Usamos pd.qcut para categorizarmos as musicas em 4 quantis de playlist
# Dessa forma podemos saber se as músicas estão em muitas playlists ou não

spotify_data['pl_quantil']=pd.qcut(spotify_data['num_spotify_pl'],q=4,
                                   labels=['Below_25%','25%_to_50%','50%_to_75%','Top_75%'])

top_75_pl=spotify_data[spotify_data['pl_quantil']=='Top_75%']

top_75_pl=top_75_pl.groupby('release_year')['track_name'].count().reset_index()
top_75_pl.rename(columns={'track_name':'num_of_tracks'},inplace=True)
top_75_pl['prop_of_tracks']=(top_75_pl['num_of_tracks']/top_75_pl['num_of_tracks'].sum())*100


#Como usamos pd.qcut, a distribuição é de 25% pra cada categoria da variável criada

print('\nAgrupamos as músicas pelo número de playlists que elas aparecem em 4 quartis. Usando pd.qcut a distribuição é de 25% dos dados pra cada categoria')
print('\nQual a distribuição dos anos dentro da categoria "Top_75%"?')
print(top_75_pl.sort_values('num_of_tracks',ascending=False))
print('\nAs músicas dos anos 2020+ compõe mais o conjunto "Top_75%" (cerca de 1/4 das entradas), ou seja, estão em mais playlists do que as dos anos 2020 abaixo. Estranhamente 2023 não aparece nesse agrupamento, indicando que apesar de ter muitas músicas lançadas, a maioria delas não faz parte de muitas playlists.')

# Para charts, usamos a média como valor diferenciador e categorizamos as musicas
# com uma função auxiliar aplicada com .apply
avg_charts=spotify_data['num_spotify_charts'].mean()

def track_chart_classify(row,threshold):
    if row['num_spotify_charts']<threshold:
        return 'Below average'
    else:
        return 'Above average'

# Charts são playlists criadas pelo próprio Spotify compostas pelas músicas mais ouvidas
# de um determinado país
spotify_data['ch_category']=spotify_data.apply(track_chart_classify,threshold=avg_charts,axis=1)

above_avg_ch=spotify_data[spotify_data['ch_category']=='Above average']

above_avg_ch=above_avg_ch.groupby('release_year')['track_name'].count().reset_index()
above_avg_ch.rename(columns={'track_name':'num_of_tracks'},inplace=True)

print('\nAgrupamos as músicas pelo número de charts que elas aparecem em duas categorias: mais que a média (12 charts) ou menos que a média')
print('\nQual a distribuição dessa variável?')
print(spotify_data['ch_category'].value_counts())
print('Cerca de 1/3 das músicas estão em mais charts do que a média!')

print('\nQual a distribuição dos anos dentro da categoria "Above average"?')
print(above_avg_ch.sort_values('num_of_tracks',ascending=False))
print('\nAs músicas dos anos 2020+ compõe mais o conjunto "Above average" (cerca de 60%), ou seja, estão em mais charts do que as dos anos 2020 abaixo. Por que será que 2023 aparece aqui mas nao na outra visualização?')
# Gráfico: número de músicas em "playlists" do Spotify por ano
df_plot=spotify_data[(spotify_data['release_year']>=2010)&(spotify_data['pl_quantil']=='Top_75%')]

df_plot=df_plot.groupby('release_year')['track_name'].count().reset_index().sort_values('track_name')

plot_f=plt.barh(y=df_plot['release_year'],
                width=df_plot['track_name'])

plt.xlabel('Número de músicas')

plt.ylabel('Ano')
plt.yticks(df_plot['release_year'].unique())

plt.title('Músicas em muitas playlists ("Top_75%") do Spotify por ano (2010-2023)')
plt.show()

# Gráfico: número de músicas em "charts" do Spotify por ano
df_plot=spotify_data[(spotify_data['release_year']>=2010)&(spotify_data['ch_category']=='Above average')]

df_plot=df_plot.groupby('release_year')['track_name'].count().reset_index().sort_values('track_name')

plot_f=plt.barh(y=df_plot['release_year'],
                width=df_plot['track_name'])

plt.xlabel('Número de músicas')

plt.ylabel('Ano')
plt.yticks(df_plot['release_year'].unique())

plt.title('Músicas em muitas charts ("Above average") do Spotify por ano (2010-2023)')
plt.show()

# Acho que o segundo gráfico não vai aparecer no replit. Caso queria ver o resultado, marque o primeiro como comentários e rode o código de novo. Mas é basicamente uma vizz da tabela já feita antes.