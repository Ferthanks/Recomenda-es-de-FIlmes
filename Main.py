import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. CARREGAMENTO E POPULARIDADE ---
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Cálculo do Weighted Rating
stats_votos = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
stats_votos.columns = ['R', 'v']
C = stats_votos['R'].mean()
m = stats_votos['v'].quantile(0.90)

def calculate_wr(x):
    v, R = x['v'], x['R']
    return (v/(v+m) * R) + (m/(m+v) * C)

stats_votos['wr_score'] = stats_votos.apply(calculate_wr, axis=1)
max_wr, min_wr = stats_votos['wr_score'].max(), stats_votos['wr_score'].min()
stats_votos['wr_norm'] = (stats_votos['wr_score'] - min_wr) / (max_wr - min_wr)
popularidade_vetor = movies['movieId'].map(stats_votos['wr_norm']).fillna(0).values

# --- 2. PROCESSAMENTO DE CONTEÚDO (TF-IDF) ---

movies['genres_space'] = movies['genres'].str.replace('|', ' ', regex=False)
tfidf = TfidfVectorizer(stop_words='english')
genres_matrix = tfidf.fit_transform(movies['genres_space'])
sim_conteudo = cosine_similarity(genres_matrix)

# --- 3. PROCESSAMENTO COLABORATIVO COM NORMALIZAÇÃO ---
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_means = user_movie_matrix.mean(axis=1)
user_movie_norm = user_movie_matrix.sub(user_means, axis=0).fillna(0)
sim_colaborativa_raw = cosine_similarity(user_movie_norm.T)

# Alinhamento
num_filmes = len(movies)
sim_colab_alinhada = np.zeros((num_filmes, num_filmes))
movie_id_to_idx_original = {id: i for i, id in enumerate(movies['movieId'])}
indices_alinhados = [movie_id_to_idx_original[m_id] for m_id in user_movie_matrix.columns]

ix1, ix2 = np.meshgrid(indices_alinhados, indices_alinhados, indexing='ij')
sim_colab_alinhada[ix1, ix2] = sim_colaborativa_raw

# Função para extrair o radical do nome (ajuda a identificar franquias)
def extrair_radical(titulo):
    # Remove o ano (ex: " (1995)") e caracteres especiais
    titulo_limpo = re.sub(r'\s*\(\d{4}\)', '', titulo)
    # Pega as duas primeiras palavras para identificar franquias (ex: "Star Wars", "Toy Story")
    palavras = titulo_limpo.split()
    if len(palavras) >= 2:
        return " ".join(palavras[:2]).lower()
    return titulo_limpo.lower()

def aplicar_mmr_com_franquia(idx_busca, indices_candidatos, matriz_sim, top_n=5, lambda_param=0.5):
    escolhidos = []
    # Guarda o radical do filme que o usuário digitou para não sugerir cópias dele
    radical_original = extrair_radical(movies.iloc[idx_busca]['title'])
    radicais_na_lista = [radical_original] 
    
    # Filtra os top 30 candidatos para re-ranquear
    candidatos = list(indices_candidatos[:30])
    
    while len(escolhidos) < top_n and candidatos:
        melhor_mmr = -np.inf
        proximo_selecionado = None
        
        for cand in candidatos:
            radical_cand = extrair_radical(movies.iloc[cand]['title'])
            
            # Se o radical já estiver na lista, ignoramos este candidato
            if radical_cand in radicais_na_lista:
                continue
                
            relevancia = matriz_sim[idx_busca, cand]
            diversidade = max([matriz_sim[cand, esc] for esc in escolhidos]) if escolhidos else 0
            
            score_mmr = lambda_param * relevancia - (1 - lambda_param) * diversidade
            
            if score_mmr > melhor_mmr:
                melhor_mmr = score_mmr
                proximo_selecionado = cand
        
        if proximo_selecionado is None:
            # Se não houver mais filmes de franquias diferentes, pegamos o melhor
            break
            
        escolhidos.append(proximo_selecionado)
        radicais_na_lista.append(extrair_radical(movies.iloc[proximo_selecionado]['title']))
        candidatos.remove(proximo_selecionado)
        
    return escolhidos

# --- 4. FUNÇÃO HÍBRIDA ---

def recomendar_super_hibrido(user_id, titulo_filme, p_conteudo=0.4, p_colab=0.4, p_pop=0.2, lambda_mmr=0.6):
    try:
        idx_filme = movies[movies['title'] == titulo_filme].index[0]
    except IndexError:
        return "Filme não encontrado."

    score_cont = sim_conteudo[idx_filme]
    score_colab = sim_colab_alinhada[idx_filme]
    score_pop = popularidade_vetor 

    if user_id not in ratings['userId'].values:
        p_conteudo += p_colab
        p_colab = 0

    final_scores = (score_cont * p_conteudo) + (score_colab * p_colab) + (score_pop * p_pop)
    
    indices_top = np.argsort(final_scores)[::-1]
    indices_top = [i for i in indices_top if i != idx_filme]
    
    # Chamada da nova função com filtro de franquia
    recomendados = aplicar_mmr_com_franquia(idx_filme, indices_top, sim_conteudo, top_n=5, lambda_param=lambda_mmr)

    print(f"\nRecomendações para '{titulo_filme}':")
    for i in recomendados:
        print(f"- {movies.iloc[i]['title']} | Gêneros: {movies.iloc[i]['genres']}")

# --- INTERFACE ---
def interface_usuario():
    print("\n" + "="*50)
    print("SISTEMA DE RECOMENDAÇÃO HÍBRIDO v2.0")
    print("="*50)
    while True:
        entrada = input("\nDigite o nome de um filme (ou 'sair'): ").strip()
        if entrada.lower() == 'sair': break
        
        filmes_encontrados = movies[movies['title'].str.contains(entrada, case=False, na=False)]
        if filmes_encontrados.empty:
            print("Filme não encontrado.")
            continue
        
        nome_selecionado = filmes_encontrados.iloc[0]['title']
        recomendar_super_hibrido(user_id=1, titulo_filme=nome_selecionado)

if __name__ == "__main__":
    interface_usuario()