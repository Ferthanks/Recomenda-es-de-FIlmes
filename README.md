Um sistema de recomendação de filmes robusto construído em Python que utiliza uma abordagem híbrida avançada. O projeto combina Filtragem Colaborativa, Filtragem Baseada em Conteúdo e métricas de Popularidade para entregar sugestões precisas. Seu grande diferencial é a implementação do algoritmo MMR (Maximal Marginal Relevance) com um filtro de franquias, garantindo que o usuário receba recomendações diversas e não apenas sequências do mesmo filme.

🚀 Principais Funcionalidades
  Motor Híbrido de Recomendação: Calcula um score final combinando pesos ajustáveis para Conteúdo (40%), Colaborativo (40%) e Popularidade (20%).

  Popularidade Ponderada (Weighted Rating): Utiliza a fórmula de Weighted Rating (semelhante à do IMDb) para evitar que filmes com poucas avaliações, mas notas altas, dominem as recomendações.

  Processamento de Conteúdo via NLP: Usa TfidfVectorizer do Scikit-Learn para vetorizar os gêneros dos filmes e calcula a similaridade do cosseno entre eles.

  Filtragem Colaborativa Normalizada: Constrói uma matriz Usuário-Filme, subtrai a média de avaliações do usuário (mean-centering) para reduzir vieses e calcula a similaridade entre os itens.

  Diversidade com MMR e Filtro de Franquia: Implementa uma heurística de extração de radicais dos títulos (ex: "Star Wars", "Toy Story") aliada ao cálculo de Maximal Marginal Relevance. Isso penaliza a redundância e impede   que o sistema recomende múltiplas sequências da mesma franquia no Top 5.

  Tratamento de Cold Start: Se o usuário não existir na base de dados, o sistema redistribui automaticamente o peso colaborativo para a filtragem baseada em conteúdo.

  Interface CLI: Uma interface interativa de linha de comando para buscas rápidas por título.

🛠️ Tecnologias Utilizadas
  Linguagem: Python

  Manipulação de Dados: Pandas, NumPy

  Machine Learning & Matemática: Scikit-Learn (TfidfVectorizer, cosine_similarity)

  Processamento de Texto: Expressões Regulares (re)
