import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os



df_games_items2 = pd.read_parquet('df_games_items2.parquet')
df_games_reviews2 = pd.read_pickle('df_games_reviews2.pkl')
df_combined = pd.read_pickle('df_combined.pkl')
category_playtime = pd.read_pickle('category_playtime.pkl')
df_combined = df_combined.reset_index(drop=True)


def userdata(User_id, df_games_reviews2, df_games_items2, df_combined):
    # Filtrar las revisiones del usuario específico en df_games_reviews2
    user_reviews = df_games_reviews2[df_games_reviews2['user_id'] == User_id]
    
    # Vincular las revisiones del usuario con los juegos en df_games_items2 usando user_id
    user_game_names = user_reviews.merge(df_games_items2, on='user_id', how='inner')['item_name']
    
    # Utilizar item_name para relacionar los juegos con los precios en df_games
    user_games_prices = df_combined[df_combined['game'].isin(user_game_names)]
    
    # Calcular la cantidad de dinero gastado por el usuario
    money_spent = user_games_prices['price'].sum()
    
    # Calcular el porcentaje de recomendación basado en las reviews
    recommend_percentage = (user_reviews['recommend'].sum() / user_reviews.shape[0]) * 100
    
    # Calcular la cantidad de items revisados
    num_items = len(user_reviews)
    
    return money_spent, recommend_percentage, num_items

# Consulta la información para el usuario con el ID deseado
User_id = 'wayfeng'  # Reemplaza 'tu_id_aqui' con el ID que desees consultar
money_spent, recommend_percentage, num_items = userdata(User_id, df_games_reviews2, df_games_items2, df_combined)

# Imprime o utiliza los resultados según sea necesario
print(f'Cantidad de dinero gastado por el usuario: ${money_spent:.2f}')
print(f'Porcentaje de recomendación: {recommend_percentage:.2f}%')
print(f'Cantidad de items revisados: {num_items}')




def countreviews(date1, df_games_reviews2, date2):
    # Convert the date strings to datetime objects
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)

    # Filter the DataFrame for reviews posted between date1 and date2
    filtered_reviews = df_games_reviews2[(df_games_reviews2['posted'] >= date1) & (df_games_reviews2['posted'] <= date2)]

    # Count the number of users who posted reviews in the filtered DataFrame
    num_users = len(filtered_reviews['user_id'].unique())

    # Calculate the percentage of recommendations
    total_reviews = len(filtered_reviews)
    if total_reviews > 0:
        percentage_recommendations = (filtered_reviews['recommend'].sum() / total_reviews) * 100
    else:
        percentage_recommendations = 0

    return num_users, percentage_recommendations
# Example usage:
date1 = '2015-12-01'  # Replace with your desired start date
date2 = '2017-12-02'  # Replace with your desired end date

# Call the function with your DataFrame and date range
num_users, percentage_recommendations = countreviews(date1, df_games_reviews2, date2)

print(f"Number of users who posted reviews between {date1} and {date2}: {num_users}")
print(f"Percentage of recommendations in the reviews: {percentage_recommendations:.2f}%")

# 3. Crea la función genre_rank
def genre_rank(category_name):
    # Convierte el nombre de la categoría a minúsculas
    category_name = category_name.lower()
    
    # Encuentra la posición de la categoría en el DataFrame ordenado
    position = category_playtime.columns.get_loc(category_name)
    
    # Devuelve la posición (1-indexed) de la categoría
    return position + 1

# Ejemplo de uso
categoria_buscada = 'action'  # Puedes cambiar esto a cualquier categoría
posicion = genre_rank(categoria_buscada)
print(f'La categoría "{categoria_buscada}" está en la posición #{posicion}')

# Llamada de ejemplo a la función genre_rank
categoria_buscada = 'action'  # Cambia esto a la categoría que desees buscar

# Llama a la función genre_rank con la categoría deseada
posicion = genre_rank(categoria_buscada)

# Imprime el resultado o devuélvelo como respuesta en tu API
print(f'La categoría "{categoria_buscada}" está en la posición #{posicion}')




categories = df_combined[['web_publishing', 'audio_production', 'strategy', 'adventure', 'photo_editing', 'rpg', 'action', 'utilities', 'accounting', 'free_to_play', 'massively_multiplayer', 'education', 'software_training', 'animation_modeling', 'racing', 'casual', 'design_illustration', 'early_access', 'simulation', 'sports', 'indie', 'video_production']]

# Calcula la matriz de similitud del coseno entre los juegos en función de las categorías
similarity_matrix = cosine_similarity(categories, categories)

# Función para obtener recomendaciones para un juego específico
def get_recommendations(game_name, similarity_matrix, num_recommendations=5):
    game_index = df_combined[df_combined['game'] == game_name].index[0]
    game_similarity = similarity_matrix[game_index]
    game_indices = game_similarity.argsort()[::-1]  # Ordenar por similitud descendente
    recommendations = []

    for i in range(1, num_recommendations + 1):
        recommended_game_index = game_indices[i]
        recommended_game_name = df_combined.loc[recommended_game_index, 'game']
        recommendations.append(recommended_game_name)

    return recommendations

# Obtener recomendaciones para un juego específico (reemplace 'NombreDelJuego' con el juego que desees)
game_name_to_recommend = 'ironbound'
num_recommendations = 5
recommendations = get_recommendations(game_name_to_recommend, similarity_matrix, num_recommendations)

print(f"Recomendaciones para el juego {game_name_to_recommend}:")
for i, recommended_game_name in enumerate(recommendations):
    print(f"{i + 1}: {recommended_game_name}")