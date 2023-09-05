import pandas as pd


from sklearn.metrics.pairwise import cosine_similarity

from typing import List

from pydantic import BaseModel

class GameRequest(BaseModel):
    game_name: str

df_games_items2 = pd.read_parquet('df_games_items2.parquet')
df_games_reviews2 = pd.read_pickle('df_games_reviews2.pkl')
df_combined = pd.read_pickle('df_combined.pkl')
category_playtime = pd.read_pickle('category_playtime.pkl')

df_combined = df_combined.reset_index(drop=True)
# Importa tus funciones y DataFrames
from Functions import userdata, countreviews, genre_rank

from fastapi import FastAPI
app = FastAPI()

# Define modelos Pydantic para las solicitudes y respuestas
class UserRequest(BaseModel):
    user_id: str

class DateRangeRequest(BaseModel):
    date1: str
    date2: str

class GenreRankRequest(BaseModel):
    category_name: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a tu API personalizada"}

@app.post("/userdata/")
def get_user_data(user_request: UserRequest):
    money_spent, recommend_percentage, num_items = userdata(
        user_request.user_id, df_games_reviews2, df_games_items2, df_combined)
    return {"money_spent": money_spent, "recommend_percentage": recommend_percentage, "num_items": num_items}

@app.post("/countreviews/")
def count_reviews(date_range_request: DateRangeRequest):
    num_users, percentage_recommendations = countreviews(
        date_range_request.date1, df_games_reviews2, date_range_request.date2)
    return {"num_users": num_users, "percentage_recommendations": percentage_recommendations}

@app.post("/genrerank/")
def get_genre_rank(genre_rank_request: GenreRankRequest):
    position = genre_rank(genre_rank_request.category_name)
    return {"position": position}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    
@app.post("/recommendations/")
def get_game_recommendations(game_request: GameRequest):
    recommendations = get_recommendations(game_request.game_name, similarity_matrix, num_recommendations)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)