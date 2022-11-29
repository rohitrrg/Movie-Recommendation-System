from flask import Flask, render_template, request, flash
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import requests


movies = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies)
movies_list = movies['title'].values
movies['title'] = movies['title'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=3000)
vectors = cv.fit_transform(movies['tags']).toarray()

def fetch_details(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=4d3a04e97e46d3313ca2082c381d09e2&language=en-US".format(movie_id))
    data = response.json()
    return data

def get_similarity(index):
    similarity = cosine_similarity([vectors[index]], vectors)
    return similarity[0]

def get_recommendations(id):
    idx = movies[movies['id']==id].index[0]
    sim_scores = list(enumerate(get_similarity(idx)))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:7]
    movie_indices = [i[0] for i in sim_scores]

    rec_movies = dict()
    for i in movies['id'].iloc[movie_indices]:
        rec_movies[i] = fetch_details(i)
    return rec_movies

app = Flask(__name__)

@app.route("/")
@app.route("/home", methods=["POST","GET"])
def home():        
    # popular_movies = get_popular_movies(movies)
    return render_template('home.html', movies_list=movies_list)


@app.route("/movie", methods=["POST","GET"])
def submit():
    if request.method=='POST':
        name = str.lower(request.form.get('name'))
        id = movies[movies['title']==name]['id'].values[0]
        data = fetch_details(id)
        recommendations = get_recommendations(id)
        return render_template("movie.html", movies_list=movies_list, data=data, recommendations=recommendations)

@app.route("/<int:id>")
def load(id):
    data = fetch_details(id)
    recommendations = get_recommendations(id)
    return render_template("movie.html", movies_list=movies_list, data=data, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)