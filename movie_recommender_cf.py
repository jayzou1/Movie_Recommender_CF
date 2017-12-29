import pandas as pd
import numpy as np
import ast

##Read in the data files from the movieLens data set into pandas dataframes.
def load_in_data():
    path = "./ml-latest-small/"
    dfs = pd.read_csv(path + "links.csv"), pd.read_csv(path + "movies.csv"), pd.read_csv(path + "ratings.csv"), pd.read_csv(path + "tags.csv")
    return dfs

links, movies, ratings, tags = load_in_data()

##Since the data is separated into multiple files that aren't immediately interpretable, let's combine movies and ratings into a bucket of dictionaries based on user id.
def preprocess():
    print "Preprocessing data..."
    
    numUsers = ratings["userId"].nunique()
    ##Create the bucket
    buckets = {}
    
    for index, row in ratings.iterrows():
        entry = {int(row["movieId"]):float(row["rating"])}
        if int(row["userId"]) in buckets:
            buckets[int(row["userId"])].update(entry)
        else:
            buckets[int(row["userId"])] = entry

    print "Done."
    return buckets

##Some useful linear algebra operations.
def dot_product(v1, v2):
    return np.sum(x*y for x,y in zip(v1,v2))

def vlength(v):
    return np.sum(x*x for x in v)

##Euclidean distance for similarity scoring.
def euclidean_distance(user1, user2):
    overlap = {}
    for movie in user1.keys():
        if movie in user2.keys():
            overlap[movie] = 1
    if len(overlap) == 0:
        return 0

    euclidean = 0
    for movie in overlap:
        euclidean += np.square(user1[movie] - user2[movie])

    return 1/(1+np.sqrt(euclidean))

##Movie look up from pandas dataframe.
def getMovieName(id):
    return movies.loc[movies["movieId"] == id]["title"].values[0]

##Interface for the user to submit queries and received recommendations.
def interface():
    #Recommendation function thaat returns the first item not shared by both users. Since dictionaries are unordered, this recommendation should be pseudo-random.
    def recommend(v1, v2):
        for key in v2.keys():
            if key not in v1.keys():
                return key

    data = preprocess()
    example_query = {31 : 2.5, 1029: 3.0, 1061 : 3.0}
    
    #UI for user input.
    query = raw_input("Enter dictionary in the form of {movie ID : rating}: ")
    
    if not query:
        query = example_query
    else:
        query = ast.literal_eval(query)

    max = 0
    similar_user = {}
    for user in data.keys():
        score = euclidean_distance(query, data[user])
        if score > max and set(data[user].items()) != set(query.items()):
            max = score
            similar_user = data[user]

    print(getMovieName(recommend(query, similar_user)))

if __name__ == "__main__":
    interface()
