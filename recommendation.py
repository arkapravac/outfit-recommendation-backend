import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.keras.optimizers import Adam

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.product_features = None
        self.product_ids = None

    def fit(self, products_df):
        # Combine relevant features into a single text field
        products_df['features'] = products_df.apply(
            lambda x: f"{x['category']} {x['style']} {x['color']} {x['occasion']}",
            axis=1
        )
        
        # Create TF-IDF matrix
        self.product_features = self.tfidf.fit_transform(products_df['features'])
        self.product_ids = products_df.index.tolist()

    def recommend(self, product_id, n_recommendations=5):
        if product_id not in self.product_ids:
            return []
        
        idx = self.product_ids.index(product_id)
        sim_scores = cosine_similarity(self.product_features[idx], self.product_features)
        sim_scores = sim_scores.flatten()
        
        # Get top similar products
        product_indices = sim_scores.argsort()[::-1][1:n_recommendations+1]
        return [self.product_ids[i] for i in product_indices]

class CollaborativeRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.model = None

    def build_model(self, n_users, n_items):
        # User input
        user_input = Input(shape=(1,))
        user_embedding = Embedding(n_users, self.n_factors)(user_input)
        user_vec = Flatten()(user_embedding)

        # Item input
        item_input = Input(shape=(1,))
        item_embedding = Embedding(n_items, self.n_factors)(item_input)
        item_vec = Flatten()(item_embedding)

        # Dot product for prediction
        prod = Dense(1, activation='sigmoid')(user_vec * item_vec)

        # Create model
        self.model = Model([user_input, item_input], prod)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam())

    def fit(self, user_item_matrix, epochs=10, batch_size=64):
        if self.model is None:
            n_users, n_items = user_item_matrix.shape
            self.build_model(n_users, n_items)

        # Convert matrix to training data
        users, items = user_item_matrix.nonzero()
        labels = user_item_matrix[users, items]

        self.model.fit(
            [users, items],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )

    def recommend(self, user_id, n_recommendations=5):
        # Predict ratings for all items
        user_ids = np.full(self.n_items, user_id)
        item_ids = np.arange(self.n_items)
        predictions = self.model.predict([user_ids, item_ids])

        # Get top predicted items
        top_items = item_ids[predictions.flatten().argsort()[::-1][:n_recommendations]]
        return top_items.tolist()

class HybridRecommender:
    def __init__(self, content_weight=0.5):
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeRecommender()
        self.content_weight = content_weight

    def fit(self, products_df, user_item_matrix):
        self.content_recommender.fit(products_df)
        self.collab_recommender.fit(user_item_matrix)

    def recommend(self, user_id, product_id, n_recommendations=5):
        # Get recommendations from both models
        content_recs = self.content_recommender.recommend(product_id, n_recommendations)
        collab_recs = self.collab_recommender.recommend(user_id, n_recommendations)

        # Combine recommendations with weighted scoring
        final_recs = {}
        for i, rec in enumerate(content_recs):
            final_recs[rec] = self.content_weight * (n_recommendations - i)
        for i, rec in enumerate(collab_recs):
            if rec in final_recs:
                final_recs[rec] += (1 - self.content_weight) * (n_recommendations - i)
            else:
                final_recs[rec] = (1 - self.content_weight) * (n_recommendations - i)

        # Sort and return top recommendations
        sorted_recs = sorted(final_recs.items(), key=lambda x: x[1], reverse=True)
        return [rec[0] for rec in sorted_recs[:n_recommendations]]