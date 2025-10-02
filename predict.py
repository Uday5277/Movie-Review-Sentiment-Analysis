import joblib
import sys

model = joblib.load("models/logreg_tfidf_pipeline.joblib")

def predict_review(text):
    label = model.predict([text])[0]
    return "positive" if label==1 else "Negative"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
        print("Review:", review)
        print("Predicted sentiment:", predict_review(review))
    else:
        print("Please provide a review text as argument")