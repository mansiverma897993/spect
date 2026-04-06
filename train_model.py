import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

def main():
    print("Downloading dataset...")
    # Standard SMS Spam Dataset
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    
    # Preprocess labels: 'ham' -> 0, 'spam' -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['message']
    y = df['label']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Pipeline: Vectorize text (includes stopwords & lowercasing), then predict
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('classifier', LogisticRegression(C=2.0, class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    # Evaluate Training Accuracy
    y_train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate Testing Accuracy
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy:   {train_accuracy:.4f}")
    print(f"Testing Accuracy:    {test_accuracy:.4f}")
    
    gap = abs(train_accuracy - test_accuracy)
    print(f"Accuracy Gap (Overfitting Check): {gap:.4f}")
    
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_test_pred))
    
    print("Saving model pipeline...")
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("Successfully compiled and saved model to `spam_model.pkl`. You can now run the web app!")

if __name__ == "__main__":
    main()
