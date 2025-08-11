# Spam SMS Detcetion

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('C:/Users/adity/Downloads/Sms/spam.csv', encoding='ISO-8859-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title('Spam vs Ham Count')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

def predict_sms(message):
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)[0]
    return "Spam" if prediction == 1 else "Not Spam ✅"

# === Test with Example ===
example_sms = "Congratulations! You've won a free iPhone. Click here to claim now!"
print(f"\nPrediction for sample SMS: '{example_sms}' => {predict_sms(example_sms)}")
