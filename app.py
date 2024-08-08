import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('model/')
    tokenizer = AutoTokenizer.from_pretrained('model/')
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text, model, tokenizer):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=200,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).cpu().numpy()[0]

    labels = ['neutral', 'positive', 'negative']
    predicted_label = labels[predicted_class]
    return predicted_label, probabilities.cpu().numpy()

def main():
    st.title('Airline Tweet Sentiment Analysis')
    tweet_text = st.text_area("Enter an airline tweet:", "Type here...")
    
    if st.button("Analyze Sentiment"):
        predicted_label, probabilities = predict_sentiment(tweet_text, model, tokenizer)
        st.write(f"The predicted sentiment is: **{predicted_label}**")
        
        # Display probabilities as a progress bar
        st.subheader("Sentiment Confidence Levels:")
        probabilities_np = probabilities.squeeze()  # Ensure it's a flat array
        for i, label in enumerate(['neutral', 'positive', 'negative']):
            prob = probabilities_np[i]  # Get the probability as a scalar
            st.write(f"{label.capitalize()}: {prob:.2f}")
            st.progress(float(prob))  # Ensure conversion to float for progress bar

if __name__ == '__main__':
    main()