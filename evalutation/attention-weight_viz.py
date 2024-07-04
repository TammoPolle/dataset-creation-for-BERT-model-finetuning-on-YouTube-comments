import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

# Lade das finetunete BERTweet-Modell und den Tokenizer
model_name = "E:\\research\\experiments\\gen\\min_len\\bertweet\\bertweet_50_all\\output"  # Ersetze dies durch den Pfad zu deinem finetuneten Modell
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)


# Funktion zur Extraktion der Attention Weights und Gradienten
def extract_attention_weights_and_gradients(text, label):
    # Tokenisiere den Text
    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.tensor([label]).unsqueeze(0)

    # Setze das Modell in den Trainingsmodus und aktiviere die Gradientenberechnung
    model.train()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    # Berechne die Gradienten
    for att in outputs.attentions:
        att.retain_grad()
    loss.backward()

    # Extrahiere die Attention Weights und die Gradienten
    attentions = outputs.attentions
    gradients = [att.grad for att in attentions]

    return attentions, gradients, inputs['input_ids']


# Beispieltext und Label (das tatsächliche Label muss bekannt sein)
text = "It doesn't trump the constitution."
label = 6  # Beispielhaftes Label

# Extrahiere die Attention Weights und Gradienten
attentions, gradients, input_ids = extract_attention_weights_and_gradients(text, label)


# Bestimme den Layer und den Kopf, die am wichtigsten für die Klassifikation sind
def find_important_layer_and_head(attentions, gradients):
    importance_scores = []
    for layer in range(len(attentions)):
        for head in range(attentions[layer].size(1)):
            # Berechne die Wichtigkeit als Summe der absoluten Werte der Gradienten multipliziert mit den Attention Weights
            score = torch.sum(torch.abs(gradients[layer][0, head]) * attentions[layer][0, head]).item()
            importance_scores.append((score, layer, head))
    importance_scores.sort(reverse=True, key=lambda x: x[0])
    return importance_scores[0][1], importance_scores[0][2]  # Layer und Kopf mit dem höchsten Score


important_layer, important_head = find_important_layer_and_head(attentions, gradients)


# Funktion zur Visualisierung der wichtigsten Attention Weights
def visualize_important_attention(attentions, input_ids, layer, head):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attention_matrix = attentions[layer][0][head].detach().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(f'Attention Weights - Layer {layer + 1}, Head {head + 1}')
    plt.show()


# Visualisiere die wichtigsten Attention Weights
visualize_important_attention(attentions, input_ids, important_layer, important_head)
