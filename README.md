# Food vs Not-Food Text Classifier

A text-classification project that uses **Hugging Face Transformers** and **DistilBERT** to predict whether a sentence or image caption is about food (`food`) or not about food (`not_food`).

The project covers the end-to-end workflow: loading a dataset, preparing labels, tokenizing text, fine-tuning a pretrained transformer, evaluating predictions, saving the model, uploading it to the Hugging Face Hub, and running inference on custom text.

## Project Links

- **GitHub repository:** https://github.com/aakashdeep-1/hugging_face_food_not_food_text_classifier
- **Hugging Face model:** https://huggingface.co/aakashdeep1/hf_food_not_food_text_classifier-distilbert-base-uncased
- **Base model:** [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- **Dataset:** [Food/Not-Food Image Captions](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions)
  
## What It Does

Given text, the fine-tuned model returns one of two classes:

- `food` — text that describes or refers to food.
- `not_food` — text that is not about food.

Example inputs include food descriptions such as “A delicious slice of pizza straight from the oven” and non-food descriptions such as “The sunset over the ocean was breathtaking.”

The notebook also tests ambiguous, figurative language—for example, expressions such as “a recipe for disaster”—to explore cases where food-related words do not necessarily mean the sentence is about food.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- Hugging Face Hub
- NumPy and Pandas
- Matplotlib for training-curve visualisation
- scikit-learn for checking test predictions

## Dataset and Preparation

The notebook loads `mrdbourke/learn_hf_food_not_food_image_captions` from the Hugging Face Hub.

The workflow includes:

1. Inspecting dataset fields, sample records, and label counts.
2. Mapping the two labels to numeric IDs.
3. Shuffling and splitting the data into training and test sets using an **80/20 split** and `seed=42`.
4. Tokenizing text with the tokenizer associated with `distilbert/distilbert-base-uncased`.
5. Padding and truncating sequences so that text can be processed by the model.

Label mapping:

| Label ID | Class |
|---:|---|
| `0` | `not_food` |
| `1` | `food` |

## Model and Training

The project uses transfer learning: a pretrained DistilBERT model is adapted for two-class sequence classification.

| Setting | Value |
|---|---|
| Base model | `distilbert/distilbert-base-uncased` |
| Task | Binary text classification |
| Number of labels | 2 |
| Learning rate | `0.0001` |
| Training batch size | `32` |
| Evaluation batch size | `32` |
| Epochs | `10` |
| Random seed | `42` |
| Evaluation schedule | Each epoch |
| Training API | Hugging Face `Trainer` |

The notebook tracks training/evaluation metrics and plots loss curves to inspect learning over time. It also saves the trained model locally and includes a step to upload it to the Hugging Face Hub.

## Installation

Use Python in a virtual environment, Jupyter Notebook, or Google Colab.

```bash
pip install -U torch transformers datasets evaluate accelerate huggingface_hub numpy pandas matplotlib scikit-learn
```

Then open the notebook in this repository and run the cells in order.

> **Note:** PyTorch device support depends on your environment. The notebook checks for Apple Silicon MPS and can fall back to CPU. A CUDA-enabled GPU can also be used when available and configured.

## Use the Trained Model

Install the required libraries, then load the published model with the Hugging Face `pipeline` API:

```python
from transformers import pipeline

model_id = "aakashdeep1/hf_food_not_food_text_classifier-distilbert-base-uncased"

classifier = pipeline(
    task="text-classification",
    model=model_id,
)

examples = [
    "A delicious slice of pizza straight from the oven.",
    "The sunset over the ocean was breathtaking.",
    "The new project was a recipe for disaster.",
]

predictions = classifier(examples)

for text, prediction in zip(examples, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {prediction['label']}")
    print(f"Score: {prediction['score']:.4f}")
    print()
```

The `score` is the model's confidence score for its selected class; it should not be interpreted as a guarantee that the prediction is correct.

## Inference Approaches Explored

The notebook demonstrates two ways to use the trained model:

1. **Transformers pipeline:** a convenient high-level interface for individual and batch predictions.
2. **Direct PyTorch inference:** loading `AutoTokenizer` and `AutoModelForSequenceClassification`, tokenizing inputs, and converting model logits to predicted labels and probabilities.

It also compares inference on individual sentences with batched prediction to explore throughput.

## Evaluation

The notebook evaluates predictions on the held-out test split using accuracy. It also:

- Converts model logits to probabilities using softmax.
- Compares predicted labels with true labels.
- Builds a Pandas DataFrame of test examples, true labels, predicted labels, and prediction scores.
- Inspects low-confidence examples and plots training/evaluation loss curves.

No fixed accuracy figure is claimed here; refer to the notebook's executed evaluation output for the results from a particular run.

## Repository Contents

- **Jupyter Notebook** — dataset exploration, preprocessing, tokenization, training, evaluation, model upload, and inference examples.
- **README.md** — project overview and usage instructions.

The trained model is hosted separately on the Hugging Face Hub, linked above.

## Limitations

- This is a **text classifier**, not an image classifier. It classifies the words in a sentence or caption; it does not inspect an image.
- The model learns from the supplied caption dataset, so performance may vary on new domains, unusual wording, sarcasm, metaphors, or text that mentions food figuratively.
- A high confidence score does not guarantee a correct prediction.
- Test performance should be interpreted in the context of dataset size, label balance, and how representative the test split is of real-world inputs.

## Possible Improvements

- Evaluate with precision, recall, macro F1, and a confusion matrix in addition to accuracy.
- Test on a separately collected set of real-world captions.
- Review false positives, false negatives, and ambiguous examples.
- Compare results against a simple baseline and document reproducible metrics.
- Add a small inference demo if a user-facing interface is needed.

Built as a hands-on project to practise transformer-based text classification with Hugging Face.
