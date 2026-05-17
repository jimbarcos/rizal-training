# Can You Beat Rizal? 
**Improving Machine Translation for Historical Tagalog Texts**

This project fine-tunes a general-purpose Tagalog-to-English machine translation model (`Helsinki-NLP/opus-mt-tl-en`) to handle 19th-century Tagalog using Low-Rank Adaptation (LoRA). The system features an interactive exhibit comparing human translations against a baseline and the improved AI model.

## 🛠️ Prerequisites
* Python 3.10+
* Node.js v18+ (For the Next.js frontend)
* Nvidia GPU with CUDA support (Highly recommended for training)

## 📦 Step 1: Environment Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt

```


3. **Install GPU-Enabled PyTorch (Windows/Nvidia Users):**
If you have an Nvidia GPU, uninstall the default CPU-only PyTorch and install the CUDA-enabled version to drastically speed up alignment and training:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```


4. **Download the SpaCy Multilingual Model:**
```bash
python -m spacy download xx_sent_ud_sm

```



## Step 2: Data Engineering Pipeline

Ensure the raw Gutenberg `.txt` files for *Noli Me Tangere* and *El Filibusterismo* are in the root directory.

1. **Neural Sentence Alignment:**
Uses LaBSE (Language-agnostic BERT Sentence Embedding) to match Tagalog sentences with their correct English translations.
```bash
python aligner.py

```


*(Outputs: `rizal_aligned_dataset.csv`)*
2. **Filter and Tokenize:**
Removes bad alignments, performs a stratified train/test split, and converts the text into tokenized arrow files for the model.
```bash
python prepare_dataset.py

```


*(Outputs: `rizal_tokenized_dataset/` directory)*

## Step 3: Model Training (LoRA)

Fine-tune the baseline model using Parameter-Efficient Fine-Tuning. This freezes the base model and only trains a small set of adapter weights.

```bash
python train.py

```

*(Outputs: `rizal-lora-adapters-final/` directory upon completion)*

## Step 4: Run the Inference Microservice

Do not load the AI models directly into the Next.js frontend. Instead, run the FastAPI Python server to handle the heavy translation generation.

```bash
python app.py

```

The API will start locally at `http://localhost:8000`. It exposes the `/api/translate` endpoint which accepts a Tagalog string and returns both the baseline and improved translations.

```

```