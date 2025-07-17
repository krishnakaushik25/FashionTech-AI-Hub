
# Fashion Sense AI: Visual Product Discovery & Personalized Styling

**An advanced web application for fashion visual search, personalized recommendations, and intelligent outfit composition—leveraging deep learning, modern LLMs, and seamless UI.**

---

## Introduction

Fashion Sense AI redefines digital shopping and personal styling by merging powerful computer vision and language models into an accessible web interface. This tool enables users to explore visually similar apparel, receive AI-curated outfit suggestions, and simulate browsing habits for a more tailored experience—all within a single platform.

---

## Key Capabilities

- **Visual Similarity Search:** Submit an image of any clothing item and instantly obtain lookalike products from the catalog.
- **Natural Language Querying:** Enter a text description (e.g., “floral maxi dress, white sneakers”) to uncover matching styles and products.
- **Customized Recommendations:** Benefit from system-generated or user-driven browsing histories that refine suggestions.
- **AI-Based Outfit Completion:** Harness a language model (Gemma-3B via Hugging Face) to enhance your ensemble with recommended accessories or complementary items.
- **Fashion Trend Insights:** Automated extraction and utilization of trending keywords from real-world data for dynamic recommendations.

---

## System Components

- **User Interface:** Streamlit-based interactive app, optimized for ease of use and rapid iteration.
- **Image Processing:** CLIP (ViT-L/14) for encoding and matching visuals.
- **Text Understanding:** SentenceTransformers for rich, contextual queries.
- **Product Indexing:** Efficient search built on FAISS and precomputed embeddings.
- **Styling Engine:** Gemma-3B LLM, integrated through Hugging Face's inference endpoint.
- **Trend Analysis:** Web data scraping and TF-IDF extraction for current style relevance.

---

## Directory Layout

```
FashionSense-AI/
│
├── Assets/         # Embeddings, indices, and catalog identifiers
├── Data/           # Cleaned datasets for product categories
├── Modules/        # Feature-specific code (search, suggestions, trends)
├── Src/            # App graphics and static resources
├── Notebooks/      # Jupyter notebooks for prototyping and exploration
├── Test_Images/    # Example images for testing
├── app.py          # Streamlit application entry point
├── requirements.txt
└── README.md
```

---

## Installation Guide

**Requirements:** Python 3.11 or newer

1. Clone this repository:
   ```
   git clone https://github.com/MohitGupta0123/Fashion-Sense-AI.git
   cd Fashion-Sense-AI
   ```
2. Install all dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the application:
   ```
   streamlit run app.py
   ```
4. For outfit suggestion features, acquire a Hugging Face API token ([get one here](https://huggingface.co/settings/tokens)) and enter it via the sidebar when prompted.

---

## User Workflow

- **Image or Text Search:** Initiate a search by uploading a product image or typing a descriptive phrase.
- **Results Customization:** Modify the number of results shown using the interactive controls.
- **Personal History Simulation:** Generate a virtual browsing history or use your real activity for contextual suggestions.
- **Outfit Enhancement:** With one click, allow the system to propose full outfit compositions tailored to your style input and current trends.

---

## Example Scenarios

- Upload an image of a denim jacket to browse visually similar alternatives.
- Search for “pastel summer skirt” to see matching inventory.
- Use generated or real browsing history to refine product recommendations.
- Ask the AI for suggestions to complete an outfit, receiving items like footwear or layering pieces.

---

## Security & Token Management

The Hugging Face token, essential for LLM-powered suggestions, is held securely for each session and is never persisted or exposed externally.

---

## Roadmap

Planned future enhancements include:
- Secure, persistent user authentication and history tracking.
- Direct “add to cart” functionality for partnered e-commerce sites.
- Support for voice-driven queries and recommendations.
- Expanded chatbot capabilities for interactive fashion assistance.

