# Neural Sentiment Classification

This project implements a Deep Averaging Network (DAN) for sentiment classification using PyTorch, featuring:

1. Optimization experiments with SGD
2. Neural network implementation for sentiment analysis
3. Handling of misspelled words using prefix embeddings or spelling correction

## Project Structure

- \`optimization.py\`: Implementation of SGD optimization for a quadratic function
- \`models.py\`: Neural network models including Deep Averaging Network
- \`neural_sentiment_classifier.py\`: Main driver for sentiment classification
- \`sentiment_data.py\`: Data loading and processing utilities
- \`utils.py\`: Helper functions and utilities

## Features

- Feedforward Neural Network implementation
- Word embedding integration (GloVe)
- Batch processing
- Typo-resistant classification
- SGD optimization visualization

## Requirements

\`\`\`
numpy
nltk
spacy
torch
scipy
matplotlib
torchvision
\`\`\`

## Setup

1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Download and place GloVe embeddings in the \`data\` directory:
   - glove.6B.50d-relativized.txt
   - glove.6B.300d-relativized.txt

## Usage

### Optimization Experiment
\`\`\`bash
python optimization.py
\`\`\`

### Sentiment Classification
\`\`\`bash
# Standard classification
python neural_sentiment_classifier.py

# Classification with typo handling
python neural_sentiment_classifier.py --use_typo_setting
\`\`\`

## Performance

- Base model achieves >77% accuracy on development set
- Typo-resistant model achieves >74% accuracy on modified development set
