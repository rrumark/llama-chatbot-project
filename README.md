# Llama3.1 Chatbot Project

This project demonstrates how to create a chatbot using the Llama 3.1 model from Ollama, combined with FAISS for vector storage and retrieval. The chatbot is designed to answer questions related to coffee, but it can be adapted for other topics by changing the data.

![Llama Chatbot](readme.png)


## Project Structure

```
llama-chatbot-project
├── data.zip                # Dataset containing JSON files to be used for training
├── llama_model.py          # Main script that processes the data and initializes the chatbot
├── requirements.txt        # List of required Python packages
├── .gitignore              # Files and directories to ignore in version control
├── readme.md               # Project documentation
└── test.py                 # Script to test the chatbot with predefined questions
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (Optional for faster embeddings)
- Ollama must be installed locally

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rrumark/llama-chatbot-project.git
cd llama-chatbot-project
```

### 2. Create and Activate a Virtual Environment

It's important to create an isolated virtual environment for the project and install dependencies within that environment.

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Required Packages

Install the required Python packages listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Ollama must be installed locally to run the chatbot. Visit the following link to install Ollama:

[Ollama Installation](https://ollama.com/download)

### 5. Test the Chatbot

Run the `test.py` script to test the chatbot with some predefined questions:

```bash
python test.py
```

