# Installation Guide

This guide will help you set up AmbedkarGPT on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Ollama (for local LLM inference)

## Step 1: Clone the Repository

```bash
git clone https://github.com/pranav271103/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

## Step 2: Set Up Python Environment

### Create and activate a virtual environment:

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Set Up Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the required model:
   ```bash
   ollama pull mistral:latest
   ```
3. Start the Ollama server in a separate terminal:
   ```bash
   ollama serve
   ```

## Step 4: Prepare Your Documents

Place your text documents in the `corpus/` directory. The system will automatically process them when you first run the application.

## Step 5: Run AmbedkarGPT

```bash
python main.py
```

## Troubleshooting

### Common Issues

1. **Ollama not found**
   - Ensure Ollama is installed and the server is running
   - Check that the Ollama executable is in your system PATH

2. **Python package installation fails**
   - Make sure you're using Python 3.8 or higher
   - Try upgrading pip: `pip install --upgrade pip`

3. **Model not found**
   - Verify the model name in the configuration
   - Ensure you've pulled the model with Ollama

For additional help, please refer to the [FAQ](../faq.md) or open an issue on GitHub.
