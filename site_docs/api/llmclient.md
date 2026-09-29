# LLM Client

## Using Local LLM (Ollama) with ClimAID

ClimAID includes built-in support for generating **AI-powered reports using local language models** via Ollama. This allows you to create detailed, scientific summaries of disease projections **without requiring internet access or API keys**.

---

### Step 1 — Install Ollama

Before using the LLM features, you need to install Ollama on your system.

1. Visit the official website:
   https://ollama.com/download

2. Download and install the version for your operating system (Windows, macOS, or Linux).

---

### Step 2 — Start the Ollama Server

After installation, start the Ollama service:

```bash
ollama serve
```

This will launch a local server at:

```
http://localhost:11434
```

ClimAID connects to this server to generate reports.

---

### Step 3 — Download a Language Model

You must download at least one model before using the LLM.

Recommended:

```bash
ollama pull mistral
```

Other options include:

* `llama3` (strong reasoning)
* `phi3` (lightweight, faster)
* `mixtral` (larger, more powerful)

---

### Step 4 — Use with ClimAID

Once Ollama is running and a model is installed, you can use it directly:

```python
from climaid.llm_client import LocalOllamaLLM

llm = LocalOllamaLLM(model="mistral")
```

---

### Step 5 — Generate Reports

Pass the LLM client into the reporting pipeline:

```python
report = model.generate_report(
    projection_summary=summary,
    llm_client=llm,
    open_browser=True
)
```

ClimAID will automatically:

* Process projection outputs
* Generate structured summaries
* Use the LLM to create a readable scientific report

---

### Common Issues

**Ollama not running**

```
Ollama is not running
```

Fix:

```bash
ollama serve
```

---

**Model not found**

Fix:

```bash
ollama pull mistral
```

---

**Timeout error**

* Use a smaller model (`phi3`)
* Reduce input size

---

### Notes

* Runs fully offline
* No API keys required
* Ideal for secure or research environments
* Performance depends on your local machine

---

### API Reference

Below is the full API for the LLM client:


::: climaid.llm_client.LocalOllamaLLM