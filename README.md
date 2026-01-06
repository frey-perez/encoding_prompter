# Encoding Prompter

A Python package for using LLMs to encode psychological constructs found within interview text using structured codebooks.

## Overview

`encoding_prompter` provides a simple interface for qualitative researchers to leverage Large Language Models (LLMs) for coding interview transcripts. Given a codebook defining psychological constructs and interview documents, it automatically identifies and extracts instances of each construct, producing a structured DataFrame suitable for further analysis.

## Features

- **Multiple document formats**: Load interviews from `.txt` or `.csv` files
- **Batch processing**: Process entire directories of interview files
- **Flexible codebooks**: Define codebooks in `.json`, `.csv`, or `.txt` format
- **Speaker identification**: Automatically detects and preserves speaker IDs from transcripts
- **Customizable prompts**: Use the default prompt or provide your own template
- **Adjustable scoring**: Customize just the scoring criteria without rewriting the entire prompt
- **Model flexibility**: Use any model available through OpenRouter (free and paid options)
- **Structured output**: Results returned as a pandas DataFrame with document ID, speaker ID, construct, quote, and confidence score

## Installation

```bash
pip install encoding_prompter
```

Or install from source:

```bash
git clone https://github.com/yourusername/encoding_prompter.git
cd encoding_prompter
pip install -e .
```

## Quick Start

```python
from encoding_prompter import EncodingPrompter

# Initialize with your OpenRouter API key
prompter = EncodingPrompter(api_key="your-openrouter-api-key")

# Run encoding on a directory of interviews
results = prompter.encode(
    documents="path/to/interviews/",
    codebook="path/to/codebook.json"
)

# View results
print(results.head())
```

## Detailed Usage

### Setting Up

You can provide your API key directly or set it as an environment variable:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "your-api-key"

from encoding_prompter import EncodingPrompter

# API key will be read from environment
prompter = EncodingPrompter()
```

### Choosing a Model

By default, the package uses a free Llama model. You can specify any model available on OpenRouter:

```python
# Use a specific model
prompter = EncodingPrompter(
    api_key="your-api-key",
    model="anthropic/claude-3-5-sonnet"  # or any OpenRouter model ID
)
```

Common model options:
- `meta-llama/llama-3.1-8b-instruct:free` (default, free)
- `google/gemma-2-9b-it:free` (free)
- `anthropic/claude-3-5-sonnet` (paid)
- `openai/gpt-4o` (paid)

### Codebook Formats

#### JSON Format

```json
{
    "constructs": [
        {
            "name": "Emotional Awareness",
            "definition": "Recognition and acknowledgment of one's own emotional states",
            "examples": ["I felt sad when...", "I noticed I was getting anxious"]
        },
        {
            "name": "Relationship Conflict",
            "definition": "Description of interpersonal disagreements or tension",
            "examples": ["We had an argument about...", "There's friction between us"]
        }
    ]
}
```

#### CSV Format

```csv
name,definition,examples
Emotional Awareness,"Recognition and acknowledgment of one's own emotional states","I felt sad when...; I noticed I was getting anxious"
Relationship Conflict,"Description of interpersonal disagreements or tension","We had an argument about...; There's friction between us"
```

#### TXT Format

```
CONSTRUCT: Emotional Awareness
DEFINITION: Recognition and acknowledgment of one's own emotional states
EXAMPLES: I felt sad when...; I noticed I was getting anxious

CONSTRUCT: Relationship Conflict
DEFINITION: Description of interpersonal disagreements or tension
EXAMPLES: We had an argument about...; There's friction between us
```

### Document Formats

#### Interview TXT Format

The package automatically detects speaker IDs from transcript format:

```
SPEAKERS
TH-001, PA-001

TH-001  
Can you tell me about your experience?

PA-001  
I felt really anxious at first, but then I started to feel more comfortable...
```

#### CSV Format

```csv
speaker,text
TH-001,Can you tell me about your experience?
PA-001,"I felt really anxious at first, but then I started to feel more comfortable..."
```

### Processing Documents

#### Single File

```python
results = prompter.encode(
    documents="interview1.txt",
    codebook="codebook.json"
)
```

#### Directory of Files

```python
results = prompter.encode(
    documents="interviews/",  # Will process all .txt and .csv files
    codebook="codebook.json"
)
```

#### Single Text String

```python
results = prompter.encode_single(
    text="I felt anxious when meeting new people...",
    codebook="codebook.json",
    doc_id="participant_001"
)
```

### Customizing the Prompt

#### Custom Scoring Criteria

Change just the scoring instructions:

```python
results = prompter.encode(
    documents="interviews/",
    codebook="codebook.json",
    scoring_criteria="""
    Rate confidence on a 5-point scale:
    1 = Very unlikely to be this construct
    2 = Somewhat unlikely
    3 = Unclear/ambiguous
    4 = Likely this construct
    5 = Definite/prototypical example
    """
)
```

#### Full Custom Prompt

Provide your own prompt template (must include `{text}` and `{codebook}` placeholders):

```python
custom_prompt = """
Analyze the following therapy transcript for psychological themes.

Transcript:
{text}

Themes to identify:
{codebook}

For each theme found, provide:
DOC_ID: [filename]
SPEAKER_ID: [speaker]
CONSTRUCT: [theme name]
QUOTE: [relevant excerpt]
CONFIDENCE: [1-5 rating]
"""

results = prompter.encode(
    documents="interviews/",
    codebook="codebook.json",
    prompt_template=custom_prompt
)
```

### Output Format

The `encode()` method returns a pandas DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| `doc_id` | Document identifier (filename or provided ID) |
| `speaker_id` | Speaker identifier from the transcript |
| `construct` | Name of the identified construct |
| `quote` | Exact quote from the text as evidence |
| `confidence` | Ordinal score (default: 0, 1, or 2) |

Example output:

```
     doc_id speaker_id            construct                                    quote  confidence
0  interview1    PA-001   Emotional Awareness  I felt really anxious at first                  2
1  interview1    PA-001   Emotional Awareness  started to feel more comfortable                1
2  interview1    PA-001  Relationship Conflict  There's been some friction with my sister       2
```

### Debugging and Preview

#### Preview the Prompt

See exactly what will be sent to the LLM:

```python
prompt_text = prompter.preview_prompt(
    document="interview1.txt",
    codebook="codebook.json"
)
print(prompt_text)
```

#### Get Raw Response

Get the unparsed LLM response:

```python
response = prompter.get_raw_response(
    document="interview1.txt",
    codebook="codebook.json"
)
print(response.content)
print(f"Tokens used: {response.usage}")
```

### Callback for Progress Tracking

Track progress with a callback function:

```python
def on_complete(doc_id, results_df):
    print(f"Completed {doc_id}: found {len(results_df)} instances")
    results_df.to_csv(f"results/{doc_id}.csv", index=False)

results = prompter.encode(
    documents="interviews/",
    codebook="codebook.json",
    on_document_complete=on_complete
)
```

## Jupyter Notebook Example

```python
# Cell 1: Setup
from encoding_prompter import EncodingPrompter
import pandas as pd

prompter = EncodingPrompter(api_key="your-api-key")

# Cell 2: Load and preview codebook
from encoding_prompter import Codebook
cb = Codebook.from_file("codebook.json")
print(f"Loaded {len(cb)} constructs:")
for construct in cb:
    print(f"  - {construct.name}")

# Cell 3: Run encoding
results = prompter.encode(
    documents="interviews/",
    codebook=cb,
    show_progress=True
)

# Cell 4: Analyze results
print(f"Total instances found: {len(results)}")
print("\nInstances by construct:")
print(results['construct'].value_counts())

# Cell 5: Filter high-confidence instances
high_confidence = results[results['confidence'] == 2]
print(f"\nHigh-confidence instances: {len(high_confidence)}")

# Cell 6: Export results
results.to_csv("encoding_results.csv", index=False)
```

## API Reference

### EncodingPrompter

Main class for encoding constructs.

```python
EncodingPrompter(
    api_key: str | None = None,
    model: str = "meta-llama/llama-3.1-8b-instruct:free"
)
```

#### Methods

- `encode()`: Process multiple documents
- `encode_single()`: Process a single text string
- `preview_prompt()`: Preview the formatted prompt
- `get_raw_response()`: Get unparsed LLM response

### Codebook

Class for loading and managing codebooks.

```python
Codebook.from_file(filepath: str) -> Codebook
```

### DocumentLoader

Class for loading interview documents.

```python
DocumentLoader.load(path: str) -> list[Document]
DocumentLoader.load_from_string(text: str, doc_id: str) -> Document
```

## License

MIT License
