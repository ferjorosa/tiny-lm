# PR Description Instructions

Use this guide when writing PR descriptions for this repo (e.g. when an LLM is asked to draft a PR description from a diff or commit history).

## Structure

1. **Title** – `## [Concise title]`  
   Short, descriptive. No period. Example: "Config dataclasses and tokenization pipeline".

2. **Summary** – One paragraph (2–4 sentences).  
   What the PR does and why. No "This PR" or "In this PR"—state it directly.

3. **Changes** – `### Changes`  
   Group by area with **Bold subsection** headers. Each bullet: `` `path/to/file` `` – brief description.

   **Subsection examples:**
   - **Core Module:** / **Data config:** / **Tokenizer config:** – library code in `tiny_lm/`
   - **Configuration:** – YAML/config files in `configs/`
   - **Scripts:** – scripts in `scripts/`
   - **Project:** – `.gitignore`, `pyproject.toml`, lockfiles

4. **Features** – `### Features`  
   Bullet list of user-facing or notable capabilities (4–6 items). Short and factual.

## Style

- Concise and factual.
- No emojis.
- One line per file or logical group under Changes.

## Example

```markdown
## Tokenizer Training Infrastructure

Implements BPE tokenizer training with HuggingFace tokenizers library.

### Changes

**Core Module:**
- `tiny_lm/tokenizer/trainer.py` - BPE tokenizer training with configurable vocab size and special tokens

**Configuration:**
- `configs/tokenizers/tinystories-8k.yaml` - 8K vocab tokenizer config
- `configs/tokenizers/tinystories-16k.yaml` - 16K vocab tokenizer config

**Scripts:**
- `scripts/tokenizer/train_tokenizer.py` - Train tokenizer from config with default test

**Project:**
- Updated `.gitignore` to exclude trained tokenizers

### Features

- Config-driven tokenizer training
- HuggingFace native format (no conversion needed)
- Support for custom special tokens (pad, eos, bos, unk)
- Automatic testing after training
```
