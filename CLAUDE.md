# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personality evaluation research system that analyzes human personality traits through prisoner's dilemma game dialogues using Large Language Models (LLMs). The system evaluates personality based on the Big Five model (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) by analyzing multi-round dialogue interactions between humans and AI agents.

## Core Architecture

The system follows a modular pipeline architecture:

1. **DialogueParser** (`dialogue_parser.py`): Parses structured dialogue files containing multi-round conversations
2. **PersonalityEvaluator** (`personality_evaluator.py`): Uses LangChain + OpenAI API to analyze personality traits from dialogues  
3. **ExperimentRunner** (`experiment_runner.py`): Orchestrates batch processing, handles parallel execution, and manages metrics calculation
4. **Main Entry Point** (`main.py`): Command-line interface for running experiments

## Key Commands

### Run a complete personality evaluation experiment:
```bash
python main.py --dir comparison --pattern '*_dialogue.txt' --ground-truth ground_truth/4o_DA.csv --output output --model gpt-4.1-nano --workers 16
```

### Process a single dialogue file:
```bash
python main.py --file comparison/1_外向性_dialogue.txt --output output
```

### Average metrics across multiple experiment runs:
```bash
python average_metrics.py --model deepseek-v3 --base_dir New_Experiment
```

### Filter by specific personality type:
```bash
python main.py --dir comparison --personality 尽责性 --output output
```

**Note**: The system now uses all 6 rounds of dialogue by default. The `--rounds` parameter has been removed as it's no longer needed.

## Data Structure

- **Input Files**: Dialogue files in format `{participant_id}_{personality_type}_dialogue.txt`
- **Dialogue Format**: Multi-round conversations with `Round{N}` markers and `User:` / `Agent:` exchanges (system processes all 6 rounds)
- **Output Structure**: 
  - Individual results: `eval_{participant_id}/{filename}_results.json`
  - Aggregated metrics: `detailed_metrics.csv`, `metrics_by_round.csv`, etc.

## Configuration

The system requires LLM API configuration:
- `--api-key`: OpenAI-compatible API key
- `--api-base`: API base URL (default: OpenAI proxy)
- `--model`: Model name (e.g., gpt-4.1-nano, deepseek-v3)

## Dependencies

Key dependencies include:
- `langchain_openai`: LLM integration
- `pandas`, `numpy`: Data processing and metrics
- `scikit-learn`: Error calculations (MAE, RMSE)
- `concurrent.futures`: Parallel processing
- `tqdm`: Progress tracking

## Personality Evaluation System

The evaluator uses a comprehensive prompt template that:
- Analyzes Big Five traits on a 1-5 scale with detailed criteria
- Requires specific evidence from dialogue content
- Handles boundary values and scoring precision
- Supports ablation studies (currently dialogue-only mode)

## Error Handling

The system includes robust error handling:
- Retry mechanisms for connection errors with exponential backoff
- Mock evaluations as fallbacks when LLM calls fail
- Comprehensive logging and failed file tracking
- Graceful degradation for missing ground truth data