#!/bin/bash

models=(
    "gemma3:4b"
    "qwen2.5:7b"
    "llama3.2:3b"
    "llama3.1:8b"
    "phi4:14b"
    "gemma3:11b"
    "llama3.2-vision:11b"
    "minicpm-v:8b"
    "granite3.2-vision:2b"
    "llava-llama3:8b"
)

for model in "${models[@]}"
do
    echo "Stahuji model $model..."
    ollama pull "$model"
    echo "Model $model byl úspěšně stažen."
done

echo "✅ Všechny modely byly staženy." 