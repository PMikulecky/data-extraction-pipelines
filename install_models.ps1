$models = @(
    "gemma3:4b",
    "qwen2.5:7b",
    "llama3.2:3b",
    "llama3.1:8b",
    "phi4:14b",
    "gemma3:11b",
    "llama3.2-vision:11b",
    "minicpm-v:8b",
    "granite3.2-vision:2b",
    "llava-llama3:8b"
)

foreach ($model in $models) {
    Write-Host "Stahuji model $model..."
    ollama pull $model
    Write-Host "Model $model byl úspěšně stažen."
}

Write-Host "✅ Všechny modely byly staženy." 