# Setup

## Local environment (Task 1 and Task 3)

Tasks 1 and 3 are network-bound and run comfortably on a modest machine. Task 2 does not run locally, see the Colab section below.

```powershell
cd E:\CDAZZDEV\CDAZZDEV-MLE-DINOJAN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## API keys

```powershell
Copy-Item .env.example .env
notepad .env
```

| Key | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | yes | https://console.groq.com/keys, free, no card |
| `OPENROUTER_API_KEY` | optional | https://openrouter.ai/keys, used only as fallback |
| `HF_TOKEN` | Task 2 only | https://huggingface.co/settings/tokens, needs write scope to push the model |
| `WANDB_API_KEY` | Task 2 only | https://wandb.ai/authorize, optional if logging manually |

`.env` is gitignored. Confirm before any push:

```powershell
git check-ignore -v .env
```

## Verify the install

```powershell
python -c "from common.llm import get_client; print(get_client().chat('You are terse.', 'Reply with OK'))"
```

## Machine notes

This laptop is an Intel Celeron N4120, 4 cores, 7.82 GB RAM, Intel UHD 600 with no CUDA.

- Close spare browser tabs before running notebooks. Free RAM measured 0.5 GB at setup time and pandas plus yfinance want roughly 400 to 700 MB.
- Commit charge sat at 18.08 GB against a 21.35 GB limit. If Python dies with a MemoryError or Windows warns that virtual memory is low, raise the pagefile: System Properties, Advanced, Performance Settings, Advanced, Virtual Memory, Change. Set a system-managed size on a drive with free space.
- The virtual environment lives on E:, which has more headroom than C:.
- Do not `pip install torch` locally. It is a multi-gigabyte download with no CUDA runtime to use it.

## Task 2 on Google Colab

Fine-tuning requires a CUDA GPU, so Task 2 runs entirely on Colab free tier.

1. Open `task2_genai/notebook.ipynb` in Colab.
2. Runtime, Change runtime type, T4 GPU.
3. Store `HF_TOKEN` and `GROQ_API_KEY` in Colab Secrets (the key icon in the left sidebar), never inline in a cell.
4. Run all cells and leave the outputs in place before downloading the notebook back into this repo.

Colab free tier disconnects after roughly 90 minutes of inactivity and caps sessions at around 12 hours. Checkpoint to Google Drive so a disconnect does not cost the whole run.
