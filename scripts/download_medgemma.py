"""Download official MedGemma weights to a Git-ignored local directory."""

import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


def main() -> int:
    model_id = "google/medgemma-1.5-4b-it"
    destination = Path(__file__).resolve().parents[1] / "checkpoints" / "medgemma-1.5-4b-it"
    token = get_token()
    print(f"Model: {model_id}", flush=True)
    print(f"Destination: {destination}", flush=True)
    print(f"Hugging Face credentials available: {bool(token)}", flush=True)
    try:
        revision = HfApi().model_info(model_id, token=token).sha
        # Check gated access before initiating large weight transfers.
        hf_hub_download(model_id, "config.json", revision=revision,
                        local_dir=destination, token=token)
        print(f"Access confirmed. Downloading revision {revision}", flush=True)
        snapshot_download(
            model_id, revision=revision, local_dir=destination, token=token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.jinja", "*.txt", "README.md", "LICENSE*"],
            max_workers=2,
        )
    except GatedRepoError:
        print("Download blocked by Hugging Face model access.", flush=True)
        print(f"Accept the model terms at https://huggingface.co/{model_id}", flush=True)
        print(r"Then authenticate locally: .\.venv\Scripts\hf.exe auth login", flush=True)
        print("Rerun this script afterward. Do not paste access tokens into chat.", flush=True)
        return 2
    except HfHubHTTPError as error:
        status = getattr(error.response, "status_code", "unknown")
        print(f"Hugging Face request failed (HTTP {status}); rerun after resolving access/network issues.", flush=True)
        return 3
    (destination / "download_manifest.json").write_text(
        json.dumps({"model_id": model_id, "revision": revision}, indent=2) + "\n", encoding="utf-8",
    )
    print("Download complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
