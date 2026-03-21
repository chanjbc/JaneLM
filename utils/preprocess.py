from pathlib import Path
import re

def preprocess(in_path: Path, out_path: Path):
    text = in_path.read_text(encoding="utf-8")

    # Normalize typographic characters
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("„", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("—", "--")

    # Normalize spacing
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse multiple horizontal spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines
    text = text.strip()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    BASE_DIR = BASE_DIR.parent / "data"

    austen_path = data_path / "austen.txt"
    dickens_path = data_path / "dickens.txt"
    austen_preprocess_path = data_path / "preprocess-austen.txt"
    dickens_preprocess_path = data_path / "preprocess-dickens.txt"

    if not austen_path.is_file() or not dickens_path.is_file():
        kagglehub.dataset_download("joshmcadams/jane-austin-and-charles-dickens", output_dir=data_path)

    if not austen_preprocess_path.is_file() or dickens_preprocess_path.is_file():
        preprocess(austen_path, austen_preprocess_path)
        preprocess(dickens_path, dickens_preprocess_path)
