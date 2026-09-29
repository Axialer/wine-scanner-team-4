# Testing wine recognition

Run commands from the repo root (`wine-scanner`). Activate your venv first if you use one:

```powershell
.\venv\Scripts\Activate.ps1
```

First run may download models and take several minutes.

## CLI (same JSON as the API)

Prints the OCR text first, then one JSON object (`ocr` is the first field).

The catalog is the full CSV (about 2103 wines). The number 18 is only the labeled phone-photo eval in `data/eval/real_photo_labels.json`. It is not the catalog size. Exact-match before/after the projection head: `data/eval/finetune_before.json` and `data/eval/finetune_after.json`.

`status` is `found` or `not_in_catalog`.

* `found` — the photo is confident enough to call the same catalog wine. The object is one card: name, category, color, region, grape, winery, description, image, slug. The web is not called.
* `not_in_catalog` — the bottle is not a confident catalog hit. `slug` and `name` are empty. The pipeline looks the label text up on the web (about 2.5s, cached by the normalized query in `index/external_cache`). `external` is `{name, grape, color, taste, source}` when a page was parsed, or `status: error` if the network failed. The page still says «В каталоге нет» and lists one primary plus up to three catalog cards. Those cards are ranked by cosine of the SigLIP text tower (`google/siglip2-base-patch16-224`) on grape, taste, and description (`index/catalog_text_emb.npy`), not by the label photo. If the web fails, the same ranker uses leftover OCR tokens (grape and color). `confidence` is the leader's score, not an identification.

Fine-tune check on the same 18 photos in `data/eval/real_photo_labels.json` (exact slug, brand if the winery stem matches): stock visual nearest-neighbor **10/18 (55.6%)** exact and **14/18 (77.8%)** brand, warm pass about 3.5 s (`data/eval/finetune_before.json`). A linear head on frozen image embeddings (`index/finetune_proj.npy`, held-out those slugs) scored **8/18 (44.4%)** exact and **10/18 (55.6%)** brand (`data/eval/finetune_after.json`). The head is not the default (`index/finetune_enabled.txt` is `0`). `84.73_07-09-2026_11-07-51.webp` stays `zhemchuzhnaya-9-aligote-czitron` on the stock path. Sibling bottles that share artwork stay hard to separate because a linear map cannot split identical image vectors.

`method` is `ocr_visual` when OCR narrowed the catalog and SigLIP ranked that shortlist, `visual_ocr` when the net searched the full catalog, `barcode` or `qr` when a code resolved to one wine. `visual_compared` is how many catalog wines SigLIP scored. Top-10 candidates are only in `_debug`. OCR text stays in `ocr` for both statuses.

```powershell
python src\hybrid_recognizer.py data\real_photo\84.73_07-09-2026_11-07-51.webp
```

Optional verbose debug (extra `_debug` in JSON and a human-readable dump):

```powershell
python src\hybrid_recognizer.py data\real_photo\84.73_07-09-2026_11-07-51.webp --debug
```

With `--debug`, JSON also includes `_debug.visual_scope` (`filtered` or `full_catalog`), `_debug.shortlist_size`, and the grape tokens used to filter.

OCR shortlist unit test (no SigLIP):

```powershell
.\venv\Scripts\python.exe tests\test_ocr_shortlist.py
```

Found / not-found and similar-wine unit test (no SigLIP; fake scores):

```powershell
.\venv\Scripts\python.exe tests\test_similar_match.py
.\venv\Scripts\python.exe tests\test_web_lookup.py
```

## HTTP API

Start the server:

```powershell
uvicorn api.server:app --app-dir src --host 0.0.0.0 --port 8091
```

`0.0.0.0` is required so a phone on the same Wi-Fi can open the UI. `127.0.0.1` only accepts this computer. On startup the process prints `http://<LAN-IPv4>:8091/` (virtual adapters, VPN, and link-local addresses are skipped). If 8091 is taken, use the next free port and the same host. Windows Firewall may need an inbound TCP allow rule for that port.

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8091/health
```

Recognize with PowerShell (multipart upload):

```powershell
$path = "data\real_photo\84.73_07-09-2026_11-07-51.webp"
$uri = "http://127.0.0.1:8091/recognize"
$form = @{ file = Get-Item -Path $path }
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

With debug:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8091/recognize?debug=true" -Method Post -Form $form
```

Equivalent `curl` (if installed):

```powershell
curl.exe -s -F "file=@data/real_photo/84.73_07-09-2026_11-07-51.webp" http://127.0.0.1:8091/recognize
```

## Dev web UI

With the server running, open on this computer:

**http://127.0.0.1:8092/**

Phone on the same network: **http://192.168.1.178:8092/** or **http://192.168.1.84:8092/** (`uvicorn api.server:app --app-dir src --host 0.0.0.0 --port 8092`).

Two tabs: **Каталог** (default) and **Распознавание**.

Catalog search (name, winery, grape), for example Жемчужная:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8091/catalog?q=жемчуж"
```

That list includes `zhemchuzhnaya-9-aligote-czitron` and `czitronnyj-magaracha-sovinon-blan`. `GET /catalog` without `q` returns the full catalog. Images are `/catalog-image/{slug}` when a file exists. The catalog is read from disk and does not wait for OCR or SigLIP.

Phone or a narrow/touch screen: **Камера** (rear camera) and **Галерея**. Choosing a photo sends it immediately.

Desktop (wide screen, fine pointer): drag-and-drop or a file picker only — no camera button.

A wine card (catalog or recognition) opens `/wine/{slug}` — for example [http://127.0.0.1:8091/wine/zhemchuzhnaya-9-aligote-czitron](http://127.0.0.1:8091/wine/zhemchuzhnaya-9-aligote-czitron). The page shows the title, photo, winery, region, grape, color, category, and the full description. **Назад** returns to the previous tab and keeps the catalog search text.

**Распознавание** is a chat: the photo is a user bubble, the reply is a wine card or a short Russian note plus similar cards, and another photo appends to the thread. OCR, JSON, and debug stay off the page. Phone: **Камера** and **Галерея**. Desktop: file picker and drag-and-drop onto the composer. The wine page shows the catalog image in full (`object-fit: contain`): image beside the text on a wide screen, stacked on a phone.

## Pipeline

1. QR / barcode (`zxing` + OpenCV), downscaled. A code already in the catalog wins immediately.
2. On a cache miss, GTIN goes to Open Food Facts `https://world.openfoodfacts.org/api/v2/product/{barcode}.json`. A QR URL is fetched and parsed (`title`, `og:title`, description, JSON-LD Product name/brand/gtin). Timeout is 2 seconds. Successes stay in `index/external_cache`. Unknown codes are negative cache for 7 days, so a repeat scan does not call the network. GS1 is not called. Transport errors are not cached. If the lookup is slow, empty, or not a single confident catalog match, OCR and SigLIP still run.
3. Before OCR, the dominant bottle/label is cropped (OpenCV). A partial second bottle is ignored. The crop is in `_debug.crop`.
4. RapidOCR (ONNX, Cyrillic mobile, one pass, max side 800) builds a shortlist. SigLIP ranks only that shortlist. Weak OCR falls back to the full catalog. Embeddings stay in memory. CUDA is used when PyTorch reports it; this machine's torch build is CPU-only.
5. EasyOCR loads only if RapidOCR fails to start.
6. The leader is `found` only when a barcode/QR hit, or OCR locked a single wine and the score is at least 0.55, or an OCR shortlist leader leads the next wine by at least 0.08 (score at least 0.58), or a full-catalog leader leads by at least 0.20 (score at least 0.75). A color or grape contradiction with readable OCR forces `not_in_catalog`. A confident hit does not call the web. Otherwise the label text is looked up on the web, and our catalog is ranked by the SigLIP text tower on grape, taste, and description. If that page fails, leftover OCR tokens are ranked the same way. The label photo is not the neighbor score.

Warm timing is the second `recognize()` in one process, after models are loaded. Cold (first call in that process) is slower. The budget for OCR + SigLIP on the warm call is under 3 seconds. Code lookup time is separate and does not replace recognition.

Measured on this machine (PyTorch CPU, CUDA not available) for `data/real_photo/84.73_07-09-2026_11-07-51.webp`:

| pass | codes | ocr | visual | ocr+visual | total |
| --- | --- | --- | --- | --- | --- |
| cold | 0.470s | 0.764s | 0.578s | 1.342s | 2.104s |
| warm | 0.409s | 0.489s | 0.317s | 0.806s | 1.256s |

Warm result: slug `zhemchuzhnaya-9-aligote-czitron`, method `ocr_visual`, `visual_compared` 1, `status` `found`.

A later warm pass on the same photo, after the found / not_found split, was total 1.150s (ocr 0.419s + visual 0.288s), still `found` on that slug. OCR+visual stayed under 3 seconds. EasyOCR is not the default.

Check a warm split from the repo root:

```powershell
.\venv\Scripts\python.exe -c "import time; from pathlib import Path; from PIL import Image; import sys; sys.path.insert(0, 'src'); from hybrid_recognizer import HybridWineRecognizer; r = HybridWineRecognizer(); im = Image.open(r'data/real_photo/84.73_07-09-2026_11-07-51.webp'); a = r.recognize(im); b = r.recognize(im); print('cold', a.get('timing'), a.get('slug')); print('warm', b.get('timing'), b.get('slug'), b.get('method'), b.get('visual_compared'))"
```
