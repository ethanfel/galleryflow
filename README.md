# GalleryFlow

A self-contained, server-side PornPics gallery browser and downloader. The complete interface runs in a normal web browser; PyQt and a desktop client are not required.

## Highlights

- Browse the current PornPics catalog, search, paste a category URL, and keep loading additional pages into one portrait gallery grid.
- Green complete, blue partial, and red ignored states, scoped correctly per profile.
- Open original-resolution images in a full-screen lightbox with zoom and keyboard navigation, then download a whole gallery or select individual images in their original order.
- Find hard-to-name poses from your own examples: a local DINOv2 vision model scans complete galleries, flags the gallery when any image is similar, and ranks the results by their best matching image.
- Build training-ready pose pairs while browsing: assign one solo, couple, or group control to each target, add a pose tag, and export matched target/control folders with shared IDs.
- Automatic profile folders with safe, server-controlled paths.
- Persistent queue with live progress, cancellation, per-image results, retries for transient failures, and restart recovery.
- Global ignore/unignore, hide-saved and hide-ignored filters, history, profile management, and responsive mobile/desktop layouts.
- Integrated visual sorter with timestamp or filename matching, reusable setups, ranked previews, paired IDs, collision-safe moves, keyboard shortcuts, and persistent undo.
- SQLite persistence, signed same-origin image proxying, strict upstream host checks, download size limits, image validation, and atomic `.part` files.
- Compatibility endpoints for the old client and an idempotent legacy importer for history, ignores, profiles, and usable sorter presets.

Only download material you are legally permitted to access and retain. Site availability and terms remain outside this application.

## Quick start

Python 3.11 or newer is recommended.

```bash
cd /media/unraid/davinci/Qwen_edit_lora/galleryflow
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run.py --host 0.0.0.0 --port 8099
```

Open `http://192.168.1.3:8099` from another device on the LAN. Run the command from this directory so the application and static assets resolve consistently.

If the legacy gallery server is still using port `8099`, keep it running side by side with `python run.py --host 0.0.0.0 --port 8100`, then open `http://192.168.1.3:8100`. Stop the legacy service before moving this version back to `8099`.

The current environment already has the required packages, so during development this is enough:

```bash
cd /media/unraid/davinci/Qwen_edit_lora/galleryflow
python run.py
```

To sort an existing server-side library, point the sorter at its common parent before starting the server. The browser can only see folders below this configured root; it never needs a local desktop client or file picker.

```bash
export PORNPIC_WEBUI_SORT_ROOT=/media/unraid/davinci/Qwen_edit_lora/pornpic
python run.py --host 0.0.0.0 --port 8099
```

## Docker

Build from the checkout:

```bash
docker compose up -d --build
```

Or use the public GitHub Container Registry image:

```bash
docker pull ghcr.io/ethanfel/galleryflow:latest
docker run -d --name galleryflow --restart unless-stopped \
  -p 8100:8099 -v galleryflow-data:/data \
  ghcr.io/ethanfel/galleryflow:latest
```

State and downloads are stored in the Docker-managed `galleryflow-data` volume. To place the library directly on another disk, replace that volume with a bind mount such as `/path/to/library:/data` and make the host directory writable by container UID/GID `10001:10001`.

An existing sort library can instead be mounted separately:

```yaml
services:
  galleryflow:
    volumes:
      - galleryflow-data:/data
      - /path/to/existing/library:/sort-library
    environment:
      PORNPIC_WEBUI_SORT_ROOT: /sort-library
```

Sorter decisions move target images and create control copies, so the sort-library mount must be writable.

Pose-pair exports default to `<sort-root>/pose_pairs`. To keep datasets on a separate disk, mount a writable output directory and set it explicitly:

```yaml
services:
  galleryflow:
    volumes:
      - galleryflow-data:/data
      - /path/to/pose-datasets:/pose-output
    environment:
      PORNPIC_WEBUI_POSE_ROOT: /pose-output
```

Both `/data` and every mounted download, sorter, or pose-output directory must be writable by container UID/GID `10001:10001`. Do not include a space after the host path's colon in a Docker `-v` specification: use `/host/path:/pose-output`, not `/host/path: /pose-output`.

Pose Finder can read any image folder already inside the mounted library. With this existing library mapping:

```bash
docker run -d --name galleryflow --restart unless-stopped \
  -p 8100:8099 \
  -v galleryflow-data:/data \
  -v /path/to/existing/library:/library \
  -e PORNPIC_WEBUI_SORT_ROOT=/library \
  ghcr.io/ethanfel/galleryflow:latest
```

You can paste either `sorted_outpaint/mating press - backview/selected_target_upscaled` or the full container path `/library/sorted_outpaint/mating press - backview/selected_target_upscaled` into Finder. No specially named folder, additional mount, or container restart is required. Finder only reads its examples; the files need read permission and their directories need traverse permission for UID `10001`.

If you previously added `PORNPIC_WEBUI_FINDER_EXAMPLES_ROOT=/references` or pointed it at a special examples folder, remove that variable once so Finder defaults to the complete `PORNPIC_WEBUI_SORT_ROOT`. Keeping the variable intentionally confines Finder to that older root.

There must be no space after either bind-mount colon. The first scan needs outbound HTTPS access to Hugging Face and downloads the pinned 85 MB DINOv2-S ONNX model into `/data/models`, where the persistent data mount caches it for later scans. For an offline container, pre-provision the verified model at `/data/models/dinov2-small.onnx` instead.

Finder uses the FP32 [DINOv2-S ONNX conversion](https://huggingface.co/onnx-community/dinov2-small-ONNX/tree/08c606e3123472a388efa59181b677d428f69bbd/onnx) pinned at revision `08c606e3123472a388efa59181b677d428f69bbd` and verifies SHA-256 `6266c3cd72db6953cecdcbfeab9422a9f783d96f1a4e296ba70ffbac43b54a18` before loading it. The upstream model is Apache-2.0 licensed; see the [Meta DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

## Import the old history

Always preview first:

```bash
python migrate_legacy.py --legacy-dir ../pornpic --dry-run
python migrate_legacy.py --legacy-dir ../pornpic
```

The importer reads the legacy SQLite database (including its WAL snapshot) and history text files, deduplicates by PornPics gallery ID, preserves profiles and global ignores, and never deletes or rewrites the legacy sources. It is safe to run more than once.

To also import `sorter_profiles.json`, configure the sort root so those old absolute paths are beneath it. Preview with the same environment setting you will use for the server:

```bash
export PORNPIC_WEBUI_SORT_ROOT=/media/unraid/davinci/Qwen_edit_lora/pornpic
python migrate_legacy.py --legacy-dir ../pornpic --dry-run
python migrate_legacy.py --legacy-dir ../pornpic
```

Missing legacy target folders skip only that preset. Missing or moved control folders are reported and omitted without aborting the rest of the import.

## Visual sorter

Open **Sort** in the WebUI, choose a target folder, and either select reference folders or let time mode discover the target's sibling folders automatically.

- **Time** ranks control images whose modification times are within the configured threshold (50 seconds by default), closest first.
- **Filename** reproduces the older exact, case-sensitive match on the filename stem before the first underscore.
- **Match** copies the reference to `selected_control/` and moves the target to `selected_target/`.
- **Solo** uses `control_selected_solo_woman/` and `selected_target_solo_woman/`.
- **No match** and **Skip** move the target to `selected_target_no_control/` and `skipped_target/` respectively.

Optional `id001_` prefixes keep target/control pairs together. Existing filenames are never overwritten: a `_copyN` suffix is allocated when needed. Sessions and undo history live in SQLite, so the last decision can still be undone after a browser or server restart. The legacy `Z`, `N`, and `S` shortcuts remain available when focus is not inside a form control.

Every sorter move is journaled before files change. If the server stops partway through an action or undo, the next startup/session read reconciles the operation automatically when its state is unambiguous. If both the source and destination exist, neither copy is deleted: the session remains marked **Recovering** until an operator resolves the duplicate and reloads or rescans. Starting another session for that target is blocked while recovery is pending.

## Pose-pair workflow

Open a gallery and switch it to **Pose dataset** mode. Mark at most one gallery image as the control for each available role: **solo**, **couple**, or **group**. Then select one or more target images, choose their primary pose tag and the appropriate control role, and apply the annotation. Pose tags are reusable across profiles and can be created directly from the gallery; each gallery draft remains scoped to its selected profile. The revisioned draft is saved on the server as you work, so closing the modal or browser does not discard it.

Each target has exactly one pose tag and uses exactly one of the gallery's three control alternatives. Before export, GalleryFlow reports incomplete targets whose selected role has no control. **Download & organize** snapshots the current draft and sends the work through the persistent job queue, where progress, cancellation, and restart recovery behave like gallery downloads.

Pairs are grouped by safe pose slug and use a shared deterministic ID:

```text
<pose-root>/
└── <pose-slug>/
    ├── selected_target/
    │   └── g<gallery-id>-<ordinal>_target.jpg
    └── selected_control/
        └── g<gallery-id>-<ordinal>_control.jpg
```

Gallery URLs and output paths are validated, symlinks and path traversal are rejected, and an existing file is never silently replaced. Re-exporting the same unchanged pair is idempotent. Changed content, a different role/control, or moving the same gallery image to another pose reports a conflict instead of overwriting the dataset. A failed or canceled multi-pair export rolls back only the files newly created by that job.

## Pose Finder

Open **Finder**, type or paste the path of any example folder inside the configured library root, select or create its pose tag, and enter the PornPics page from which scanning should begin. Both library-relative paths and full container paths under that root are accepted; spaces and hyphens are preserved. The page can be the home page, a search result, a category, or another supported browse page. Set the number of pages and the minimum similarity, then start the background scan.

Finder opens every listed gallery and compares all of its preview images against all examples in the selected folder, including mirrored examples. A gallery is flagged when at least one image crosses the threshold. Its score is exactly the best individual-image cosine similarity; images from a gallery are not averaged together. Scores are useful for ranking candidates but are not statistical probabilities.

Results remain ordinary galleries. **Open gallery** loads the complete gallery in Pose dataset mode, checks and highlights the matching image, and fills in the pose tag. You can then find the related solo, couple, or group control image in the same gallery and explicitly apply or export the pair. Accepting or rejecting a Finder suggestion only changes that scan's review state; it never downloads, globally ignores, or silently tags a gallery.

Scans and results live in SQLite and survive browser or container restarts. They can be paused, resumed, canceled, and filtered by review state or similarity. Image embeddings are cached, so rescanning an unchanged image avoids another model pass.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORNPIC_WEBUI_DATA_DIR` | `./data` | SQLite and application state |
| `PORNPIC_WEBUI_DOWNLOAD_ROOT` | `./data/downloads` | Profile libraries |
| `PORNPIC_WEBUI_SORT_ROOT` | download root | Highest server folder exposed to the visual sorter |
| `PORNPIC_WEBUI_POSE_ROOT` | `<sort-root>/pose_pairs` | Training-ready pose-pair datasets |
| `PORNPIC_WEBUI_FINDER_EXAMPLES_ROOT` | sort root | Optional override for the highest folder Finder may read; normally unnecessary |
| `PORNPIC_WEBUI_FINDER_MODEL_PATH` | `<data-dir>/models/dinov2-small.onnx` | Cached or pre-provisioned DINOv2-S ONNX model |
| `PORNPIC_WEBUI_FINDER_WORKERS` | `1` | Concurrent background Finder scans (maximum 2) |
| `PORNPIC_WEBUI_FINDER_NETWORK_WORKERS` | `3` | Concurrent Finder preview requests (maximum 8) |
| `PORNPIC_WEBUI_FINDER_REQUEST_DELAY` | `0.15` | Minimum delay between Finder network requests in seconds |
| `PORNPIC_WEBUI_FINDER_MAX_EXAMPLES` | `500` | Maximum reference images in one example folder |
| `PORNPIC_WEBUI_FINDER_MAX_GALLERY_IMAGES` | `2000` | Maximum images scored in one source gallery |
| `PORNPIC_WEBUI_FINDER_MAX_IMAGE_BYTES` | `12582912` | Per-image Finder byte ceiling |
| `PORNPIC_WEBUI_FINDER_MAX_IMAGE_PIXELS` | `40000000` | Per-image Finder decoded-pixel ceiling |
| `PORNPIC_WEBUI_JOB_WORKERS` | `2` | Concurrent gallery jobs |
| `PORNPIC_WEBUI_IMAGE_WORKERS` | `6` | Global concurrent image requests |
| `PORNPIC_WEBUI_REQUEST_TIMEOUT` | `25` | Browse timeout in seconds |
| `PORNPIC_WEBUI_IMAGE_TIMEOUT` | `45` | Image timeout in seconds |
| `PORNPIC_WEBUI_MAX_IMAGE_BYTES` | `83886080` | Per-image byte ceiling |
| `PORNPIC_WEBUI_MEDIA_KEY` | random at startup | Optional stable proxy-signing secret |
| `PORNPIC_WEBUI_SQLITE_VFS` | `unix-dotfile` on Linux | SQLite locking mode; set `default` for local-disk WAL |

Concurrency and theme can also be adjusted in the WebUI. A changed gallery-worker count takes effect after restart; image concurrency and request timeout apply immediately.

## Data layout

```text
data/
├── pornpic_webui.sqlite3
├── models/
│   └── dinov2-small.onnx
└── downloads/
    ├── pose_pairs/
    │   └── <pose-slug>/
    │       ├── selected_target/
    │       └── selected_control/
    ├── Default/
    └── <profile>/
        └── <safe-title>--<gallery-id>/
            ├── 0001.jpg
            └── ...
```

Profile names and gallery titles never become arbitrary paths. All resolved destinations are checked to remain below the configured download root. Renaming a profile changes its display name without moving its directory.

The sorter root may be separate from this tree. Only supported image files directly inside a selected target/control folder are queued. Output folders are ignored during automatic sibling discovery, and symlinked folders/files are not scanned.

## API and operations

- Interactive API documentation: `/docs`
- Health check: `/api/health`
- Live job events: `/api/events` (server-sent events)
- Queue reconciliation: `/api/downloads`

This service is intended for a trusted LAN. If it is exposed outside the LAN, put it behind an authenticated HTTPS reverse proxy; the application intentionally does not ship a user-account system.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
