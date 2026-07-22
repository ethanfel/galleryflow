# GalleryFlow

A self-contained, server-side PornPics gallery browser and downloader. The complete interface runs in a normal web browser; PyQt and a desktop client are not required.

## Highlights

- Browse the current PornPics catalog, search, paste a category URL, and keep loading additional pages into one portrait gallery grid.
- Green complete, blue partial, and red ignored states, scoped correctly per profile.
- Open original-resolution images in a full-screen lightbox with zoom and keyboard navigation, then download a whole gallery or select individual images in their original order.
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

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORNPIC_WEBUI_DATA_DIR` | `./data` | SQLite and application state |
| `PORNPIC_WEBUI_DOWNLOAD_ROOT` | `./data/downloads` | Profile libraries |
| `PORNPIC_WEBUI_SORT_ROOT` | download root | Highest server folder exposed to the visual sorter |
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
└── downloads/
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
