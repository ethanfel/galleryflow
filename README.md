# GalleryFlow

A self-contained, server-side PornPics gallery browser and downloader. The complete interface runs in a normal web browser; PyQt and a desktop client are not required.

## Highlights

- Browse the current PornPics catalog, search, paste a category URL, and keep loading additional pages into one portrait gallery grid.
- Green complete, blue partial, and red ignored states, scoped correctly per profile.
- Open original-resolution images in a full-screen lightbox with zoom and keyboard navigation, then download a whole gallery or select individual images in their original order.
- Find hard-to-name poses from your own examples with high-precision pose ranking: RTMO-L supplies confidence-aware multi-person geometry, strict person-count and body/limb evidence gates remove sparse false positives, spatial DINOv2 handles uncertain or badly cropped detections as a lower-priority fallback, and perceptual hashes recognize true source-image matches.
- Switch Finder to Tag search when explicit JoyTag concepts are better signals than body geometry: analyze one or more reference images, require every useful tag, optionally exclude false-positive tags, and retain only images that satisfy the complete rule.
- Search a persistent local gallery index before exploring the live source, reusing previously computed descriptors without downloading or running the models again.
- Continue an exhausted Finder scan from another PornPics source while retaining cumulative candidates and reviews and skipping galleries already completed in that scan.
- Refine each pose from reviewed results: curate the exact gallery images that are genuinely correct, keep uncertain galleries neutral, reject false matches, and apply pose-only score refinement plus strong-negative vetoes to future scans without retraining the vision models.
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

For an NVIDIA GPU, use the CUDA image. The NVIDIA Container Toolkit must already make `--gpus all` available to Docker:

```bash
docker pull ghcr.io/ethanfel/galleryflow:gpu
docker run -d --name galleryflow --restart unless-stopped \
  --gpus all \
  -p 8100:8099 \
  -v galleryflow-data:/data \
  -v /path/to/existing/library:/library \
  -e PORNPIC_WEBUI_SORT_ROOT=/library \
  -e PORNPIC_WEBUI_FINDER_EXECUTION_PROVIDER=auto \
  ghcr.io/ethanfel/galleryflow:gpu
```

From a checkout, the equivalent Compose command layers the GPU override over the normal ports, volumes, and environment:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

`auto` prefers CUDA for DINOv2, RTMO-L, and JoyTag and falls back to CPU if CUDA cannot initialize. Use `cuda` when a missing GPU should be treated as an error, or `cpu` to force CPU inference.

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

There must be no space after either bind-mount colon. The first Pose search needs outbound HTTPS access and downloads the pinned 85 MB DINOv2-S model plus the pinned 168 MB RTMO-L model into `/data/models`; the persistent data mount caches both. For an offline container, pre-provision the verified files at `/data/models/dinov2-small.onnx` and `/data/models/rtmo-l.onnx`.

Finder uses the FP32 [DINOv2-S ONNX conversion](https://huggingface.co/onnx-community/dinov2-small-ONNX/tree/08c606e3123472a388efa59181b677d428f69bbd/onnx) pinned at revision `08c606e3123472a388efa59181b677d428f69bbd` and verifies SHA-256 `6266c3cd72db6953cecdcbfeab9422a9f783d96f1a4e296ba70ffbac43b54a18` before loading it. The upstream model is Apache-2.0 licensed; see the [Meta DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

Pose geometry uses OpenMMLab RTMO-L body7's official ONNX SDK artifact. GalleryFlow pins and verifies both the archive and its exact `end2end.onnx` member before publishing the model into the persistent cache; zip paths, links, oversized members, and hash mismatches are rejected.

Tag search uses [fancyfeast/joytag](https://huggingface.co/fancyfeast/joytag) pinned at revision `6b7f16331a6ccf0fdce37d5a9564715f6e772b22`. GalleryFlow verifies the approximately 350 MB `model.onnx` with SHA-256 `f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097` and `top_tags.txt` with SHA-256 `32b1963a234af848643b2bbf47d8eff1f1c7889406810c57b980f41b2b9e01d0`, then stores them under `/data/models/joytag`. The same `PORNPIC_WEBUI_FINDER_EXECUTION_PROVIDER` and `PORNPIC_WEBUI_FINDER_INFERENCE_BATCH_SIZE` settings used by the other Finder models also control JoyTag. The model is public: no Hugging Face token, additional bind mount, or container restart is required.

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

Open a gallery and switch it to **Pose dataset** mode. Check one thumbnail and press **Set solo control**, **Set couple control**, or **Set group control** to assign that control directly without opening the full-size viewer. Then check one or more target images, choose their primary pose tag and the appropriate control role, and press **Apply targets** (or the bulk Apply button). Pose tags are reusable across profiles and can be created directly from the gallery; each gallery draft remains scoped to its selected profile. The revisioned draft is saved on the server as you work, so closing the modal or browser does not discard it.

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

## Finder

### Pose search

Open **Finder**, type or paste the path of any example folder inside the configured library root, select or create its pose tag, and enter the PornPics page from which scanning should begin. Both library-relative paths and full container paths under that root are accepted; spaces and hyphens are preserved. The page can be the home page, a search result, a category, or another supported browse page. Set the number of pages and the minimum match score, then start the background scan.

Finder opens every listed gallery and compares every preview image against all examples, including mirrored examples. It ranks evidence in four fixed tiers: exact source-image match, high-precision RTMO pose match, DINOv2 visual-layout fallback when strict pose evidence is incomplete, then high-confidence pose mismatch. A higher tier always ranks before a lower tier, regardless of its numeric score. Within the pose tiers, the displayed score is joint geometry; within the fallback tier, it is DINOv2 layout similarity. The minimum score filter is applied inside each evidence tier.

A high-precision pose match must detect the same number of people as the example and match every person with meaningful body evidence. Each matched person needs at least seven common body joints, shoulder and hip anchors, three distal limb joints, 60% body coverage, three supported joint articulations, evidence from both an arm and a leg, and adequate detector confidence. Evidence uses RTMO's 0.15 visible-joint threshold. The strict score combines normalized body shape with shoulder, elbow, hip, and knee angles, includes relative person layout for couples and groups, and must reach 0.68. Face-only detections, upper-body-only fragments, materially different limb configurations, a strong first person hiding a weak second person, and solo/couple/group count mismatches therefore cannot enter the pose-match tier. Incomplete detections may still appear as lower-priority visual fallbacks when DINOv2 finds a similar layout; fully observed geometry that disagrees is ranked below those fallbacks as a pose mismatch.

New reference and preview analyses are submitted to the vision models in batches of 8 by default. Set `PORNPIC_WEBUI_FINDER_INFERENCE_BATCH_SIZE=1` to disable inference batching, or raise it up to 64 when the GPU has enough memory; larger CUDA batches use more GPU memory. Descriptor cache hits bypass model inference entirely. `PORNPIC_WEBUI_FINDER_NETWORK_WORKERS` is separate and controls concurrent preview downloads, not the inference batch size.

A gallery is flagged when its strongest image crosses the threshold, and its three strongest images are retained using the same tier-first order. Each result labels the evidence tier and shows visual-layout, pose, and exact-match diagnostics, detected person count, and a skeleton overlay when pose output is reliable. Scores are calibrated ranking signals, not statistical probabilities, and images within a gallery are never averaged together.

### Tag search

Switch Finder from **Pose search** to **Tag search**, then enter any image folder under the configured library root. One reference image is sufficient; add more only when they represent the same concept. Press **Analyze** to run JoyTag over the folder. The compact analysis area can be collapsed while scanning and expanded to inspect the suggested tags, aggregate confidence, and the score on every reference image.

Build the query with one or more **Required** tags and optional **Excluded** tags. For example, requiring `pov` and `paizuri` means both signals must reach the required-confidence threshold on the same image. If any excluded tag reaches the separate reject-confidence threshold, that image is discarded. Exclusion is evaluated per image, so an unrelated frame elsewhere in the gallery does not hide a valid candidate. The full 5,813-tag catalog is searchable even when an excluded tag is correctly absent from the strongest reference signals.

Tag search classifies source previews in the same bounded batches as Pose search. A gallery is proposed only when at least one image satisfies every required tag without triggering an exclusion. Candidates are ranked by their weakest required signal, preventing one very strong tag from hiding another weak one. GalleryFlow retains the best one, two, or three qualifying images for manual review; it never pads a result with rejected or lower-scoring images merely to display three. The final accepted, rejected, or **Maybe** decision remains manual, and the complete gallery is still available when the suggested frame is adjacent to a better one.

Every successful JoyTag inference writes its complete 5,813-score vector to a separate SQLite cache as unsigned 8-bit values. That is approximately 5.8 KiB per image before SQLite overhead. The cache lives with the database under persistent `/data`, survives container restarts, and is reused directly when another scan changes its required tags, excluded tags, or thresholds. Once a vector exists, every later query is only a local threshold comparison: adding query terms does not duplicate image data, download the preview, or run JoyTag again. This cache is independent of DINO descriptors, RTMO geometry, and pose-feedback learning, so Tag reviews cannot alter Pose-search refinement.

An ordinary Tag search never backfills older uncached corpus images implicitly. It searches the vectors already available in the Local Gallery Index, skips missing vectors, and then explores the selected Source URL. The Tag-search setup shows exact cached/total coverage and provides an explicit **Index local corpus** job when you want to classify the missing legacy images once. That background job is batched, persistent across restarts, reports progress and bounded failures, can be canceled without discarding completed vectors, and resumes future searches from the saved cache. Images encountered during live source exploration continue to save both their gallery link and JoyTag vector automatically.

Results remain ordinary galleries. Pending and **Maybe** results open with the scanner suggestions highlighted in Pose dataset mode. Accepted and rejected results open in **Finder review**, where the saved feedback subset can be edited against the complete gallery. Every Finder-opened gallery now keeps a review rail above the workspace: **Accept**, **Maybe**, and **Reject** save the decision without closing the gallery, while Previous/Next (or the left/right arrow keys) move through the current filtered review queue. The queue remains stable while decisions remove cards from the underlying tab, and pose drafts finish saving before navigation. GalleryFlow prefetches the adjacent gallery details and up to three low-priority previews per neighbor, so moving backward or forward normally reuses the warmed data instead of repeating the source request. The cache is profile-aware, bounded, and disabled for preview images when the browser reports Data Saver, a 2G connection, or a hidden tab. This makes it possible to review, assign controls and targets, then continue without returning to the result grid after every gallery.

Accepted results also provide **Prepare controls & targets**, which carries the selected targets and pose tag into Pose dataset mode without assigning anything automatically. You can then find the related solo, couple, or group control in the same gallery and explicitly apply or export the pair. Check one thumbnail and press **Set solo control**, **Set couple control**, or **Set group control** to assign it directly; the full-size viewer is optional. Accepting, rejecting, or marking a Finder result Maybe never downloads, globally ignores, or silently tags a gallery.

The three proposed matches on each Pose-search result are independently checkable and start checked. Uncheck an incorrect image before reviewing the gallery: only checked images are saved as pose feedback. The full-gallery **Finder review** tab is available before the first decision, so the rail can Accept a better adjacent frame immediately. Finder-feedback checks are deliberately separate from the temporary Pose dataset checks: assigning or clearing a control/target cannot change what Finder learns. After accepting or rejecting, edit the feedback checks and use **Save selection**. GalleryFlow attempts pose analysis for newly chosen images, but your manual selection is always saved even when the model or media is temporarily unavailable or RTMO cannot confirm high-precision pose evidence. Such images are labeled **Saved · pose pending**, remain available for control/target preparation, and are currently excluded from automatic Finder reranking; a later successful pose analysis can make them ranking-eligible. **Accept** requires at least one selected image.

**Maybe** is the neutral state for a gallery that is close or uncertain: it is removed from the review queue but supplies neither positive nor negative feedback. A **Reject** with selected images teaches negative pose feedback. A Reject with no selected images records a definite rejected review without teaching a negative example. The newest review for the same pose and gallery replaces its older feedback across scans, so changing a previously accepted or rejected gallery to Maybe truly neutralizes it for future scans.

Feedback is scoped to the pose tag and uses stored RTMO geometry only; it does not train RTMO or DINOv2, and appearance similarity never becomes a feedback signal. Two distinct galleries with high-precision pose detections activate either the accepted or rejected side independently. Normal positive and negative adjustments are capped at ±0.08 and reorder results inside their pose lane. In `pose-precision-v2`, repeated strong negatives can also veto a known false-positive shape: when at least two independently rejected galleries agree closely with a candidate and that negative affinity clearly exceeds both the original examples and active accepted feedback, the candidate is demoted out of the pose-match tier. Exact-image matches and visual fallbacks are never vetoed this way. Every new scan snapshots the current feedback revision, so later reviews or a reset cannot change that scan midway through a pause, resume, or extension.

The Finder setup panel shows accepted/rejected sample counts, usable-gallery progress, and a pose-specific reset action. Resetting removes learned feedback without deleting galleries, cached descriptors, or historical review states. On upgrade, existing accepted and rejected results are migrated from their retained top matches; usable pose data already present in the current cache is preserved automatically, while missing or non-precise pose snapshots simply do not count toward activation.

Before live exploration begins, every new scan compares its examples against the persistent local corpus of all previously indexed galleries. Local results are not limited to the selected category; the source URL controls which additional galleries are explored afterward. As Finder opens each live gallery, it records all image-to-descriptor associations, so later poses can search retained descriptors without network or model inference. Existing descriptor rows survive the corpus migration. Historical top-three matches are linked immediately as partial galleries, while revisiting those galleries completes their index using the already cached descriptors wherever possible. Older cached images outside those persisted top-three results cannot be assigned to a gallery retroactively; they remain cached and are linked when Finder revisits their gallery.

Scans, corpus mappings, results, and pose feedback live in SQLite under `/data` and survive browser or container restarts. Keep the same `/data` volume or bind mount when replacing the container; this feature requires no additional Docker mount. Current `pose-precision-v2` scans can be paused, resumed, canceled, and filtered by review state or match score. Finder results are server-paged 24 galleries at a time: changing the review tab, score threshold, or page queries only that slice from SQLite instead of loading every candidate into the browser. Results from earlier appearance-first and `pose-first-v1` scans remain viewable after upgrading, but cannot be resumed or extended; start a new scan to avoid mixing ranking systems. Category, tag, and pasted `/?q=...` search scans follow PornPics' JSON infinite-scroll cursor beyond the first 20 HTML cards; search terms are preserved when the HTML page switches to the JSON endpoint, and repeated gallery IDs are skipped automatically. When a completed or paused current scan still has another page in its current source, **Extend search** adds up to 50 pages to that same scan and resumes its saved cursor without revisiting completed galleries or losing review decisions. When a completed scan has exhausted that source, **Explore another source** accepts a new PornPics home, search, category, gallery, or other supported browse URL and up to 50 additional pages, then uses **Continue same scan** to accumulate deduplicated results while preserving the scan ID, existing candidates and reviews, reference snapshot, and feedback snapshot. Continuations survive restarts, and one cumulative scan is capped at 500 completed pages.

Versioned spatial descriptors, hashes, and pose analyses are cached separately from the corpus mappings. A current spatial descriptor uses about 32 KiB plus pose metadata; typical total storage is roughly 33–36 KiB per image before SQLite overhead. The least-recently-used descriptor cache is bounded to 50,000 entries or 2 GiB by default, while corpus mappings remain available when a descriptor is pruned or a model version changes. The WebUI reports complete and partial galleries, reusable images, and cache usage. The existing `PORNPIC_WEBUI_FINDER_CACHE_MAX_ENTRIES` and `PORNPIC_WEBUI_FINDER_CACHE_MAX_BYTES` settings can raise the retention limit when more `/data` storage is available.

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
| `PORNPIC_WEBUI_FINDER_POSE_MODEL_PATH` | `<data-dir>/models/rtmo-l.onnx` | Cached or pre-provisioned RTMO-L ONNX model |
| `PORNPIC_WEBUI_FINDER_JOYTAG_MODEL_PATH` | `<data-dir>/models/joytag/model.onnx` | Optional cached or pre-provisioned JoyTag ONNX model path |
| `PORNPIC_WEBUI_FINDER_JOYTAG_TAGS_PATH` | `<data-dir>/models/joytag/top_tags.txt` | Optional cached or pre-provisioned JoyTag vocabulary path |
| `PORNPIC_WEBUI_FINDER_EXECUTION_PROVIDER` | `auto` | `auto`, `cuda`, or `cpu` inference provider preference |
| `PORNPIC_WEBUI_FINDER_POSE_ENABLED` | `true` | Enable RTMO pose diagnostics and geometry scoring |
| `PORNPIC_WEBUI_FINDER_WORKERS` | `1` | Concurrent background Finder scans (maximum 2) |
| `PORNPIC_WEBUI_FINDER_NETWORK_WORKERS` | `3` | Concurrent Finder preview requests (maximum 8) |
| `PORNPIC_WEBUI_FINDER_INFERENCE_BATCH_SIZE` | `8` | Maximum new Finder images submitted together to batch-capable vision models (maximum 64) |
| `PORNPIC_WEBUI_FINDER_REQUEST_DELAY` | `0.15` | Minimum delay between Finder network requests in seconds |
| `PORNPIC_WEBUI_FINDER_MAX_EXAMPLES` | `500` | Maximum reference images in one example folder |
| `PORNPIC_WEBUI_FINDER_MAX_GALLERY_IMAGES` | `2000` | Maximum images scored in one source gallery |
| `PORNPIC_WEBUI_FINDER_MAX_IMAGE_BYTES` | `12582912` | Per-image Finder byte ceiling |
| `PORNPIC_WEBUI_FINDER_MAX_IMAGE_PIXELS` | `40000000` | Per-image Finder decoded-pixel ceiling |
| `PORNPIC_WEBUI_FINDER_CACHE_MAX_BYTES` | `2147483648` | Maximum persistent descriptor-cache bytes before LRU pruning |
| `PORNPIC_WEBUI_FINDER_CACHE_MAX_ENTRIES` | `50000` | Maximum persistent descriptor-cache entries before LRU pruning |
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
│   ├── dinov2-small.onnx
│   ├── rtmo-l.onnx
│   └── joytag/
│       ├── model.onnx
│       └── top_tags.txt
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
