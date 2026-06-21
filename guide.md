# Trash-Track / Colby-Trash-App — Complete Codebase Guide

A from-scratch walkthrough of how this web app works, written for someone who knows
Python and some HTML/CSS but has never built or deployed a backend. Read it top to
bottom once; after that, use it as a reference.

---

## PART 1 — THE MENTAL MODEL (read this first)

### Client and server
Every web app is a conversation between two computers:

- **Client** = the user's browser (Chrome, Safari). It shows the interface and *sends requests*.
- **Server** = a computer running your app's code. It *receives requests, does work, sends responses*.

A **request** is the browser asking for something ("give me the map", "process these images").
A **response** is the server's answer (a web page, or some data). The whole app is just this
request → work → response loop repeated.

The website you built before (HTML/CSS only) was **client-side only** — files the browser
displayed, with no server doing work. This app is different: it has a server that runs ML models.

### Frontend vs backend vs full-stack
- **Frontend** = what runs in the browser: HTML (structure), CSS (styling), JavaScript (behavior). What the user sees and clicks.
- **Backend** = what runs on the server: the Python, the models, the logic, the data. The user never sees it directly — only its results.
- **Full-stack** = an app with both. **Yours is full-stack.**

The key fact about THIS app: when a user uploads a drone image, the **frontend** sends it to the
**backend**, the backend runs Grounding DINO + CLIP (heavy computation), and sends results back.
**The models run on the server, not in the browser.** That single fact is why the app needs real
compute and can't be hosted as a "frontend-only" site.

### What "running locally" means
Right now the app runs on your Mac at `localhost:8000`. **localhost = this computer.** Nobody else
can reach it. **Deploying** = putting it on an internet-connected server so others can use it.

---

## PART 2 — THE TECH STACK (the tools the app is built from)

| Tool | What it is | Role in this app |
|---|---|---|
| **Python** | Programming language | The whole backend is Python |
| **Flask** | A Python *web framework* | Turns Python functions into web endpoints; handles requests/responses |
| **Grounding DINO** | Object-detection model | Finds debris in images (draws boxes) |
| **CLIP** | Image-classification model | Labels each box (plastic, metal, wood…) |
| **SIFT** (OpenCV) | Classical computer-vision algorithm | Finds duplicate objects across overlapping photos |
| **pandas** | Python data library | Holds detections as a table (DataFrame) |
| **Folium** | Map library | Builds the interactive map |
| **scikit-learn** | Machine-learning library | K-means clustering for the plots |
| **HTML/CSS/JavaScript** | Web languages | The frontend (the `templates/` files) |

**Framework** = a toolkit that handles the boring, universal parts (listening for web requests,
routing them, sending responses) so you only write your app's specific logic. Flask is a
"micro" framework — small and simple, good for projects like this.

---

## PART 3 — THE REPOSITORY MAP (every folder and file)

```
Colby-Trash-App/
├── app.py                  ← MAIN backend file: startup + the Home/CSV routes
├── script.py               ← Helper functions: detection filters, EXIF/GPS reading
├── python/
│   └── config.py           ← Creates the Flask app object; holds shared state (the data, the map)
├── routes/
│   ├── remove_overlap.py    ← Backend logic for the Duplicates tab (SIFT)
│   ├── mapping.py           ← Backend logic for the Map tab (Folium)
│   └── plots.py             ← Backend logic for the Plot tab (K-means)
├── templates/              ← THE FRONTEND — one HTML file per tab
│   ├── index.html           ← Home page (upload + detect)
│   ├── csv.html             ← CSV tab (view/download/clear/upload data)
│   ├── overlap.html         ← Duplicates tab
│   ├── map.html             ← Map tab
│   ├── plots.html           ← Plot tab
│   └── about.html           ← About page
├── static/                 ← CSS / images / shared frontend assets
├── GroundingDINO/          ← The detection model's library code (not yours — installed)
├── weights/                ← Trained model weight files (.pth) — the models' "knowledge"
├── detections.csv          ← The OUTPUT data: every detected object (boxes, GPS, class)
├── requirements.txt        ← The list of Python libraries the app needs
└── README.md               ← Setup instructions (note: outdated for modern Mac/Py3.12)
```

**Folder vs file terms:**
- A **module** = a single `.py` file you can import.
- A **package** = a folder of modules (like `python/` and `routes/`).
- `templates/` is special to Flask — it automatically looks here for HTML files to serve.
- `static/` is special to Flask — it serves these files (CSS, images) directly to the browser.

---

## PART 4 — HOW A REQUEST FLOWS (the core loop, concretely)

Take the Home page "detect debris" action, end to end:

1. User opens `index.html` in the browser (frontend). Flask served them this file.
2. User selects images and clicks the detect button.
3. **JavaScript** in `index.html` packages the images and sends them to the backend URL `/process_images` (this is a `fetch()` call — JS's way of making a request).
4. In `app.py`, the function attached to `/process_images` (its **route**) runs. It:
   - loops over each uploaded image
   - runs Grounding DINO (via `GroundingDINO/` + `weights/`) to get boxes
   - runs CLIP to classify each box
   - reads GPS/altitude from the image's EXIF (via `script.py`)
   - appends rows to the data table and saves `detections.csv`
5. The route sends the results back as the **response**.
6. JavaScript in `index.html` receives the response and builds the on-screen cards/dropdowns.

Every tab follows this same shape: **HTML page → JS sends request to a route → Python route does
work → returns response → JS updates the page.** Learn this one loop and the whole app makes sense.

---

## PART 5 — KEY TERMS GLOSSARY (web + Flask)

- **Route / endpoint** — a URL plus the Python function that runs when that URL is requested. Written in Flask as `@app.route('/process_images', methods=['POST'])` above a function.
- **GET vs POST** — two kinds of request. **GET** = "give me something" (loading a page). **POST** = "here's data, do something with it" (uploading images). You'll see `methods=['POST']` on routes that receive data.
- **API** — broadly, the set of endpoints your backend exposes for the frontend to call. Your routes (`/process_images`, `/plotting`, etc.) *are* your API.
- **Request / response** — the question and the answer in the loop above.
- **JSON** — a text format for sending structured data between frontend and backend (looks like a Python dict). The Plot tab sends `{ num_clusters: 5 }` as JSON.
- **Template** — an HTML file Flask serves, optionally with data injected. Lives in `templates/`.
- **render / serve** — the server "serving" a page means sending its HTML to the browser.
- **DataFrame** — pandas' table object. The whole app passes detections around as a DataFrame with 10 columns (see Part 6).
- **EXIF** — metadata embedded in a photo (GPS, altitude, timestamp, camera). The app reads GPS/altitude from here to place objects on the map.
- **Dependency** — an external library your code needs (Flask, pandas…). Listed in `requirements.txt`.
- **Environment** — the OS + Python version + all dependencies, together. The thing that's painful to recreate (which is why Docker exists — see Part 8).
- **localhost / port** — `localhost:8000` means "this computer, channel 8000." A **port** is just a numbered channel so multiple programs can use the network at once.

---

## PART 6 — THE DATA CONTRACT (the 10-column table)

Everything in this app revolves around one table. Every detection is a row with these 10 columns
(defined as `column_titles` in `app.py`):

```
x1, y1, x2, y2, longitude, latitude, altitude, image_name, number, type
```

- `x1,y1,x2,y2` — the bounding box corners in pixels (full-res, 0–5472 wide)
- `longitude, latitude, altitude` — from the image's EXIF GPS
- `image_name` — which photo it came from
- `number` — the object's index within that photo
- `type` — the CLIP class (plastic, metal, wood, cage, fishing gear, nature, wheel)

This table lives in two places:
- **`detections.csv`** on disk (survives restarts)
- **`python.config.csv_file`** in memory (the live copy the running app reads/writes)

**Why this matters:** almost every bug you hit was about this table. The `row[0]` vs `row['x1']`
bugs happened because newer pandas wants you to access a row's value *by column name* (`row['x1']`),
not by position (`row[0]`). Knowing the 10-column contract is the key to reading any route.

---

## PART 7 — FILE-BY-FILE DEEP DIVE

### `python/config.py` — the shared core
Small but central. It does two things:
1. **Creates the Flask application object**: `application = Flask(__name__)`. Every other file
   imports this *same* object (`from python.config import application as app`) so all routes
   attach to one app.
2. **Holds shared state** that any route can read/write:
   - `csv_file` — the in-memory DataFrame of detections
   - `map` — the current Folium map object

Putting shared state here (instead of in `app.py`) lets `routes/mapping.py`, `routes/plots.py`,
etc. all reach the same data via `python.config.csv_file`. This is a simple (if fragile) way to
share state across files. **Note for later:** this in-memory shared state is part of why hosting
is tricky — it assumes one long-running process with memory that persists. (Relevant to the
Render "spin-down / ephemeral filesystem" problem.)

### `app.py` — startup + Home/CSV routes
The biggest file and the entry point. Run order:
1. **Imports** — Flask app, the models, pandas, `script.py` helpers.
2. **Startup block** — defines `column_titles`, loads `detections.csv` into a DataFrame (with the
   fallback logic you fixed so an empty/missing file still gets the 10 columns), loads the
   Grounding DINO + CLIP models into memory *once* at startup (loading them is slow, so it's done
   once, not per request).
3. **Routes** — the functions that handle each URL:
   - `/` → serves `index.html` (the Home page)
   - `/process_images` (POST) → the detection+classification pipeline (see Part 4)
   - `/update_class` → applies a user's dropdown correction to an object's class
   - `/get_results`, `/download_results`, `/upload_csv`, `/clear_results` → the CSV-tab operations
   - `/csv/`, `/map/`, `/plots/`, `/about/`, `/overlap/` → serve those HTML pages
4. **`app.run(...)`** at the bottom — starts the server listening on port 8000.

The device line — `device = "cuda" if torch.cuda.is_available() else "cpu"` — is why it runs on
your Mac (falls back to CPU) and will use a GPU automatically if one exists. **This is exactly the
pattern any GPU-needing addition (like LightGlue) would reuse.**

### `script.py` — detection helpers
Pure functions called by `app.py`. The important ones:
- The **detection filters** — `delete_big`, `delete_rock`, `delete_overlap`, `delete_box` — clean
  up Grounding DINO's raw output (remove boxes that are too big, that are rocks, that overlap, etc.).
- `bb_intersection_over_union` — computes IoU (box overlap), the same metric your `evaluate.py` uses.
- **EXIF/GPS functions** — `get_exif`, `get_lat_lon`, `get_altitude`. These read location from the
  photo. **They raise an error if an image has no GPS** — the robustness gap you noted (Bug 5):
  any non-drone image without EXIF will crash the request unless wrapped in try/except.

### `routes/remove_overlap.py` — the Duplicates tab (SIFT)
Handles `/remove_overlap` (POST). The duplicate-removal you've been comparing against AKAZE.
Flow: re-reads the uploaded images, crops each detected object from the 10-column table, runs SIFT
on each crop, and compares crops that are close in GPS. If two crops share ≥ a threshold of matching
features (Ray used 50), they're judged the same physical object and the duplicate row is removed.
This is the file your `compare_dedup.py` mirrors. (It had the same `row[0]`→`row['x1']` bug.)

### `routes/mapping.py` — the Map tab (Folium)
Handles the map routes. Reads the detections table, drops rows with missing GPS, converts each
box + altitude into a real-world position (using the drone's focal length / sensor width —
hardcoded `13.2, 8.8, 5472`, tuned for the DJI Phantom 4 Pro), and drops a marker per object on a
Folium map. The map is saved to `python.config.map`. (Also had the positional-indexing bug.)

### `routes/plots.py` — the Plot tab (K-means)
Handles `/plotting` (POST). Takes a number of clusters from the frontend (as JSON), runs K-means on
the longitude/latitude to group detections geographically, makes a bar chart of debris types per
cluster (with seaborn/matplotlib), encodes each chart as base64, and returns them. It also adds
cluster-center markers to `python.config.map` — so it depends on the map existing first. This file
was already clean (used column names correctly).

---

## PART 8 — THE FRONTEND (HTML/CSS/JS refresher via your own files)

Each file in `templates/` is one page. They all share the same skeleton:

```html
<!DOCTYPE html>            <!-- "this is HTML5" -->
<html lang="en">
<head>                     <!-- metadata + styling, not visible content -->
  <title>...</title>
  <style> ... CSS ... </style>
</head>
<body>                     <!-- the visible page -->
  <nav> ... links to other tabs ... </nav>
  <header> ... title text ... </header>
  ... page content (upload zones, buttons, tables) ...
  <script> ... JavaScript ... </script>
</body>
</html>
```

**HTML refresher (structure):**
- **Tags** like `<div>`, `<button>`, `<table>`, `<input>` define elements. They nest inside each other.
- **`id`** (`<button id="submitBtn">`) — a unique name JS uses to find an element.
- **`class`** (`<button class="button-36">`) — a reusable name CSS uses to style many elements.
- `<input type="file">` — the file picker. `<table id="results">` — where results get inserted.

**CSS refresher (styling):**
- CSS rules say "for this selector, apply these styles": `.button-36 { color: white; }`.
- `.button-36` targets everything with `class="button-36"`; `#results` targets `id="results"`.
- Your redesign put the theme (colors, fonts, spacing) in `static/theme.css` and shared it across pages.

**JavaScript refresher (behavior) — the part that talks to the backend:**
The `<script>` block is what makes the page interactive. The pattern in every page:
```js
document.getElementById("submitBtn").addEventListener("click", () => {  // when button clicked…
  const formData = new FormData();                  // package the files
  formData.append("img_directory", files[i]);
  fetch("/remove_overlap", { method: "POST", body: formData })  // send to backend route
    .then(response => response.text())              // wait for the answer
    .then(data => { /* update the page with the result */ });
});
```
- **`getElementById`** — find an element by its `id`.
- **`addEventListener("click", …)`** — "when this is clicked, run this function."
- **`fetch(url, …)`** — make a request to a backend route. This is the frontend↔backend bridge.
- **`.then(…)`** — "when the response comes back, do this." (Requests take time, so JS handles them asynchronously.)

**The IDs and routes are a contract.** The JS calls `fetch("/remove_overlap")` and looks for
`id="submitBtn"`; the backend must have a `/remove_overlap` route, and the HTML must have that ID.
This is why, when you redesigned the templates, you had to preserve every `id` and every `fetch`
URL exactly — break either and the button stops talking to the backend.

**Per-page quick reference (IDs → routes):**
- `index.html` (Home): upload → `/process_images`; class fix → `/update_class`
- `csv.html`: `load`→`/get_results`, `download`→`/download_results`, `upload-csv`→`/upload_csv`, `clear-button`→`/clear_results`
- `overlap.html`: `submitBtn`→`/remove_overlap`
- `map.html`: `reload`→`/show_map`, `fast_reload`→`/fast_show_map`, `download_map`→`/download_map`, the `<iframe>`→`/get_cur`
- `plots.html`: `run`→`/plotting` (sends `num_clusters` as JSON)

---

## PART 9 — HOW IT ALL CONNECTS TO DEPLOYMENT

Now the hosting picture makes sense:
- The app is a **full-stack Flask app** whose **backend runs heavy ML models** → needs real CPU/RAM (ideally GPU).
- It keeps **state in memory** (`python.config.csv_file`, `map`) and **writes files** (`detections.csv`) → a host that wipes the filesystem or restarts often (free tiers) is a poor fit.
- The plan: **package the whole thing (code + environment) into a Docker container** so it runs identically anywhere, then run that container somewhere with enough compute — short-term a cheap host, long-term Colby's server.

**Docker** = a way to package the app *with its entire environment* (OS, Python, every dependency,
your code) into one sealed unit (a **container**) that runs the same on any machine. It's the answer
to "it works on my Mac but breaks on the server." It's *how* you make the app portable to Colby —
the container is the portable unit; the host is just where you currently run it.

---

## PART 10 — WHAT TO LEARN NEXT (anchored to this app)

You learn backend fastest by upgrading this real app. Natural next steps, easiest first:
1. **Read every route in `app.py` with this guide open** until the request loop feels obvious.
2. **Trace one full action** (e.g. Home detect) from the HTML button → JS fetch → Python route → response → page update. Do it on paper.
3. **Add proper error handling** (e.g. wrap the EXIF read — Bug 5) — small, safe, teaches you the request lifecycle.
4. **Move `detections.csv` to a real database** (SQLite → PostgreSQL) — the single best backend learning project here; teaches databases, the thing CSVs are standing in for.
5. **Dockerize the app** — directly serves Tahiya's goal and teaches deployment.
6. **Then** the bigger experiments (model swaps, LightGlue) once you have GPU access.

---

*Built from the actual source of your app. Keep it open while you work; update it as the code changes.*