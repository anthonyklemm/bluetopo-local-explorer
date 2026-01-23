# BlueTopo Local Explorer (Docker)

A local-first BlueTopo explorer that:
- downloads BlueTopo tiles for a user-defined area of interest (AOI),
- builds VRT mosaics,
- renders an interactive Folium web map,
- optionally supports an OpenAI-powered “Chat Assistant” **if the user provides their own API key**.

**Local compute:** All processing runs on your machine via Docker.  
**Network use:** The app downloads BlueTopo data (and will only call OpenAI if you enable the Chat Assistant).

Read about it here in my blog post: https://medium.com/python-in-plain-english/chatting-with-the-seafloor-building-an-ai-interface-for-noaa-bluetopo-0aca4351c430

---

## Quickstart

1) Install Docker Desktop  
2) Download or clone this repo  
3) In a terminal in the repo folder:

```bash
docker compose up --build
```

4) Open:

`http://localhost:8777/index.html`

Stop with `Ctrl + C`, then:

```bash
docker compose down
```

---

## Requirements

- Docker Desktop (recommended) on Mac or Windows
- Internet access to download BlueTopo tiles (and optional OpenAI API calls if enabled)

You do **not** need Python or GDAL installed on your host machine.

---

## Install and run (Mac)

### 1) Install Docker Desktop
- Install Docker Desktop for Mac
- Launch it and wait until it shows “running”

### 2) Get the project
Option A: Git (recommended)

```bash
git clone https://github.com/anthonyklemm/bluetopo-local-explorer
cd bluetopo-local-explorer
```

Option B: Download ZIP from GitHub
- Click the green **Code** button → **Download ZIP**
- Unzip it
- Open Terminal and `cd` into the unzipped folder

### 3) Start the app

```bash
docker compose up --build
```

### 4) Open the UI
`http://localhost:8777/index.html`

### 5) Stop the app
In the terminal running Docker, press `Ctrl + C`, then:

```bash
docker compose down
```

---

## Install and run (Windows)

### 1) Install Docker Desktop
- Install Docker Desktop for Windows
- Launch it and wait until it’s running (whale icon)

> Tip: If Docker prompts you about WSL2, follow its recommended setup.

### 2) Get the project
Option A: Git (Git Bash / Windows Terminal)

```bash
git clone https://github.com/anthonyklemm/bluetopo-local-explorer
cd bluetopo-local-explorer
```

Option B: Download ZIP from GitHub
- Click **Code** → **Download ZIP**
- Unzip it
- Open Windows Terminal/PowerShell in that folder

### 3) Start the app

```bash
docker compose up --build
```

### 4) Open the UI
`http://localhost:8777/index.html`

### 5) Stop the app
Press `Ctrl + C`, then:

```bash
docker compose down
```

---

## Using the tool

1) Open the web page.
2) Draw an AOI polygon (top-right drawing tools) or use the viewport bbox.
3) Choose Mode and parameters.
4) Click Run.
5) The container downloads tiles, builds VRTs, and updates the map outputs.

All cached data is stored on your computer in `./data`.

---

## Optional: Enable OpenAI Chat Assistant (each user uses their own key)

Manual mode works without any API key.

To enable Chat Assistant:
- the container must include the `openai` Python package, and
- you must set `OPENAI_API_KEY` (your own key) when running.

### Recommended method: `.env` file (safe + simple)

1) Copy the example env file to a real one:

**Mac/Linux:**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
copy .env.example .env
```

2) Edit `.env` and paste your key:

```env
OPENAI_API_KEY=sk-your-key-here
# Optional: override the model
# OPENAI_MODEL=gpt-4o-mini
```

3) Rebuild and run:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### Security notes
- **Never commit `.env`** to GitHub.
- Each user should create their own `.env` locally.
- If you think your key was exposed, rotate it immediately in your OpenAI dashboard.

---

## Configuration

### Port
Default: `8777`

If you see “port already in use”, change `docker-compose.yml`:

```yaml
ports:
  - "127.0.0.1:8788:8777"
```

Then open:

`http://localhost:8788/index.html`

### Data/cache location
Docker mounts `./data` to `/data` in the container:

```yaml
volumes:
  - ./data:/data
```

Delete `./data` if you want a clean slate (it may be large).

---

## Troubleshooting

### “docker: command not found”
Docker Desktop isn’t installed or isn’t running.

### “No module named 'osgeo'” (GDAL)
Rebuild clean:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### NumPy “multiarray failed to import”
This usually means a NumPy 2.x vs compiled-extension mismatch. This project pins `numpy<2`. Rebuild with no cache:

```bash
docker compose build --no-cache
```

### The map loads but doesn’t update
- Make sure `docker compose up` is still running
- Hard refresh your browser (Shift+Reload)
- Check logs in the terminal

---

## Disclaimer

This is a personal project and not an official NOAA product.
