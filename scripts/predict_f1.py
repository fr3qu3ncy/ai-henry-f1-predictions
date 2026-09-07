#!/usr/bin/env python3
"""
F1 Race Prediction Engine
=========================
Fetches F1 data from F1DB (https://github.com/f1db/f1db) and BBC news,
then generates top-10 race predictions using an LLM. Three prediction types:
  - early: No sessions have run yet (season context only)
  - pre-qualifying: Practice sessions done, before qualifying
  - post-qualifying: Qualifying done, before the race

Usage:
  python3 predict_f1.py --type early|pre-qualifying|post-qualifying [--repo /path/to/repo]
  python3 predict_f1.py --detect [--repo /path/to/repo]   # auto-detect type from schedule
"""

import argparse
import json
import os
import re
import sys
import time
import subprocess
import shutil
import yaml
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ── Configuration ──────────────────────────────────────────────────────────

F1DB_BASE = "https://raw.githubusercontent.com/f1db/f1db/main/src/data"
F1DB_SEASON = "2026"
BBC_F1_NEWS = "https://www.bbc.co.uk/sport/formula1"
SCHEDULE_SCRIPT = os.path.expanduser("~/.hermes/skills/motorsport-schedule/scripts/get_schedule.py")

# LLM configuration
LLM_HOST = "http://ml02.dmz:9931"
LLM_MODEL = "qwen/qwen3.8-27b"  # 27B model (matches Hermes config model.default)
LLM_CONTEXT_WINDOW = 262000  # Matches Hermes config model.context_length
LLM_MAX_TOKENS = 32768  # Qwen3 thinking mode spends most of the budget on reasoning_content;
                        # 8192 was exhausted by reasoning leaving content empty (finish_reason=length).
                        # Raised to 32768 for headroom; model stops naturally at finish_reason=stop.

# Session end times (BST) used to auto-detect prediction window
PRACTICE3_END_BST = "14:00"
QUALIFYING_END_BST = "14:00"
RACE_START_BST = "14:00"

# ── F1DB Data fetching ────────────────────────────────────────────────────

# Global cache for driver names (persists across function calls in same run)
_driver_name_cache = {}
_driver_team_cache = {}
_entrants_cache = None
_race_slug_cache = None

def _get_entrants():
    """Get entrants data with caching."""
    global _entrants_cache
    if _entrants_cache is None:
        _entrants_cache = f1db_fetch(f"seasons/{F1DB_SEASON}/entrants.yml")
    return _entrants_cache

def _get_driver_name(driver_id):
    """Get driver name with caching."""
    if driver_id not in _driver_name_cache:
        drv_data = f1db_fetch(f"drivers/{driver_id}.yml")
        if drv_data:
            _driver_name_cache[driver_id] = drv_data.get("name", driver_id.replace("-", " ").title())
        else:
            _driver_name_cache[driver_id] = driver_id.replace("-", " ").title()
    return _driver_name_cache[driver_id]

def _get_driver_team(driver_id):
    """Get driver team with caching."""
    if not _driver_team_cache:
        entrants = _get_entrants()
        if entrants:
            for entrant in entrants:
                team_name = entrant.get("constructorId", "").replace("-", " ").title()
                for drv in entrant.get("drivers", []):
                    _driver_team_cache[drv["driverId"]] = team_name
    return _driver_team_cache.get(driver_id, "Unknown")

def f1db_fetch(yaml_path):
    """Fetch and parse a YAML file from F1DB."""
    url = f"{F1DB_BASE}/{yaml_path}"
    try:
        req = Request(url, headers={"User-Agent": "Henry-F1-Predictions/1.0"})
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        return yaml.safe_load(text)
    except HTTPError as e:
        if e.code != 404:
            print(f"  Warning: Failed to fetch {url}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Warning: Failed to fetch {url}: {e}", file=sys.stderr)
        return None

def f1db_fetch_raw(yaml_path):
    """Fetch raw text from F1DB (to check if file exists)."""
    url = f"{F1DB_BASE}/{yaml_path}"
    try:
        req = Request(url, headers={"User-Agent": "Henry-F1-Predictions/1.0"})
        with urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except HTTPError:
        return None
    except Exception:
        return None

def get_current_standings():
    """Fetch current driver standings from F1DB."""
    data = f1db_fetch(f"seasons/{F1DB_SEASON}/driver-standings.yml")
    if not data:
        return []
    
    standings = []
    for entry in data:
        driver_id = entry.get("driverId", "")
        pos = entry.get("position", 0)
        pts = entry.get("points", 0)
        
        standings.append({
            "position": pos,
            "name": _get_driver_name(driver_id),
            "driverId": driver_id,
            "team": _get_driver_team(driver_id),
            "points": pts
        })
    
    return standings

def get_race_info(race_slug):
    """Fetch race info for a specific race."""
    data = f1db_fetch(f"seasons/{F1DB_SEASON}/races/{race_slug}/race.yml")
    return data if data else {}

def is_sprint_race(race_slug):
    """Check if a race weekend includes a sprint (has sprintQualifyingFormat in race.yml)."""
    race_info = get_race_info(race_slug)
    return race_info.get("sprintQualifyingFormat") is not None

def get_sprint_dates(race_slug):
    """Get sprint-related dates from race info for scheduling."""
    race_info = get_race_info(race_slug)
    return {
        "sprint_qualifying_date": race_info.get("sprintQualifyingDate", ""),
        "sprint_qualifying_time": race_info.get("sprintQualifyingTime", ""),
        "sprint_race_date": race_info.get("sprintRaceDate", ""),
        "sprint_race_time": race_info.get("sprintRaceTime", ""),
    }

def get_race_results(race_slug):
    """Fetch race results for a specific race."""
    data = f1db_fetch(f"seasons/{F1DB_SEASON}/races/{race_slug}/race-results.yml")
    if not data:
        return []
    
    results = []
    for entry in data:
        driver_id = entry.get("driverId", "")
        retired = entry.get("reasonRetired", "")
        results.append({
            "position": entry.get("position", 0),
            "name": _get_driver_name(driver_id),
            "driverId": driver_id,
            "team": _get_driver_team(driver_id),
            "points": entry.get("points", 0),
            "gridPosition": entry.get("gridPosition", 0),
            "time": entry.get("time", ""),
            "reasonRetired": retired
        })
    
    return results

def get_sprint_results(race_slug):
    """Fetch sprint race results for a specific race."""
    data = f1db_fetch(f"seasons/{F1DB_SEASON}/races/{race_slug}/sprint-race-results.yml")
    if not data:
        return []
    
    results = []
    for entry in data:
        driver_id = entry.get("driverId", "")
        retired = entry.get("reasonRetired", "")
        results.append({
            "position": entry.get("position", 0),
            "name": _get_driver_name(driver_id),
            "driverId": driver_id,
            "team": _get_driver_team(driver_id),
            "points": entry.get("points", 0),
            "gridPosition": entry.get("gridPosition", 0),
            "time": entry.get("time", ""),
            "reasonRetired": retired
        })
    
    return results

def get_session_results(race_slug, session_type):
    """Fetch results for a specific session (practice/qualifying/sprint)."""
    session_files = {
        "practice1": "free-practice-1-results.yml",
        "practice2": "free-practice-2-results.yml",
        "practice3": "free-practice-3-results.yml",
        "qualifying": "qualifying-results.yml",
        "sprint_qualifying": "sprint-qualifying-results.yml",
        "sprint_race": "sprint-race-results.yml",
    }
    
    filename = session_files.get(session_type)
    if not filename:
        return None
    
    data = f1db_fetch(f"seasons/{F1DB_SEASON}/races/{race_slug}/{filename}")
    if not data:
        return None
    
    results = []
    for entry in data:
        driver_id = entry.get("driverId", "")
        result = {
            "position": entry.get("position", 0),
            "name": _get_driver_name(driver_id),
            "driverId": driver_id,
            "team": _get_driver_team(driver_id),
        }
        
        if session_type == "qualifying" or session_type == "sprint_qualifying":
            result["q1"] = entry.get("q1", "")
            result["q2"] = entry.get("q2", "")
            result["q3"] = entry.get("q3", "")
            result["gap"] = entry.get("gap", "")
        else:
            # practice sessions and sprint_race
            result["time"] = entry.get("time", "")
            result["laps"] = entry.get("laps", 0)
            if session_type == "sprint_race":
                result["gridPosition"] = entry.get("gridPosition", 0)
                result["points"] = entry.get("points", 0)
                result["reasonRetired"] = entry.get("reasonRetired", "")
        
        results.append(result)
    
    return results

def check_session_exists(race_slug, session_type):
    """Check if a session results file exists in F1DB."""
    session_files = {
        "practice1": "free-practice-1-results.yml",
        "practice2": "free-practice-2-results.yml",
        "practice3": "free-practice-3-results.yml",
        "qualifying": "qualifying-results.yml",
        "sprint_qualifying": "sprint-qualifying-results.yml",
        "sprint_race": "sprint-race-results.yml",
    }
    
    filename = session_files.get(session_type)
    if not filename:
        return False
    
    raw = f1db_fetch_raw(f"seasons/{F1DB_SEASON}/races/{race_slug}/{filename}")
    return raw is not None

def get_all_race_slugs():
    """Get list of all race slugs for the season."""
    races_dir = f"src/data/seasons/{F1DB_SEASON}/races"
    api_url = f"https://api.github.com/repos/f1db/f1db/contents/{races_dir}"
    try:
        req = Request(api_url, headers={"User-Agent": "Henry-F1-Predictions/1.0"})
        with urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        return [item["name"] for item in items if item["type"] == "dir"]
    except Exception:
        return []

# F1 race location aliases — slug names often differ from race display names
_RACE_ALIASES = {
    "british": "great-britain",
    "great-britain": "great-britain",
    "united-states": "united-states",
    "american": "united-states",
    "usa": "united-states",
    "sao-paulo": "sao-paulo",
    "brazilian": "sao-paulo",
    "brazil": "sao-paulo",
    "barcelona-catalunya": "barcelona-catalunya",
    "spanish": "barcelona-catalunya",
    "catalunya": "barcelona-catalunya",
    "azerbaijan": "azerbaijan",
    "baku": "azerbaijan",
    "las-vegas": "las-vegas",
    "nevada": "las-vegas",
    "belgian": "belgium",
    "belgium": "belgium",
    "hungarian": "hungary",
    "hungary": "hungary",
    "dutch": "netherlands",
    "netherlands": "netherlands",
    "italian": "italy",
    "italy": "italy",
    "italia": "italy",
    "mexican": "mexico",
    "mexico": "mexico",
    "qatari": "qatar",
    "qatar": "qatar",
    "spain": "spain",
    "espana": "spain",
    "españa": "spain",
    "español": "spain",
    "gran premio de espana": "spain",
}

def _normalize_location(name):
    """Normalize F1 race location names for matching."""
    name_lower = name.lower()
    for alias, normalized in _RACE_ALIASES.items():
        if alias in name_lower:
            return normalized
    return name_lower

def find_race_slug(race_name):
    """Find the F1DB slug for a race by name."""
    global _race_slug_cache
    if _race_slug_cache is None:
        slugs = get_all_race_slugs()
        _race_slug_cache = slugs

    race_name_lower = race_name.lower()
    race_normalized = _normalize_location(race_name)

    for slug in _race_slug_cache:
        slug_lower = slug.lower()
        # Direct substring match (e.g. "monaco" in "Monaco GP")
        # Strip leading numbers from slug for matching
        slug_stripped = slug_lower.split("-", 1)[-1] if "-" in slug_lower else slug_lower
        if slug_stripped in race_name_lower or race_name_lower in slug_stripped:
            return slug
        # Normalized location match (handles "british" vs "great-britain")
        slug_normalized = _normalize_location(slug)
        if slug_normalized and race_normalized and slug_normalized == race_normalized:
            return slug
        # Check if slug keywords appear in race name (slug is shorter, race name is longer)
        # Normalize: replace hyphens with spaces and check overlap
        slug_words = slug_lower.replace("-", " ").split()
        matches = sum(1 for word in slug_words if word in race_name_lower and len(word) > 2)
        if matches >= 2:
            return slug
    return None

def fetch_standings():
    """Alias for get_current_standings for backward compatibility."""
    return get_current_standings()

def fetch_race_results():
    """Fetch results from all completed races this season (only races that have results data)."""
    race_slugs = get_all_race_slugs()
    completed_races = []
    
    for slug in race_slugs:
        # Quick check: skip if race-results.yml doesn't exist (future race)
        if not check_session_exists(slug, "practice1"):
            # If even practice1 doesn't exist, this is likely a future race
            # Do a quick check on race-results directly
            raw = f1db_fetch_raw(f"seasons/{F1DB_SEASON}/races/{slug}/race-results.yml")
            if raw is None:
                continue
        
        results = get_race_results(slug)
        if results:
            race_info = get_race_info(slug)
            completed_races.append({
                "name": race_info.get("officialName", slug.replace("-", " ").title()),
                "slug": slug,
                "date": race_info.get("date", ""),
                "round": race_info.get("round", 0),
                "circuitType": race_info.get("circuitType", "UNKNOWN"),
                "results": results
            })
    
    return completed_races

def fetch_news():
    """Fetch BBC F1 news headlines via RSS feed (HTML scraping broken - BBC is JS-rendered SPA)."""
    headlines = []
    try:
        rss_url = "https://feeds.bbci.co.uk/sport/formula1/rss.xml"
        req = Request(rss_url, headers={"User-Agent": "Henry-F1-Predictions/1.0"})
        with urlopen(req, timeout=15) as resp:
            rss_text = resp.read().decode("utf-8")
        
        # Extract from <item> elements - limit to first N items, not N headlines
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(rss_text)
            items = root.findall(".//item")
            for item in items[:12]:  # 12 items to catch penalty/news from earlier in the weekend
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    clean = title_el.text.strip()
                    if len(clean) > 15 and len(clean) < 300:
                        headlines.append(clean)
                desc_el = item.find("description")
                if desc_el is not None and desc_el.text:
                    clean_desc = re.sub(r'<[^>]+>', '', desc_el.text).strip()
                    if len(clean_desc) > 30 and len(clean_desc) < 500:
                        headlines.append(f"[detail] {clean_desc}")
        except ET.ParseError:
            # Fallback: regex extraction
            for m in re.finditer(r'<title><!\[CDATA\[(.+?)\]\]></title>', rss_text):
                headlines.append(m.group(1).strip())
    except Exception as e:
        print(f"  Warning: News fetch failed: {e}", file=sys.stderr)
    
    return headlines[:25]  # Enough to capture penalty/news items deeper in the feed

# ── FIA Document fetching ─────────────────────────────────────────────────

FIA_DOCS_BASE = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2026-2072"
FIA_PDF_BASE = "https://www.fia.com/system/files/decision-document/"
FIA_GRID_CACHE = os.path.expanduser("~/.hermes/data/f1_fia_grid_cache.json")
FIA_GRID_CACHE_TTL = 6 * 3600  # 6 hours — grid doesn't change much after qualifying

def fetch_fia_starting_grid(race_name):
    """Fetch the Provisional/Final Starting Grid PDF from FIA docs page, extract text via PyMuPDF.
    Returns dict with 'grid' (list of {position, driver, car_number, time}) and 'penalties' (list of strings).
    Uses cache to avoid re-downloading within a race weekend."""
    
    # Check cache first
    if os.path.exists(FIA_GRID_CACHE):
        try:
            mtime = os.path.getmtime(FIA_GRID_CACHE)
            if time.time() - mtime < FIA_GRID_CACHE_TTL:
                with open(FIA_GRID_CACHE) as f:
                    cache = json.load(f)
                if cache.get('race_name', '').lower() in race_name.lower() or race_name.lower() in cache.get('race_name', '').lower():
                    return cache.get('data')
        except Exception:
            pass
    
    # Scrape FIA docs page for PDF links
    try:
        req = Request(FIA_DOCS_BASE, headers={"User-Agent": "Henry-F1-Predictions/1.0"})
        with urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  Warning: Failed to fetch FIA docs page: {e}", file=sys.stderr)
        return None
    
    # Extract PDF hrefs
    pdf_hrefs = re.findall(r'href="(/system/files/decision-document/[^"]+\.pdf)"', html)
    if not pdf_hrefs:
        print("  Warning: No PDF links found on FIA docs page", file=sys.stderr)
        return None
    
    # Find the starting grid PDF — look for "starting_grid" in filename
    grid_pdf = None
    race_lower = race_name.lower()
    # Normalize race name for matching (e.g. "Belgian Grand Prix" -> "belgian_grand_prix")
    race_slug = re.sub(r'[^a-z0-9]+', '_', race_lower).strip('_')
    
    for href in pdf_hrefs:
        filename = href.split('/')[-1].lower()
        # Check if this PDF is for the current race
        if race_slug in filename or any(word in filename for word in ['belgian', 'british', 'monaco', 'italian', 'spanish', 'french', 'hungarian', 'dutch', 'belgian', 'swiss', 'azerbaijan', 'singapore', 'japanese', 'qatari', 'american', 'mexican', 'brazilian', 'las_vegas', 'abu_dhabi', 'chinese', 'australian', 'saudi', 'emilia_romagna', 'miami', 'canadian', 'turkish']):
            if 'starting_grid' in filename:
                grid_pdf = FIA_PDF_BASE + href.split('/')[-1]
                break
    
    if not grid_pdf:
        print(f"  Warning: No starting grid PDF found for {race_name}", file=sys.stderr)
        return None
    
    # Download and extract PDF text via PyMuPDF
    try:
        import fitz
        req = Request(grid_pdf, headers={"User-Agent": "Henry-F1-Predictions/1.0"})
        with urlopen(req, timeout=30) as resp:
            pdf_data = resp.read()
        
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"  Warning: Failed to extract FIA starting grid PDF: {e}", file=sys.stderr)
        return None
    
    # Parse the extracted text
    result = parse_starting_grid(text)
    
    # Cache the result
    cache = {
        'race_name': race_name,
        'timestamp': time.time(),
        'data': result
    }
    try:
        with open(FIA_GRID_CACHE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass
    
    return result


def parse_starting_grid(text):
    """Parse FIA starting grid PDF text into structured data.
    The PDF has a two-column layout with multi-line entries:
      POSITION
      CAR_NUMBER DRIVER_NAME
      TEAM_NAME
      (lap time — skipped, available from F1DB)
    Returns dict with 'grid' (list of {position, driver, car_number}) and 'penalties' (list of strings)."""

    grid = []
    penalties = []

    lines = text.strip().split('\n')

    # Parse grid entries — the PDF uses format:
    # Line 1: POSITION (standalone number 1-22)
    # Line 2: CAR_NUMBER DRIVER_NAME
    # Line 3: TEAM_NAME (skipped)
    pos_pattern = re.compile(r'^(2[0-2]|1?[0-9])$')  # position 1-22
    car_driver_pattern = re.compile(r'^(\d+)\s+(.+)')  # car_number + driver_name

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line is a standalone position number (1-22)
        pos_match = pos_pattern.match(line)
        if pos_match:
            pos = int(pos_match.group(1))

            # Next line should be car number + driver name
            i += 1
            if i < len(lines):
                cd_match = car_driver_pattern.match(lines[i].strip())
                if cd_match:
                    car_num = cd_match.group(1)
                    driver = cd_match.group(2).strip()

                    # Next line is team name — skip it
                    i += 1

                    # Clean up driver name (remove trailing asterisk for penalties)
                    driver = driver.replace('*', '').strip()

                    grid.append({
                        'position': pos,
                        'car_number': car_num,
                        'driver': driver
                    })
        i += 1

    # Parse penalty notes — lines starting with "* PENALTIES", "Car X" or "Cars X"
    in_penalties = False
    for line in lines:
        line = line.strip()
        if line.upper().startswith('* PENALTIES'):
            in_penalties = True
            continue
        if in_penalties and (line.startswith('Car ') or line.startswith('Cars ')):
            penalties.append(line)
        elif in_penalties and not line.startswith('Car') and not line.startswith('*'):
            # End of penalty section
            break

    # Sort grid by position
    grid.sort(key=lambda x: x['position'])

    return {
        'grid': grid,
        'penalties': penalties
    }

SCHEDULE_CACHE = os.path.expanduser("~/.hermes/data/f1_schedule_cache.json")
SCHEDULE_CACHE_TTL = 12 * 3600  # 12 hours — cache is valid within a race weekend
HISTORICAL_CACHE = os.path.expanduser("~/.hermes/data/f1_historical_cache.json")
HISTORICAL_CACHE_TTL = 24 * 3600  # 24 hours — historical data changes rarely

def get_historical_circuit_data(grand_prix_id, years=3):
    """Fetch race results for a circuit from the last N years, aggregated by driver/team.
    Returns dict: {driver_name: [(year, position, team), ...]}"""
    import datetime
    
    # Check cache first
    if os.path.exists(HISTORICAL_CACHE):
        try:
            with open(HISTORICAL_CACHE) as f:
                cache = json.load(f)
            if cache.get('grand_prix_id') == grand_prix_id:
                age = time.time() - cache.get('timestamp', 0)
                if age < HISTORICAL_CACHE_TTL:
                    return cache.get('data')
        except Exception:
            pass
    
    current_year = datetime.datetime.now().year
    driver_history = defaultdict(list)
    
    for year in range(current_year - years, current_year):
        try:
            # Get races for this year
            req = Request(
                f'https://api.github.com/repos/f1db/f1db/contents/src/data/seasons/{year}/races',
                headers={'User-Agent': 'Henry/1.0'}
            )
            with urlopen(req, timeout=30) as resp:
                races = json.loads(resp.read())
            
            # Find the matching circuit
            for race in races:
                if grand_prix_id in race['name'].lower():
                    # Fetch race results
                    race_url = f"https://raw.githubusercontent.com/f1db/f1db/main/src/data/seasons/{year}/races/{race['name']}/race-results.yml"
                    req = Request(race_url, headers={'User-Agent': 'Henry/1.0'})
                    with urlopen(req, timeout=30) as resp:
                        results = yaml.safe_load(resp.read())
                    
                    if results:
                        for r in results:
                            driver_name = r.get('driverId', '').replace('-', ' ').title()
                            team = r.get('constructorId', '').replace('-', ' ').title()
                            pos = r.get('position')
                            # Skip DNF/retired entries for stats
                            if isinstance(pos, int):
                                driver_history[driver_name].append({
                                    'year': year,
                                    'position': pos,
                                    'team': team
                                })
                    break
        except Exception as e:
            continue
    
    # Cache the result
    cache = {
        'grand_prix_id': grand_prix_id,
        'timestamp': time.time(),
        'data': dict(driver_history)
    }
    try:
        with open(HISTORICAL_CACHE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass
    
    return dict(driver_history)

def get_schedule():
    """Get F1 schedule: try cached file first, fall back to live API with retries."""
    # 1. Try reading from cache
    if os.path.exists(SCHEDULE_CACHE):
        try:
            mtime = os.path.getmtime(SCHEDULE_CACHE)
            if time.time() - mtime < SCHEDULE_CACHE_TTL:
                with open(SCHEDULE_CACHE) as f:
                    data = json.load(f)
                if data and isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Fall back to live API call via motorsport-schedule script (with retries)
    if not os.path.exists(SCHEDULE_SCRIPT):
        return None

    for attempt in range(3):
        try:
            result = subprocess.run(
                ["python3", SCHEDULE_SCRIPT, "f1"],
                capture_output=True, text=True, timeout=45
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Update cache for next time
                try:
                    with open(SCHEDULE_CACHE, "w") as f:
                        json.dump(data, f)
                except OSError:
                    pass
                return data
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            if attempt < 2:
                time.sleep(2 * (attempt + 1))  # 2s, 4s backoff
            continue
    return None

def detect_prediction_type():
    """Detect which prediction type to run based on current session status."""
    schedule = get_schedule()
    if not schedule:
        return "early"
    
    now = datetime.now(timezone(timedelta(hours=1)))  # BST
    
    for cat in schedule:
        if cat["category"] != "f1":
            continue
        race = cat.get("next_race")
        if not race:
            return "early"
        
        sessions = race.get("sessions", [])
        if not sessions:
            return "early"
        
        # Check what sessions exist and their times
        session_names = [s["name"].lower() for s in sessions]
        session_times = {}
        for s in sessions:
            dt = datetime.fromisoformat(s["datetime"])
            # Make timezone-aware (schedule API returns naive datetimes)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone(timedelta(hours=1)))  # BST
            session_times[s["name"].lower()] = dt
        
        # Detect if this is a sprint weekend
        is_sprint = any("sprint" in name for name in session_names)
        
        # If only race is listed, qualifying has been done
        if "race" in session_names and len(session_names) == 1:
            return "post-qualifying"
        
        # Session duration estimates (used to determine if a session has actually ENDED)
        # A session is "done" when: start_time + duration < now
        SESSION_DURATION = {
            "practice": timedelta(hours=1, minutes=30),
            "qualifying": timedelta(hours=1, minutes=15),
            "sprint": timedelta(hours=1),  # sprint race
            "race": timedelta(hours=1, minutes=30),
        }
        
        def session_ended(name, dt):
            """Check if a session has likely ended based on start time + duration."""
            if "practice" in name:
                return dt + SESSION_DURATION["practice"] < now
            elif "sprint" in name and "qualifying" in name:
                return dt + SESSION_DURATION["qualifying"] < now
            elif "sprint" in name:
                return dt + SESSION_DURATION["sprint"] < now
            elif "qualifying" in name:
                return dt + SESSION_DURATION["qualifying"] < now
            elif "race" in name:
                return dt + SESSION_DURATION["race"] < now
            return False
        
        # Check if sessions are in the past (using duration-aware checks)
        practice_sessions = []
        first_practice_time = None
        sprint_qualifying_done = False
        sprint_race_done = False
        qualifying_done = False
        
        for name, dt in session_times.items():
            # Track all practice sessions for completion check
            if "practice" in name:
                practice_sessions.append((name, dt))
            # Track first practice time for pre-practice detection
            if ("practice 1" in name or "practice1" in name) and first_practice_time is None:
                first_practice_time = dt
            if "sprint" in name and "qualifying" in name and session_ended(name, dt):
                sprint_qualifying_done = True
            if ("sprint" in name and "race" in name) or name == "sprint":
                if session_ended(name, dt):
                    sprint_race_done = True
            if "qualifying" in name and "sprint" not in name and session_ended(name, dt):
                qualifying_done = True
        
        # practice_done = ALL practice sessions have ended (not just started)
        practice_done = len(practice_sessions) > 0 and all(
            session_ended(name, dt) for name, dt in practice_sessions
        )
        
        # Pre-practice: 24 hours before FP1
        pre_practice_time = first_practice_time - timedelta(hours=24) if first_practice_time else None
        if pre_practice_time and now >= pre_practice_time and not practice_done:
            return "pre-practice"
        
        if is_sprint:
            # Sprint weekend logic
            if qualifying_done:
                return "post-qualifying"
            elif sprint_race_done:
                return "pre-qualifying"
            elif sprint_qualifying_done:
                return "post-sprint-qualifying"
            elif practice_done:
                return "pre-sprint-qualifying"
            else:
                return "early"
        else:
            # Normal weekend logic
            if qualifying_done:
                return "post-qualifying"
            elif practice_done:
                return "pre-qualifying"
            else:
                return "early"
    
    return "early"

# ── Prediction generation ─────────────────────────────────────────────────

def build_prediction_prompt(prediction_type, standings, races, news, session_data=None, race_info=None, historical_data=None, fia_grid=None):
    """Build the LLM prompt for prediction."""

    standings_text = ""
    if standings:
        standings_text = "## Current Driver Standings (2026 Season)\n"
        for d in standings[:22]:
            standings_text += f"{d['position']}. {d['name']} ({d['team']}) - {d['points']} pts\n"

    races_text = ""
    if races:
        races_text = "## Recent Race Results (2026 Season)\n"
        for race in races:
            circuit_type = race.get("circuitType", "UNKNOWN")
            races_text += f"\n### {race['name']} (Round {race.get('round', '?')}) [{circuit_type}]\n"
            if 'results' in race:
                for r in race['results'][:10]:
                    retired = f" (retired: {r['reasonRetired']})" if r.get('reasonRetired') else ""
                    races_text += f"- {r['position']}. {r['name']} ({r['team']}) - {r['points']}pts{retired}\n"

    news_text = ""
    if news:
        news_text = "## Recent F1 News\n"
        for h in news:
            news_text += f"- {h}\n"

    # FIA starting grid section — official grid with penalties applied
    fia_grid_text = ""
    if fia_grid:
        fia_grid_text = "## Official FIA Starting Grid\n"
        fia_grid_text += "**IMPORTANT: The grid positions below are the FINAL starting order after all penalties have been applied.**\n"
        fia_grid_text += "A driver's grid position may differ significantly from their qualifying position due to engine penalties.\n\n"
        grid_entries = fia_grid.get('grid', [])
        if grid_entries:
            for entry in grid_entries:
                driver = entry.get('driver', 'Unknown')
                pos = entry.get('position', '?')
                car_num = entry.get('car_number', '')
                fia_grid_text += f"P{pos}: #{car_num} {driver}\n"

        penalty_notes = fia_grid.get('penalties', [])
        if penalty_notes:
            fia_grid_text += "\n**Grid Penalties Applied:**\n"
            for pn in penalty_notes:
                fia_grid_text += f"- {pn}\n"
            fia_grid_text += "\n⚠️ USE THE GRID POSITIONS ABOVE (penalties already applied) for your predictions. Do NOT use qualifying order - a driver who qualified on pole but received a 10-place penalty starts 11th on the grid.\n"

    session_text = ""
    if session_data:
        session_text = f"## Current Weekend Session Data ({prediction_type.upper()})\n"
        for key, value in session_data.items():
            session_text += f"### {key}\n"
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        session_text += f"- Pos {item.get('position', '?')}: {item.get('name', 'Unknown')} ({item.get('team', '?')})"
                        if 'time' in item:
                            session_text += f" - {item['time']}"
                        if 'laps' in item:
                            session_text += f" ({item['laps']} laps)"
                        if 'q3' in item:
                            session_text += f" - Q3: {item['q3']}"
                        if 'gap' in item:
                            session_text += f" gap: {item['gap']}"
                        session_text += "\n"
                    else:
                        session_text += f"- {item}\n"
            else:
                session_text += f"{value}\n"
    
    # Dynamic circuit info
    circuit_info = ""
    if race_info:
        circuit_type = race_info.get("circuitType", "UNKNOWN")
        circuit_name = race_info.get("officialName", "the upcoming race")
        course_length = race_info.get("courseLength", "")
        laps = race_info.get("laps", "")
        turns = race_info.get("turns", "")
        circuit_info = f"## Race Details\n"
        circuit_info += f"- Race: {circuit_name}\n"
        circuit_info += f"- Circuit Type: {circuit_type}\n"
        if course_length:
            circuit_info += f"- Course Length: {course_length}km\n"
        if laps:
            circuit_info += f"- Laps: {laps}\n"
        if turns:
            circuit_info += f"- Turns: {turns}\n"
    
    # Historical circuit performance
    historical_text = ""
    if historical_data:
        historical_text = "## Historical Circuit Performance (Last 3 Years)\n"
        historical_text += "Driver performance at this circuit in recent years:\n"
        # Sort by best average position
        driver_stats = []
        for driver, results in historical_data.items():
            if results:
                positions = [r['position'] for r in results if r.get('position')]
                if positions:
                    avg_pos = sum(positions) / len(positions)
                    best = min(positions)
                    podiums = sum(1 for p in positions if p <= 3)
                    driver_stats.append((driver, avg_pos, best, podiums, len(positions)))
        
        # Sort by average position
        driver_stats.sort(key=lambda x: x[1])
        
        for driver, avg_pos, best, podiums, races in driver_stats[:15]:
            historical_text += f"- {driver}: avg {avg_pos:.1f}, best P{best}, {podiums} podiums ({races} races)\n"
    
    prompt = f"""You are an F1 race prediction expert. Predict the TOP 10 finishers for the upcoming race.

PREDICTION TYPE: {prediction_type.upper()}
{'=' * 50}

{standings_text}

{races_text}

{news_text}

{circuit_info}

{historical_text}

{fia_grid_text}

{session_text}

## Your Task

Predict the top 10 drivers who will finish the race, considering:
1. Current championship form and points
2. Recent race results and consistency
3. Team performance and car development
4. Circuit characteristics (provided above if available)
5. Recent news, driver form, and team developments
6. Starting grid position (if available) - **CRITICAL: Use the official FIA starting grid with penalties applied, NOT raw qualifying positions**. Grid position influences race result, especially on street/low-overtaking circuits
7. Weather conditions and reliability factors

## Output Format

Return a JSON array of exactly 10 objects, each with:
- "position": 1-10
- "driver": driver name
- "team": team name
- "confidence": "high"/"medium"/"low"
- "reasoning": brief explanation (max 100 chars)

Example:
[
  {{"position": 1, "driver": "Kimi Antonelli", "team": "Mercedes", "confidence": "high", "reasoning": "Dominant form this season, leading championship"}}
]
"""
    return prompt

def call_llm(prompt):
    """Call the LLM to generate predictions."""
    try:
        payload = {
            "model": LLM_MODEL,
            "context_window": LLM_CONTEXT_WINDOW,
            "messages": [
                {"role": "system", "content": "You are an F1 racing prediction expert. Analyze the data carefully and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": LLM_MAX_TOKENS
        }
        
        result = subprocess.run(
            [
                "curl", "-s",
                "-X", "POST",
                f"{LLM_HOST}/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload)
            ],
            capture_output=True, text=True, timeout=1200
        )
        
        if result.returncode != 0:
            print(f"  curl error: {result.stderr}", file=sys.stderr)
            return None
            
        response = json.loads(result.stdout)
        if "choices" in response:
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            if not content.strip():
                # Empty content: usually Qwen3 reasoning exhausted the max_tokens budget.
                fr = choice.get("finish_reason")
                usage = response.get("usage", {})
                print(f"  LLM returned EMPTY content (finish_reason={fr}, usage={usage}).", file=sys.stderr)
                print(f"  Likely cause: reasoning_content consumed the max_tokens budget before any content was produced.", file=sys.stderr)
                return None
            return content
        else:
            print(f"  LLM response error: {result.stdout[:200]}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print("  LLM call timed out after 1200s", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  LLM call failed: {e}", file=sys.stderr)
    
    return None

def parse_prediction(llm_output):
    """Parse the LLM output into a structured prediction."""
    if not llm_output:
        return None
    
    # Try to extract JSON from the output
    json_match = re.search(r'\[[\s\S]*\]', llm_output)
    if json_match:
        try:
            predictions = json.loads(json_match.group())
            if isinstance(predictions, list):
                return predictions
        except json.JSONDecodeError:
            pass
    
    return None

# ── Scoring ────────────────────────────────────────────────────────────────

def score_prediction(predictions, race_results):
    """Score a prediction against actual race results.

    Scoring:
    - 2 points per correct finishing position (positions 1-10) = max 20 points
    - 5 points for correct podium set (top 3 drivers, any order) = max 5 points
    - Total: 25 points max

    Returns a dict with:
      - total_score: int (0-25)
      - position_score: int (0-20)
      - podium_score: int (0 or 5)
      - position_details: list of dicts with per-position scoring
      - predicted_podium: list of 3 driver names
      - actual_podium: list of 3 driver names
      - podium_correct: bool
    """
    # Build lookup: position -> driver name (from prediction)
    pred_by_pos = {}
    for p in predictions:
        pred_by_pos[p["position"]] = p["driver"]

    # Build lookup: position -> driver name (from actual results)
    actual_by_pos = {}
    for r in race_results:
        actual_by_pos[r["position"]] = r["name"]

    # Score individual positions (1-10)
    position_details = []
    position_score = 0
    for pos in range(1, 11):
        pred_driver = pred_by_pos.get(pos, "")
        actual_driver = actual_by_pos.get(pos, "")
        correct = pred_driver == actual_driver and pred_driver != ""
        if correct:
            position_score += 2
        position_details.append({
            "position": pos,
            "predicted": pred_driver,
            "actual": actual_driver,
            "correct": correct,
            "points": 2 if correct else 0
        })

    # Score podium (top 3 as a set, order doesn't matter)
    predicted_podium = set()
    for pos in range(1, 4):
        driver = pred_by_pos.get(pos, "")
        if driver:
            predicted_podium.add(driver)

    actual_podium = set()
    for pos in range(1, 4):
        driver = actual_by_pos.get(pos, "")
        if driver:
            actual_podium.add(driver)

    podium_correct = predicted_podium == actual_podium and len(actual_podium) == 3
    podium_score = 5 if podium_correct else 0

    return {
        "total_score": position_score + podium_score,
        "position_score": position_score,
        "podium_score": podium_score,
        "position_details": position_details,
        "predicted_podium": sorted(predicted_podium),
        "actual_podium": sorted(actual_podium),
        "podium_correct": podium_correct
    }

# ── Website generation ────────────────────────────────────────────────────

def _get_circuit_svg_url(circuit_id, race_name, circuit_map):
    """Get the SVG URL for a circuit, preferring circuitId from F1DB.

    circuit_id: F1DB circuitId field (e.g. 'catalunya', 'spielberg')
    race_name: fallback race name for old history entries
    circuit_map: fallback mapping for old history entries without circuitId
    """
    if circuit_id:
        svg_file = f"{circuit_id}-1.svg"
        return f"https://raw.githubusercontent.com/f1db/f1db/main/src/assets/circuits/white/{svg_file}"
    # Fallback 1: exact match in circuit_map (for old history entries)
    svg_file = circuit_map.get(race_name, "")
    if svg_file:
        return f"https://raw.githubusercontent.com/f1db/f1db/main/src/assets/circuits/white/{svg_file}"
    # Fallback 2: keyword match — try each map key that appears in the race name
    race_lower = race_name.lower()
    for key, val in circuit_map.items():
        if key.lower() in race_lower:
            return f"https://raw.githubusercontent.com/f1db/f1db/main/src/assets/circuits/white/{val}"
    return ""

# Fallback circuit SVG mapping for old history entries without circuitId
# (New entries use F1DB circuitId directly)
_CIRCUIT_MAP = {
    "Monaco": "monaco-1.svg",
    "Bahrain": "bahrain-1.svg",
    "Saudi Arabia": "jeddah-1.svg",
    "Australia": "albert-park-1.svg",
    "Japan": "suzuka-1.svg",
    "China": "shanghai-1.svg",
    "USA": "austin-1.svg",
    "Miami": "miami-1.svg",
    "Emilia Romagna": "imola-1.svg",
    "Spain": "catalunya-1.svg",
    "Barcelona-Catalunya": "catalunya-1.svg",
    "Austria": "spielberg-1.svg",
    "Great Britain": "silverstone-1.svg",
    "Hungary": "hungaroring-1.svg",
    "Belgium": "spa-francorchamps-1.svg",
    "Netherlands": "zandvoort-1.svg",
    "Italy": "monza-1.svg",
    "Singapore": "marina-bay-1.svg",
    "Qatar": "lusail-1.svg",
    "Russia": "sochi-1.svg",
    "Mexico": "mexico-city-1.svg",
    "Brazil": "interlagos-1.svg",
    "Las Vegas": "las-vegas-1.svg",
    "Abu Dhabi": "yas-marina-1.svg",
    "Canada": "montreal-1.svg",
    "Azerbaijan": "baku-1.svg",
    "France": "paul-ricard-1.svg",
    "Turkey": "istanbul-1.svg",
    "Portugal": "portimao-1.svg",
    "Adelaide": "adelaide-1.svg",
    "Dallas": "dallas-1.svg",
    "Losail": "lusail-1.svg",
    "Mugello": "mugello-1.svg",
}

# Short display names keyed by F1DB circuitId
# Used to replace long official names (e.g. "Formula 1 MSC Cruises Gran Premio de Barcelona-Catalunya 2026")
# with clean short names (e.g. "Barcelona") on the website
_SHORT_NAME_MAP = {
    "monaco": "Monaco",
    "bahrain": "Bahrain",
    "jeddah": "Saudi Arabia",
    "melbourne": "Australia",
    "suzuka": "Japan",
    "shanghai": "China",
    "austin": "USA",
    "miami": "Miami",
    "imola": "Emilia Romagna",
    "catalunya": "Barcelona",
    "spielberg": "Austria",
    "silverstone": "Great Britain",
    "hungaroring": "Hungary",
    "spa-francorchamps": "Belgium",
    "zandvoort": "Netherlands",
    "monza": "Italy",
    "marina-bay": "Singapore",
    "lusail": "Qatar",
    "mexico-city": "Mexico",
    "interlagos": "Brazil",
    "las-vegas": "Las Vegas",
    "yas-marina": "Abu Dhabi",
    "montreal": "Canada",
    "baku": "Azerbaijan",
    "paul-ricard": "France",
    "istanbul": "Turkey",
    "portimao": "Portugal",
    "adelaide": "Adelaide",
    "dallas": "Dallas",
    "mugello": "Mugello",
    "sochi": "Russia",
}

def _get_short_name(circuit_id, race_name):
    """Get a short display name for a race.
    
    circuit_id: F1DB circuitId (e.g. 'catalunya')
    race_name: full official name as fallback
    """
    if circuit_id:
        short = _SHORT_NAME_MAP.get(circuit_id.lower())
        if short:
            return short
    # Fallback: try to extract a reasonable short name from the long race name
    # Strip "Formula 1" prefix and year suffix, then try to extract location
    name = race_name
    for prefix in ["Formula 1 ", "F1 "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Remove trailing year
    import re as _re
    name = _re.sub(r'\s+\d{4}$', '', name)
    # Try to find a known short name as a substring
    for key, short in _SHORT_NAME_MAP.items():
        if key.lower() in race_name.lower():
            return short
    # Last resort: truncate to first meaningful part
    for sep in [" Gran Premio de ", " Grand Prix of ", " Grand Prix "]:
        if sep in name:
            parts = name.split(sep)
            if len(parts) > 1:
                return parts[-1].strip()
    return race_name

def generate_index_html(weekends):
    """Generate the main index page listing all race weekends."""

    # Compute running score from scored weekends (both race + sprint scores)
    scored_races = []
    cumulative_score = 0
    max_score = 0
    for weekend in weekends:
        pq = weekend.get("scores", {}).get("post-qualifying")
        if pq and pq.get("total_score") is not None:
            circuit_id = weekend.get("circuit_id", "")
            short = _get_short_name(circuit_id, weekend["race_name"])
            scored_races.append({
                "date": weekend["date"],
                "short_name": short,
                "total_score": pq["total_score"],
                "max_score": 25,
                "type": "race",
            })
            cumulative_score += pq["total_score"]
            max_score += 25
        sq = weekend.get("scores", {}).get("post-sprint-qualifying")
        if sq and sq.get("sprint_score") is not None:
            circuit_id = weekend.get("circuit_id", "")
            short = _get_short_name(circuit_id, weekend["race_name"])
            scored_races.append({
                "date": weekend["date"],
                "short_name": short + " (Sprint)",
                "total_score": sq["sprint_score"],
                "max_score": 15,
                "type": "sprint",
            })
            cumulative_score += sq["sprint_score"]
            max_score += 15
    num_scored = len(scored_races)

    # Build score summary
    score_rows_html = ""
    if num_scored:
        for sr in scored_races:
            pct = (sr["total_score"] / sr["max_score"]) * 100
            row_class = "score-row"
            if sr["type"] == "sprint":
                row_class += " score-row-sprint"
            score_rows_html += f'''            <div class="{row_class}">
                <span class="score-row-date">{sr["date"]}</span>
                <span class="score-row-name">{sr["short_name"]}</span>
                <div class="score-row-bar">
                    <div class="score-row-fill" style="width: {pct}%"></div>
                </div>
                <span class="score-row-val">{sr["total_score"]}/{sr["max_score"]}</span>
            </div>
'''
        pct_total = (cumulative_score / max_score) * 100 if max_score else 0
        score_section = f'''
        <div class="score-summary">
            <div class="score-summary-header">
                <div class="score-summary-ring">
                    <span class="score-summary-num">{cumulative_score}</span>
                    <span class="score-summary-max">of {max_score}</span>
                </div>
                <div class="score-summary-stats">
                    <span class="score-stat">{num_scored} race{'s' if num_scored != 1 else ''} scored</span>
                    <span class="score-stat">{pct_total:.0f}% average</span>
                </div>
                <button class="score-toggle-btn" onclick="this.parentElement.parentElement.querySelector('.score-rows-container').classList.toggle('collapsed'); this.querySelector('.icon').textContent = this.parentElement.parentElement.querySelector('.score-rows-container').classList.contains('collapsed') ? '▼' : '▲';">
                    <span class="label">Details</span><span class="icon">▼</span>
                </button>
            </div>
            <div class="score-rows-container collapsed">
{score_rows_html}            </div>
        </div>
'''
    else:
        score_section = ""

    cards = ""
    for weekend in weekends:
        date_str = weekend["date"]
        race_name = weekend["race_name"]
        circuit_id = weekend.get("circuit_id", "")
        short_name = _get_short_name(circuit_id, race_name)
        prediction_types = weekend.get("prediction_types", [])
        badges = ""
        scores = weekend.get("scores", {})
        has_race_score = any(isinstance(sc, dict) and "total_score" in sc for sc in scores.values()) if scores else False
        has_sprint_score = any(isinstance(sc, dict) and "sprint_score" in sc for sc in scores.values()) if scores else False

        for pt in prediction_types:
            if pt == "early":
                badges += '<span class="badge badge-early">Early</span> '
            elif pt == "pre-qualifying":
                badges += '<span class="badge badge-preq">Pre-Qualifying</span> '
            elif pt == "pre-practice":
                badges += '<span class="badge badge-prepr">Pre-Practice</span> '
            elif pt == "post-qualifying":
                badges += '<span class="badge badge-postq">Post-Qualifying</span> '
            elif pt == "pre-sprint-qualifying":
                badges += '<span class="badge badge-presq">Pre-Sprint-Qualifying</span> '
            elif pt == "post-sprint-qualifying":
                badges += '<span class="badge badge-postsq">Post-Sprint-Qualifying</span> '
                # Insert sprint score badge right after post-sprint-Q (chronological order)
                if has_sprint_score and not has_race_score:
                    badges += '<span class="badge badge-sprintready">Post-Sprint Results Ready</span> '

        # Post-race badge at the end (after all predictions)
        if has_race_score:
            badges += '<span class="badge badge-postrace">Post-Race Results Ready</span> '

        # Circuit SVG (prefer circuitId from F1DB, fallback to circuit_map)
        svg_url = _get_circuit_svg_url(circuit_id, race_name, _CIRCUIT_MAP)
        svg_html = f'<img src="{svg_url}" alt="{short_name}" class="circuit-svg">' if svg_url else ''
        
        cards += f"""    <a href="weekends/{date_str}.html" class="race-card">
        <div class="race-card-left">
            {svg_html}
        </div>
        <div class="race-card-right">
            <div class="race-card-header">
                <span class="race-date">{date_str}</span>
                <span class="race-name">{short_name}</span>
            </div>
            <div class="race-card-badges">{badges}</div>
        </div>
    </a>
"""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Race Predictions | Henry AI</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }}
        header {{
            text-align: center;
            padding: 2rem 0;
            border-bottom: 1px solid #333;
            margin-bottom: 2rem;
        }}
        h1 {{
            font-size: 2rem;
            color: #fff;
            margin-bottom: 0.5rem;
        }}
        h1 span {{ color: #e11d48; }}
        .subtitle {{
            color: #888;
            font-size: 0.95rem;
        }}
        .race-card {{
            display: flex;
            background: #111118;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s ease;
        }}
        .race-card:hover {{
            background: #1a1a2e;
            border-color: #444;
            transform: translateY(-2px);
        }}
        .race-card-left {{
            flex-shrink: 0;
            width: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1.25rem;
        }}
        .circuit-svg {{
            width: 80px;
            height: 80px;
            object-fit: contain;
            filter: brightness(0.8);
        }}
        .race-card-right {{
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .race-card-header {{
            display: flex;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }}
        .race-date {{
            color: #666;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        .race-name {{
            color: #fff;
            font-size: 1.25rem;
            font-weight: 600;
        }}
        .race-card-badges {{
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-early {{ background: #1e3a5f; color: #93c5fd; }}
        .badge-prepr {{ background: #1a3a3a; color: #6ee7b7; }}
        .badge-presq {{ background: #4a1942; color: #f0abfc; }}
        .badge-postsq {{ background: #3b1f5e; color: #c4b5fd; }}
        .badge-preq {{ background: #1f3b5e; color: #93c5fd; }}
        .badge-postq {{ background: #1a3a2a; color: #6ee7b7; }}
        .badge-postrace {{
            background: #3a3000;
            color: #ffd700;
            border: 1px solid #ffd700;
        }}
        .badge-sprintready {{
            background: #1e3a5f;
            color: #60a5fa;
            border: 1px solid #60a5fa;
        }}
        .score-summary {{
            background: #111118;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 2rem;
        }}
        .score-summary-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .score-summary-ring {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 4px solid #fbbf24;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .score-summary-num {{
            font-size: 2.5rem;
            font-weight: 800;
            color: #fbbf24;
            line-height: 1;
        }}
        .score-summary-max {{
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.15rem;
        }}
        .score-summary-stats {{
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            flex: 1;
        }}
        .score-stat {{
            color: #aaa;
            font-size: 0.85rem;
        }}
        .score-toggle-btn {{
            background: transparent;
            border: 1px solid #333;
            border-radius: 6px;
            color: #aaa;
            font-size: 0.8rem;
            padding: 0.35rem 0.75rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.35rem;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }}
        .score-toggle-btn:hover {{
            background: #1a1a2e;
            border-color: #555;
            color: #fff;
        }}
        .score-rows-container {{
            overflow: hidden;
            max-height: 500px;
            transition: max-height 0.3s ease, opacity 0.3s ease, margin-top 0.3s ease;
            max-height: 500px;
            opacity: 1;
            margin-top: 1rem;
        }}
        .score-rows-container.collapsed {{
            max-height: 0;
            opacity: 0;
            margin-top: 0;
            overflow: hidden;
        }}
        .score-row {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0;
            border-top: 1px solid #1a1a2e;
        }}
        .score-row-date {{
            color: #666;
            font-size: 0.8rem;
            min-width: 80px;
        }}
        .score-row-name {{
            color: #fff;
            font-size: 0.9rem;
            font-weight: 500;
            flex: 1;
        }}
        .score-row-bar {{
            width: 60px;
            height: 6px;
            background: #1a1a2e;
            border-radius: 3px;
            overflow: hidden;
        }}
        .score-row-fill {{
            height: 100%;
            background: #fbbf24;
            border-radius: 3px;
        }}
        .score-row-val {{
            color: #aaa;
            font-size: 0.85rem;
            font-weight: 600;
            min-width: 40px;
            text-align: right;
        }}
        .score-row-sprint {{
            background: #1e3a5f11;
            border-left: 3px solid #3b82f6;
            padding-left: 0.5rem;
        }}
        .score-row-sprint .score-row-name {{
            color: #60a5fa;
        }}
        @media (max-width: 640px) {{
            .score-summary-header {{
                gap: 0.75rem;
            }}
            .score-summary-ring {{
                width: 72px;
                height: 72px;
            }}
            .score-summary-num {{
                font-size: 1.75rem;
            }}
            .score-toggle-btn .label {{
                display: none;
            }}
            .score-row {{
                flex-wrap: wrap;
            }}
            .score-row-name {{
                min-width: 0;
            }}
        }}
        footer {{
            text-align: center;
            padding: 2rem 0;
            color: #555;
            font-size: 0.85rem;
            border-top: 1px solid #222;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>F1 Race <span>Predictions</span></h1>
            <p class="subtitle">AI-powered top-10 predictions by Henry | 2026 Season</p>
        </header>
        
{score_section}{cards}
        <footer>
            <p>Generated by Henry AI &middot; Powered by data from F1DB &amp; BBC Sport</p>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} </p>
        </footer>
    </div>
</body>
</html>"""

def generate_weekend_html(weekend_data):
    """Generate an individual weekend page."""
    date_str = weekend_data["date"]
    race_name = weekend_data["race_name"]
    circuit_id = weekend_data.get("circuit_id", "")
    short_name = _get_short_name(circuit_id, race_name)
    predictions = weekend_data.get("predictions", {})
    
    # Circuit SVG (prefer circuitId from F1DB, fallback to circuit_map)
    svg_url = _get_circuit_svg_url(circuit_id, race_name, _CIRCUIT_MAP)
    
    badge_classes = {
        "early": "badge-early",
        "pre-practice": "badge-prepr",
        "pre-sprint-qualifying": "badge-presq",
        "post-sprint-qualifying": "badge-postsq",
        "pre-qualifying": "badge-preq",
        "post-qualifying": "badge-postq"
    }
    card_border_classes = {
        "early": "pred-card-early",
        "pre-practice": "pred-card-prepr",
        "pre-sprint-qualifying": "pred-card-presq",
        "post-sprint-qualifying": "pred-card-postsq",
        "pre-qualifying": "pred-card-preq",
        "post-qualifying": "pred-card-postq"
    }
    pred_order = ["early", "pre-practice", "pre-sprint-qualifying", "post-sprint-qualifying", "pre-qualifying", "post-qualifying"]
    
    pred_labels = {
        "early": "Early",
        "pre-practice": "Pre-Practice",
        "pre-sprint-qualifying": "Pre-Sprint-Q",
        "post-sprint-qualifying": "Post-Sprint-Q",
        "pre-qualifying": "Pre-Qualifying",
        "post-qualifying": "Post-Qualifying"
    }
    # Score cards (computed upfront, inserted in chronological order below)
    scores = weekend_data.get("scores", {})

    # Post-race score card (only when main race is scored)
    score_card = ""
    if scores:
        last_ptype = None
        last_score = None
        for ptype in reversed(pred_order):
            if ptype in scores and scores[ptype].get("total_score") is not None:
                last_ptype = ptype
                last_score = scores[ptype]
                break
        if last_score is not None:
            label = pred_labels.get(last_ptype, last_ptype)
            total = last_score.get("total_score", 0)
            score_card = f"""        <div class="pred-card pred-card-result">
            <a href="{date_str}_score_{last_ptype}.html" class="pred-card-link">
                <h3><span class="badge badge-result">Post-Race</span> Prediction Results</h3>
                <p class="pred-date">Score for {label} prediction</p>
                <div class="score-display">{total} / 25</div>
            </a>
        </div>
"""

    # Sprint score card (if sprint weekend, scored)
    sprint_score_card = ""
    if "post-sprint-qualifying" in scores:
        sq = scores["post-sprint-qualifying"]
        sprint_total = sq.get("sprint_score", 0)
        sprint_score_card = f"""        <div class="pred-card pred-card-sprint">
            <a href="{date_str}_score_sprint.html" class="pred-card-link">
                <h3><span class="badge badge-sprint">Post-Sprint</span> Sprint Race Results</h3>
                <p class="pred-date">Score for Post-Sprint-Q prediction</p>
                <div class="score-display">{sprint_total} / 15</div>
            </a>
        </div>
"""

    # Build cards in chronological order: predictions + score cards interleaved
    # Sprint score card appears between post-sprint-Q and pre-Q (the order they're made)
    pred_links = ""
    for ptype in pred_order:
        if ptype in predictions:
            badge_cls = badge_classes[ptype]
            card_cls = card_border_classes[ptype]
            badge_label = pred_labels[ptype]
            pred_links += f"""        <div class="pred-card {card_cls}">
            <a href="{date_str}_{ptype}.html" class="pred-card-link">
                <h3><span class="badge {badge_cls}">{badge_label}</span> Prediction</h3>
                <p class="pred-date">Generated: {predictions[ptype].get('generated_at', 'N/A')}</p>
                <p class="pred-count">{len(predictions[ptype].get('predictions', []))} drivers predicted</p>
            </a>
        </div>
"""
            # Insert sprint score card after post-sprint-qualifying prediction
            if ptype == "post-sprint-qualifying" and sprint_score_card:
                pred_links += sprint_score_card
                sprint_score_card = ""  # Don't duplicate at the end
            # Insert main race score card after post-qualifying prediction
            if ptype == "post-qualifying" and score_card:
                pred_links += score_card
                score_card = ""

    standings_html = ""
    if "standings" in weekend_data:
        standings_html = """
        <section class="standings-section">
            <h2>Current Championship Standings</h2>
            <div class="standings-grid">"""
        for d in weekend_data["standings"][:10]:
            standings_html += f"""
                <div class="standing-item">
                    <span class="pos">{d['position']}</span>
                    <span class="name">{d['name']}</span>
                    <span class="team">{d['team']}</span>
                    <span class="pts">{d['points']}pts</span>
                </div>"""
        standings_html += """
            </div>
        </section>"""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{race_name} Predictions | Henry AI F1</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }}
        .back-link {{
            display: inline-block;
            color: #888;
            text-decoration: none;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }}
        .back-link:hover {{ color: #fff; }}
        .page-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.25rem;
            gap: 1rem;
        }}
        .page-header h1 {{
            font-size: 2rem;
            color: #fff;
            margin: 0;
        }}
        .race-subtitle {{
            color: #666;
            font-size: 0.9rem;
            font-weight: 400;
            margin-top: 0.15rem;
            margin-bottom: 0;
        }}
        .page-circuit {{
            flex-shrink: 0;
            width: 80px;
            height: 80px;
            object-fit: contain;
            filter: brightness(0.8);
        }}
        .race-date {{
            color: #888;
            margin-bottom: 2rem;
        }}
        .pred-card {{
            background: #111118;
            border: 1px solid #222;
            border-left: 3px solid;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .pred-card:hover {{
            border-color: #444;
            box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        }}
        .pred-card-early {{ border-left-color: #93c5fd; }}
        .pred-card-early:hover {{ border-color: #93c5fd; }}
        .pred-card-prepr {{ border-left-color: #6ee7b7; }}
        .pred-card-prepr:hover {{ border-color: #6ee7b7; }}
        .pred-card-presq {{ border-left-color: #f0abfc; }}
        .pred-card-presq:hover {{ border-color: #f0abfc; }}
        .pred-card-postsq {{ border-left-color: #c4b5fd; }}
        .pred-card-postsq:hover {{ border-color: #c4b5fd; }}
        .pred-card-preq {{ border-left-color: #93c5fd; }}
        .pred-card-preq:hover {{ border-color: #93c5fd; }}
        .pred-card-postq {{ border-left-color: #6ee7b7; }}
        .pred-card-postq:hover {{ border-color: #6ee7b7; }}
        .pred-card-result {{
            border-left-color: #ffd700;
            border-width: 3px;
            background: linear-gradient(135deg, #1a1508 0%, #111118 100%);
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.08);
        }}
        .pred-card-result:hover {{
            border-color: #ffd700;
            box-shadow: 0 0 30px rgba(255, 215, 0, 0.15);
        }}
        .badge-result {{ background: #3a3000; color: #ffd700; }}
        .pred-card-sprint {{
            border-color: #1e3a5f;
        }}
        .pred-card-sprint:hover {{
            border-color: #3b82f6;
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.15);
        }}
        .badge-sprint {{ background: #1e3a5f; color: #60a5fa; }}
        .score-display {{
            font-size: 2rem;
            font-weight: 800;
            color: #ffd700;
            margin: 0.5rem 0;
        }}
        .pred-card h3 {{
            color: #fff;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            flex-shrink: 0;
        }}
        .badge-early {{ background: #1e3a5f; color: #93c5fd; }}
        .badge-presq {{ background: #4a1942; color: #f0abfc; }}
        .badge-postsq {{ background: #3b1f5e; color: #c4b5fd; }}
        .badge-preq {{ background: #1f3b5e; color: #93c5fd; }}
        .badge-postq {{ background: #1a3a2a; color: #6ee7b7; }}
        .pred-card-link {{
            text-decoration: none;
            color: inherit;
        }}
        .pred-date {{ color: #888; font-size: 0.85rem; }}
        .pred-count {{ color: #60a5fa; font-size: 0.85rem; }}
        .standings-section {{
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #222;
        }}
        .standings-section h2 {{
            color: #fff;
            margin-bottom: 1rem;
        }}
        .standings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 0.5rem;
        }}
        .standing-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: #111118;
            border-radius: 4px;
        }}
        .standing-item .pos {{
            color: #60a5fa;
            font-weight: bold;
            min-width: 1.5rem;
        }}
        .standing-item .name {{ flex: 1; }}
        .standing-item .team {{ color: #888; font-size: 0.85rem; }}
        .standing-item .pts {{ color: #6ee7b7; font-size: 0.85rem; }}
        footer {{
            text-align: center;
            padding: 2rem 0;
            color: #555;
            font-size: 0.85rem;
            border-top: 1px solid #222;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">&larr; Back to all races</a>
        <div class="page-header">
            <div>
                <h1>{short_name}</h1>
                <p class="race-subtitle">{race_name}</p>
            </div>
            {f'<img src="{svg_url}" alt="{short_name} circuit" class="page-circuit">' if svg_url else ''}
        </div>
        <p class="race-date">{date_str}</p>
        
        <div class="predictions-grid">
{pred_links}
{score_card}
{sprint_score_card}
        </div>
        
{standings_html}
        
        <footer>
            <p>Generated by Henry AI &middot; Powered by data from F1DB &amp; BBC Sport</p>
        </footer>
    </div>
</body>
</html>"""

def generate_prediction_html(prediction_data):
    """Generate a prediction detail page showing top 10."""
    predictions = prediction_data.get("predictions", [])
    ptype = prediction_data.get("type", "unknown")
    race_name = prediction_data.get("race_name", "F1 Race")
    short_name = prediction_data.get("short_name", race_name)
    generated_at = prediction_data.get("generated_at", "Unknown")
    reasoning = prediction_data.get("reasoning", "")
    date_str = prediction_data.get("date", "")

    labels = {
        "early": "Early Prediction",
        "pre-practice": "Pre-Practice Prediction",
        "pre-qualifying": "Pre-Qualifying Prediction",
        "post-qualifying": "Post-Qualifying Prediction",
        "pre-sprint-qualifying": "Pre-Sprint-Qualifying Prediction",
        "post-sprint-qualifying": "Post-Sprint-Qualifying Prediction"
    }
    
    reasoning_block = ""
    if reasoning:
        reasoning_block = '        <div class="overall-reasoning">\n            <h3>Analysis</h3>\n            <p>' + reasoning + '</p>\n        </div>\n\n'
    
    rows = ""
    for p in predictions:
        pos = p.get("position", "?")
        driver = p.get("driver", "Unknown")
        team = p.get("team", "")
        confidence = p.get("confidence", "medium")
        pred_reasoning = p.get("reasoning", "")
        
        conf_colors = {
            "high": "#6ee7b7",
            "medium": "#fbbf24",
            "low": "#f87171"
        }
        conf_color = conf_colors.get(confidence, "#888")
        
        pos_colors = {
            "1": "#ffd700",
            "2": "#c0c0c0",
            "3": "#cd7f32"
        }
        pos_color = pos_colors.get(str(pos), "#60a5fa")
        
        rows += f"""        <tr class="pred-row">
            <td rowspan="2" style="color: {pos_color}; font-weight: bold; font-size: 1.2rem;">{pos}</td>
            <td><strong>{driver}</strong><br><span class="driver-team">{team}</span></td>
            <td><span style="color: {conf_color};">{confidence.title()}</span></td>
        </tr>
        <tr class="reasoning-row">
            <td colspan="3" style="color: #888; font-size: 0.9rem; padding-top: 0.25rem; padding-bottom: 1rem;">{pred_reasoning}</td>
        </tr>
"""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{labels.get(ptype, ptype)} | {race_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }}
        .back-link {{
            display: inline-block;
            color: #888;
            text-decoration: none;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }}
        .back-link:hover {{ color: #fff; }}
        h1 {{
            font-size: 1.75rem;
            color: #fff;
            margin-bottom: 0.15rem;
        }}
        .race-subtitle {{
            color: #666;
            font-size: 0.85rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }}
        .meta {{
            color: #888;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}
        .type-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }}
        .type-early {{ background: #1e3a5f; color: #93c5fd; }}
        .type-pre-practice {{ background: #1a3a3a; color: #6ee7b7; }}
        .type-pre-sprint-qualifying {{ background: #4a1942; color: #f0abfc; }}
        .type-post-sprint-qualifying {{ background: #3b1f5e; color: #c4b5fd; }}
        .type-pre-qualifying {{ background: #1f3b5e; color: #93c5fd; }}
        .type-post-qualifying {{ background: #1a3a2a; color: #6ee7b7; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
        }}
        th {{
            text-align: left;
            padding: 0.75rem 1rem;
            background: #1a1a2e;
            color: #aaa;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 0.75rem 1rem;
        }}
        tr:hover {{ background: #111118; }}
        .driver-team {{
            color: #888;
            font-size: 0.85rem;
        }}
        .reasoning-row {{
            background: #0d0d12;
            border-bottom: 1px solid #222;
        }}
        .reasoning-row:hover {{
            background: #0d0d12;
        }}
        .overall-reasoning {{
            background: #111118;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .overall-reasoning h3 {{
            color: #fff;
            margin-bottom: 0.75rem;
        }}
        .overall-reasoning p {{
            color: #aaa;
            font-size: 0.95rem;
        }}
        footer {{
            text-align: center;
            padding: 2rem 0;
            color: #555;
            font-size: 0.85rem;
            border-top: 1px solid #222;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="{date_str}.html" class="back-link">&larr; Back to {short_name}</a>
        <h1>{labels.get(ptype, ptype)} — {short_name}</h1>
        <p class="race-subtitle">{race_name}</p>
        <p class="meta">Generated: {generated_at}</p>
        <span class="type-badge type-{ptype}">{labels.get(ptype, ptype).upper()}</span>
        
{reasoning_block}
        <table>
            <thead>
                <tr>
                    <th>Pos</th>
                    <th>Driver</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
{rows}
            </tbody>
        </table>
        
        <footer>
            <p>Generated by Henry AI &middot; Powered by data from F1DB &amp; BBC Sport</p>
        </footer>
    </div>
</body>
</html>"""

def generate_score_html(score_data):
    """Generate a score detail page showing prediction vs results comparison."""
    score = score_data["score"]
    race_name = score_data.get("race_name", "F1 Race")
    short_name = score_data.get("short_name", race_name)
    date_str = score_data.get("date", "")
    prediction_type = score_data.get("prediction_type", "post-qualifying")
    scored_at = score_data.get("scored_at", "Unknown")
    pred_generated = score_data.get("prediction_generated", "Unknown")

    labels = {
        "early": "Early Prediction",
        "pre-practice": "Pre-Practice Prediction",
        "pre-qualifying": "Pre-Qualifying Prediction",
        "post-qualifying": "Post-Qualifying Prediction",
        "pre-sprint-qualifying": "Pre-Sprint-Qualifying Prediction",
        "post-sprint-qualifying": "Post-Sprint-Qualifying Prediction"
    }
    label = labels.get(prediction_type, prediction_type)

    # Score ring color
    total = score["total_score"]
    if total >= 20:
        ring_color = "#ffd700"  # gold
    elif total >= 15:
        ring_color = "#6ee7b7"  # green
    elif total >= 10:
        ring_color = "#60a5fa"  # blue
    elif total >= 5:
        ring_color = "#fbbf24"  # amber
    else:
        ring_color = "#f87171"  # red

    # Position detail rows
    pos_rows = ""
    for detail in score["position_details"]:
        pos = detail["position"]
        pred = detail["predicted"]
        actual = detail["actual"]
        correct = detail["correct"]
        pts = detail["points"]

        pos_colors = {
            "1": "#ffd700", "2": "#c0c0c0", "3": "#cd7f32"
        }
        pos_color = pos_colors.get(str(pos), "#60a5fa")

        if correct:
            row_class = "score-row-correct"
            status_icon = "&#10003;"  # checkmark
            status_color = "#6ee7b7"
        else:
            row_class = "score-row-wrong"
            status_icon = "&#10007;"  # X
            status_color = "#f87171"

        pos_rows += f"""        <tr class="{row_class}">
            <td class="pos-cell" style="color: {pos_color};">{pos}</td>
            <td class="pred-driver">{pred}</td>
            <td class="vs-cell">vs</td>
            <td class="actual-driver">{actual}</td>
            <td class="status-cell" style="color: {status_color};">{status_icon} {pts}/2</td>
        </tr>
"""

    # Podium section
    podium_correct = score["podium_correct"]
    podium_class = "podium-section-correct" if podium_correct else "podium-section-wrong"
    podium_icon = "&#10003;" if podium_correct else "&#10007;"
    podium_color = "#6ee7b7" if podium_correct else "#f87171"

    pred_podium_html = ", ".join(score["predicted_podium"])
    actual_podium_html = ", ".join(score["actual_podium"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Score: {label} | {race_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }}
        .back-link {{
            display: inline-block;
            color: #888;
            text-decoration: none;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }}
        .back-link:hover {{ color: #fff; }}
        h1 {{
            font-size: 1.75rem;
            color: #fff;
            margin-bottom: 0.15rem;
        }}
        .race-subtitle {{
            color: #666;
            font-size: 0.85rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }}
        .meta {{
            color: #888;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}
        .score-header {{
            display: flex;
            align-items: center;
            gap: 2.5rem;
            margin-bottom: 2rem;
            padding: 2rem;
            background: #111118;
            border: 1px solid #222;
            border-radius: 12px;
        }}
        .score-ring {{
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 4px solid {ring_color};
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 0 30px rgba({ring_color[:3]}{ring_color[3] if len(ring_color) > 3 else '0'}, 0.2);
        }}
        .score-ring .score-num {{
            font-size: 3rem;
            font-weight: 800;
            color: {ring_color};
            line-height: 1;
        }}
        .score-ring .score-max {{
            font-size: 0.9rem;
            color: #888;
            margin-top: 0.25rem;
        }}
        .score-breakdown {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        .score-line {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .score-line .label {{
            color: #aaa;
            font-size: 0.95rem;
            min-width: 120px;
        }}
        .score-line .bar-bg {{
            flex: 1;
            height: 8px;
            background: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
        }}
        .score-line .bar-fill {{
            height: 100%;
            border-radius: 4px;
            background: {ring_color};
        }}
        .score-line .value {{
            color: #fff;
            font-weight: 600;
            min-width: 50px;
            text-align: right;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
        }}
        th {{
            text-align: left;
            padding: 0.75rem 1rem;
            background: #1a1a2e;
            color: #aaa;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 0.75rem 1rem;
        }}
        .pos-cell {{
            font-weight: bold;
            font-size: 1.1rem;
            width: 50px;
        }}
        .pred-driver {{
            color: #93c5fd;
        }}
        .actual-driver {{
            color: #6ee7b7;
        }}
        .vs-cell {{
            color: #555;
            font-size: 0.8rem;
            text-align: center;
            width: 40px;
        }}
        .status-cell {{
            text-align: right;
            font-weight: 600;
            width: 80px;
        }}
        .score-row-correct {{
            background: #0d1a0d;
        }}
        .score-row-wrong {{
            background: #0d0d12;
        }}
        tr:hover {{
            background: #111118 !important;
        }}
        .podium-section {{
            background: #111118;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .podium-section-correct {{
            border-color: #6ee7b7;
        }}
        .podium-section-wrong {{
            border-color: #f87171;
        }}
        .podium-section h3 {{
            color: #fff;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .podium-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
        }}
        .podium-row .label {{
            color: #888;
            font-size: 0.9rem;
        }}
        .podium-row .drivers {{
            color: #fff;
            font-weight: 500;
        }}
        .section-label {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
            background: #1a1a2e;
            color: #aaa;
        }}
        footer {{
            text-align: center;
            padding: 2rem 0;
            color: #555;
            font-size: 0.85rem;
            border-top: 1px solid #222;
            margin-top: 2rem;
        }}
        @media (max-width: 640px) {{
            .container {{
                padding: 1rem 0.5rem;
            }}
            h1 {{
                font-size: 1.35rem;
            }}
            .score-header {{
                flex-direction: column;
                align-items: center;
                gap: 1.25rem;
                padding: 1.25rem;
            }}
            .score-ring {{
                width: 110px;
                height: 110px;
            }}
            .score-ring .score-num {{
                font-size: 2.25rem;
            }}
            .score-line .label {{
                min-width: 0;
                font-size: 0.85rem;
            }}
            table {{
                display: block;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }}
            .pred-driver, .actual-driver {{
                word-break: break-word;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="{date_str}.html" class="back-link">&larr; Back to {short_name}</a>
        <h1>Prediction Score — {short_name}</h1>
        <p class="race-subtitle">{race_name}</p>
        <p class="meta">{label} &middot; Generated: {pred_generated} &middot; Scored: {scored_at}</p>

        <div class="score-header">
            <div class="score-ring">
                <span class="score-num">{total}</span>
                <span class="score-max">out of 25</span>
            </div>
            <div class="score-breakdown">
                <div class="score-line">
                    <span class="label">Positions (1-10)</span>
                    <div class="bar-bg"><div class="bar-fill" style="width: {(score['position_score'] / 20) * 100}%"></div></div>
                    <span class="value">{score['position_score']}/20</span>
                </div>
                <div class="score-line">
                    <span class="label">Podium (top 3)</span>
                    <div class="bar-bg"><div class="bar-fill" style="width: {(score['podium_score'] / 5) * 100}%"></div></div>
                    <span class="value">{score['podium_score']}/5</span>
                </div>
            </div>
        </div>

        <span class="section-label">POSITION-BY-POSITION BREAKDOWN</span>
        <table>
            <thead>
                <tr>
                    <th>Pos</th>
                    <th>Predicted</th>
                    <th></th>
                    <th>Actual</th>
                    <th style="text-align: right;">Points</th>
                </tr>
            </thead>
            <tbody>
{pos_rows}
            </tbody>
        </table>

        <span class="section-label">PODIUM CHECK</span>
        <div class="podium-section {podium_class}">
            <h3 style="color: {podium_color};">{podium_icon} Podium Result</h3>
            <div class="podium-row">
                <span class="label">Predicted podium:</span>
                <span class="drivers">{pred_podium_html}</span>
            </div>
            <div class="podium-row">
                <span class="label">Actual podium:</span>
                <span class="drivers">{actual_podium_html}</span>
            </div>
            <div class="podium-row">
                <span class="label">Score:</span>
                <span class="drivers" style="color: {podium_color};">{score['podium_score']}/5 points</span>
            </div>
        </div>

        <footer>
            <p>Generated by Henry AI &middot; Powered by data from F1DB &amp; BBC Sport</p>
        </footer>
    </div>
</body>
</html>"""

def generate_sprint_score_html(score_data):
    """Generate a sprint score detail page showing prediction vs sprint results."""
    score = score_data["score"]
    race_name = score_data.get("race_name", "F1 Race")
    short_name = score_data.get("short_name", race_name)
    date_str = score_data.get("date", "")
    prediction_type = score_data.get("prediction_type", "post-sprint-qualifying")
    scored_at = score_data.get("scored_at", "Unknown")
    pred_generated = score_data.get("prediction_generated", "Unknown")

    labels = {
        "post-sprint-qualifying": "Post-Sprint-Qualifying Sprint Score",
        "pre-sprint-qualifying": "Pre-Sprint-Qualifying Sprint Score"
    }
    label = labels.get(prediction_type, "Sprint Score")

    # Score ring color (scaled for /15 max)
    total = score["sprint_score"]
    if total >= 13:
        ring_color = "#ffd700"  # gold
    elif total >= 10:
        ring_color = "#6ee7b7"  # green
    elif total >= 7:
        ring_color = "#60a5fa"  # blue
    elif total >= 4:
        ring_color = "#fbbf24"  # amber
    else:
        ring_color = "#f87171"  # red

    # Position detail rows
    pos_rows = ""
    for detail in score["position_details"]:
        pos = detail["position"]
        pred = detail["predicted"]
        actual = detail["actual"]
        correct = detail["correct"]
        pts = detail["points"]

        pos_colors = {
            "1": "#ffd700", "2": "#c0c0c0", "3": "#cd7f32"
        }
        pos_color = pos_colors.get(str(pos), "#60a5fa")

        if correct:
            row_class = "score-row-correct"
            status_icon = "&#10003;"
            status_color = "#6ee7b7"
        else:
            row_class = "score-row-wrong"
            status_icon = "&#10007;"
            status_color = "#f87171"

        pos_rows += f"""        <tr class="{row_class}">
            <td class="pos-cell" style="color: {pos_color};">{pos}</td>
            <td class="pred-driver">{pred}</td>
            <td class="vs-cell">vs</td>
            <td class="actual-driver">{actual}</td>
            <td class="status-cell" style="color: {status_color};">{status_icon} {pts}/1</td>
        </tr>
"""

    # Podium section
    podium_correct = score["podium_correct"]
    podium_class = "podium-section-correct" if podium_correct else "podium-section-wrong"
    podium_icon = "&#10003;" if podium_correct else "&#10007;"
    podium_color = "#6ee7b7" if podium_correct else "#f87171"

    pred_podium_html = ", ".join(score["predicted_podium"])
    actual_podium_html = ", ".join(score["actual_podium"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sprint Score: {label} | {race_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }}
        .back-link {{
            display: inline-block;
            color: #888;
            text-decoration: none;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }}
        .back-link:hover {{ color: #fff; }}
        .score-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        .score-header h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}
        .score-header .subtitle {{
            color: #888;
            font-size: 0.95rem;
        }}
        .score-ring {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2rem 0;
        }}
        .score-ring-circle {{
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 6px solid {ring_color};
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 30px rgba(0,0,0,0.3), 0 0 60px {ring_color}33;
        }}
        .score-ring-num {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {ring_color};
        }}
        .score-ring-max {{
            font-size: 0.9rem;
            color: #888;
        }}
        .score-breakdown {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1.5rem 0;
        }}
        .score-breakdown-item {{
            text-align: center;
        }}
        .score-breakdown-val {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #60a5fa;
        }}
        .score-breakdown-label {{
            font-size: 0.8rem;
            color: #888;
            text-transform: uppercase;
        }}
        .section-label {{
            display: block;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #666;
            margin: 2rem 0 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            text-align: left;
            padding: 0.75rem 1rem;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #888;
            border-bottom: 1px solid #333;
        }}
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #1a1a2e;
        }}
        .pos-cell {{
            font-weight: 700;
            font-size: 1.1rem;
            width: 50px;
            text-align: center;
        }}
        .pred-driver {{ color: #a78bfa; }}
        .vs-cell {{
            color: #666;
            font-size: 0.8rem;
            text-align: center;
        }}
        .actual-driver {{ color: #60a5fa; }}
        .status-cell {{
            text-align: right;
            font-weight: 600;
        }}
        .score-row-correct {{ background: #6ee7b711; }}
        .score-row-wrong {{ background: transparent; }}
        .podium-section {{
            background: #111827;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
        }}
        .podium-section-correct {{ border: 1px solid #6ee7b744; }}
        .podium-section-wrong {{ border: 1px solid #f8717144; }}
        .podium-section h3 {{
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        .podium-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #1a1a2e;
        }}
        .podium-row:last-child {{ border-bottom: none; }}
        .podium-row .label {{ color: #888; }}
        .podium-row .drivers {{ color: #e0e0e0; }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #222;
            color: #666;
            font-size: 0.85rem;
        }}
        @media (max-width: 600px) {{
            .score-header h1 {{ font-size: 1.3rem; }}
            td {{ padding: 0.5rem; font-size: 0.9rem; }}
            .score-breakdown {{ gap: 1rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">&larr; Back to all races</a>

        <div class="score-header">
            <h1>Sprint Score: {short_name}</h1>
            <div class="subtitle">{label} &middot; {date_str}</div>
        </div>

        <div class="score-ring">
            <div class="score-ring-circle">
                <span class="score-ring-num">{total}</span>
                <span class="score-ring-max">of 15</span>
            </div>
        </div>

        <div class="score-breakdown">
            <div class="score-breakdown-item">
                <div class="score-breakdown-val">{score['position_score']}/10</div>
                <div class="score-breakdown-label">Positions</div>
            </div>
            <div class="score-breakdown-item">
                <div class="score-breakdown-val">{score['podium_score']}/5</div>
                <div class="score-breakdown-label">Podium</div>
            </div>
        </div>

        <span class="section-label">POSITION DETAILS</span>
        <table>
            <thead>
                <tr>
                    <th>Pos</th>
                    <th>Predicted</th>
                    <th></th>
                    <th>Actual</th>
                    <th style="text-align:right;">Pts</th>
                </tr>
            </thead>
            <tbody>
{pos_rows}            </tbody>
        </table>

        <span class="section-label">PODIUM CHECK</span>
        <div class="podium-section {podium_class}">
            <h3 style="color: {podium_color};">{podium_icon} Podium Result</h3>
            <div class="podium-row">
                <span class="label">Predicted podium:</span>
                <span class="drivers">{pred_podium_html}</span>
            </div>
            <div class="podium-row">
                <span class="label">Actual podium:</span>
                <span class="drivers">{actual_podium_html}</span>
            </div>
            <div class="podium-row">
                <span class="label">Score:</span>
                <span class="drivers" style="color: {podium_color};">{score['podium_score']}/5 points</span>
            </div>
        </div>

        <footer>
            <p>Sprint Score by Henry AI &middot; Powered by data from F1DB &amp; BBC Sport</p>
        </footer>
    </div>
</body>
</html>"""

# ── Storage ────────────────────────────────────────────────────────────────

def load_history(repo_path):
    """Load existing prediction history from JSON file."""
    history_file = Path(repo_path) / "data" / "history.json"
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return {"weekends": []}

def save_history(repo_path, history):
    """Save prediction history to JSON file."""
    history_file = Path(repo_path) / "data" / "history.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

# ── Main workflow ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Race Prediction Engine")
    parser.add_argument("--type", choices=["early", "pre-practice", "pre-sprint-qualifying", "post-sprint-qualifying", "pre-qualifying", "post-qualifying"],
                        help="Prediction type to generate")
    parser.add_argument("--detect", action="store_true",
                        help="Auto-detect prediction type from schedule")
    parser.add_argument("--repo", default="/home/debadmin/projects/ai-henry-f1-predictions",
                        help="Path to the GitHub Pages repo")
    parser.add_argument("--publish", action="store_true",
                        help="Publish to GitHub after generating")
    parser.add_argument("--check-session", action="store_true",
                        help="Check if required session data exists in F1DB and exit")
    parser.add_argument("--score", action="store_true",
                        help="Score a prediction against actual race results (run 24h after race)")
    parser.add_argument("--score-sprint", action="store_true",
                        help="Score post-sprint-qualifying prediction against sprint race results (1pt per position, max 15)")
    args = parser.parse_args()

    # --check-session: pre-flight check for required session data
    if args.check_session:
        if not args.type:
            print("Error: --check-session requires --type", file=sys.stderr)
            sys.exit(1)
        pred_type = args.type
        # Resolve race
        schedule = get_schedule()
        check_slug = None
        race_sessions = []
        if schedule:
            for cat in schedule:
                if cat["category"] == "f1":
                    rn = cat.get("next_race", {}).get("name", "")
                    check_slug = find_race_slug(rn)
                    race_sessions = cat.get("next_race", {}).get("sessions", [])
                    break
        if not check_slug:
            print("SESSION_MISSING:no_race")
            sys.exit(1)
        # Build session timing map (BST timezone)
        session_timing = {}
        bst = timezone(timedelta(hours=1))
        for s in race_sessions:
            dt = datetime.fromisoformat(s["datetime"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=bst)
            session_timing[s["name"].lower()] = dt
        now = datetime.now(bst)
        # Buffer: wait 4 hours after session start before checking F1DB for results
        # F1DB can be slow to publish data, especially for sprint sessions
        session_buffer = timedelta(hours=4)
        def session_ready_for_check(session_name):
            """Check if enough time has passed since session start for F1DB to have data."""
            lower = session_name.lower()
            for sname, sdt in session_timing.items():
                if lower in sname:
                    ready_time = sdt + session_buffer
                    if now < ready_time:
                        return False, ready_time
                    return True, ready_time
            return True, None  # No timing info, proceed anyway
        # Check required sessions
        sprint_weekend = is_sprint_race(check_slug)

        if pred_type == "pre-sprint-qualifying":
            ready, _ = session_ready_for_check("practice1")
            if not ready:
                print("SESSION_NOT_READY:pre-sprint-qualifying,practice1")
                sys.exit(2)
            missing = []
            if not check_session_exists(check_slug, "practice1"):
                missing.append("practice1")
            if missing:
                print(f"SESSION_MISSING:pre-sprint-qualifying,{','.join(missing)}")
                sys.exit(1)
            print("SESSION_READY:pre-sprint-qualifying")
            sys.exit(0)
        elif pred_type == "post-sprint-qualifying":
            ready, _ = session_ready_for_check("sprint qualifying")
            if not ready:
                print("SESSION_NOT_READY:post-sprint-qualifying,sprint_qualifying")
                sys.exit(2)
            missing = []
            if not check_session_exists(check_slug, "practice1"):
                missing.append("practice1")
            if not check_session_exists(check_slug, "sprint_qualifying"):
                missing.append("sprint_qualifying")
            if missing:
                print(f"SESSION_MISSING:post-sprint-qualifying,{','.join(missing)}")
                sys.exit(1)
            print("SESSION_READY:post-sprint-qualifying")
            sys.exit(0)
        elif pred_type == "pre-qualifying":
            # For sprint weekends, skip the timing check on sprint session —
            # the 4h buffer from session start is too conservative (sprint starts 12:00,
            # ends ~13:00, but buffer doesn't clear until 16:00).
            # check_session_exists below will verify F1DB actually has the data.
            # For normal weekends, still check practice3 timing.
            if not sprint_weekend:
                ready, _ = session_ready_for_check("practice3")
                if not ready:
                    print("SESSION_NOT_READY:pre-qualifying,practice3")
                    sys.exit(2)
            missing = []
            if sprint_weekend:
                if not check_session_exists(check_slug, "practice1"):
                    missing.append("practice1")
                if not check_session_exists(check_slug, "sprint_qualifying"):
                    missing.append("sprint_qualifying")
                if not check_session_exists(check_slug, "sprint_race"):
                    missing.append("sprint_race")
            else:
                for p in ["practice1", "practice2", "practice3"]:
                    if not check_session_exists(check_slug, p):
                        missing.append(p)
            if missing:
                print(f"SESSION_MISSING:pre-qualifying,{','.join(missing)}")
                sys.exit(1)
            print("SESSION_READY:pre-qualifying")
            sys.exit(0)
        elif pred_type == "post-qualifying":
            ready, _ = session_ready_for_check("qualifying")
            if not ready:
                print("SESSION_NOT_READY:post-qualifying,qualifying")
                sys.exit(2)
            missing = []
            if not check_session_exists(check_slug, "qualifying"):
                missing.append("qualifying")
            if missing:
                print(f"SESSION_MISSING:post-qualifying,{','.join(missing)}")
                sys.exit(1)
            print("SESSION_READY:post-qualifying")
            sys.exit(0)
        elif pred_type == "pre-practice":
            # Pre-practice runs before FP1 — no session data needed
            print("SESSION_READY:pre-practice")
            sys.exit(0)
        else:
            # early predictions don't need session data
            print("SESSION_READY:early")
            sys.exit(0)

    # --score: score a prediction against actual race results
    if args.score:
        if not args.type:
            print("Error: --score requires --type", file=sys.stderr)
            sys.exit(1)
        pred_type = args.type
        repo_path = args.repo

        # Load history and find an unscored prediction of this type.
        # History is newest-first, so scan from the END: pick the OLDEST unscored
        # candidate. A future race (no results published yet) must not block
        # scoring of a past race that is overdue.
        history = load_history(repo_path)
        weekend_entry = None
        for w in reversed(history["weekends"]):
            has_prediction = pred_type in w.get("predictions", {})
            has_score = pred_type in w.get("scores", {})
            if has_prediction and not has_score:
                weekend_entry = w
                break

        if not weekend_entry:
            print(f"No unscored {pred_type} predictions found in history", file=sys.stderr)
            sys.exit(1)

        race_name = weekend_entry["race_name"]
        race_date = weekend_entry["date"]
        print(f"Scoring weekend: {race_name} ({race_date})")

        # Find the F1DB slug for this race
        race_slug = find_race_slug(race_name)
        if not race_slug:
            # Fallback: try all slugs and pick the one with race results
            print(f"Warning: Could not find slug for '{race_name}', trying all slugs...")
            all_slugs = get_all_race_slugs()
            for slug in all_slugs:
                if get_race_results(slug):
                    race_slug = slug
                    print(f"  Found slug with results: {race_slug}")
                    break

        if not race_slug:
            print(f"Error: Could not find F1DB slug for {race_name}", file=sys.stderr)
            sys.exit(1)

        # Fetch race result
        print(f"Fetching race result for {race_slug}...")
        race_result = get_race_results(race_slug)
        if not race_result:
            print("Error: No race result found — race may not have finished yet", file=sys.stderr)
            sys.exit(1)

        print(f"  Found {len(race_result)} classified drivers")

        # Get the prediction to score
        pred_data = weekend_entry["predictions"].get(pred_type)
        if not pred_data:
            print(f"Error: No {pred_type} prediction found for this weekend", file=sys.stderr)
            sys.exit(1)

        predictions = pred_data["predictions"]
        print(f"Scoring {pred_type} prediction ({len(predictions)} drivers)...")

        # Build actual results lookup: driver_name -> position
        # Also build a normalized lookup for fuzzy matching (handles "Oscar Jack Piastri" vs "Oscar Piastri")
        actual_results = {}
        actual_norm = {}
        for entry in race_result:
            driver = entry.get("name", "")
            pos = entry.get("position", "")
            if driver and pos:
                actual_results[driver] = int(pos) if isinstance(pos, str) and pos.isdigit() else pos
                # Normalized key: first + last name, lowercase, no suffixes/middle names
                parts = driver.lower().replace("jr.", "").replace("sr.", "").replace("jr", "").replace("sr", "").split()
                if len(parts) >= 2:
                    norm_key = f"{parts[0]} {parts[-1]}"
                    actual_norm[norm_key] = actual_results[driver]

        # Score the prediction
        position_score = 0
        position_details = []
        predicted_podium = []
        actual_podium = []

        # Get actual podium (top 3)
        for entry in race_result[:3]:
            driver = entry.get("name", "")
            if driver:
                actual_podium.append(driver)

        # Score each predicted position (1-10)
        for pred in predictions[:10]:
            pred_driver = pred.get("driver", "")
            pred_pos = pred.get("position", 0)

            # Try exact match first, then normalized (first+last name)
            actual_pos = actual_results.get(pred_driver)
            if actual_pos is None:
                norm_parts = pred_driver.lower().split()
                if len(norm_parts) >= 2:
                    norm_key = f"{norm_parts[0]} {norm_parts[-1]}"
                    actual_pos = actual_norm.get(norm_key)

            correct = (actual_pos == pred_pos)
            pts = 2 if correct else 0
            position_score += pts

            position_details.append({
                "position": pred_pos,
                "predicted": pred_driver,
                "actual": actual_pos if actual_pos is not None else "N/A",
                "correct": correct,
                "points": pts
            })

            if pred_pos <= 3:
                predicted_podium.append(pred_driver)

        # Podium score: 5 points if all 3 podium drivers predicted correctly (any order)
        podium_score = 0
        podium_correct = False
        if len(predicted_podium) == 3 and len(actual_podium) == 3:
            podium_correct = set(predicted_podium) == set(actual_podium)
            podium_score = 5 if podium_correct else 0

        total_score = position_score + podium_score

        scored_at = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
        score_data = {
            "total_score": total_score,
            "position_score": position_score,
            "podium_score": podium_score,
            "position_details": position_details,
            "predicted_podium": predicted_podium,
            "actual_podium": actual_podium,
            "podium_correct": podium_correct,
            "scored_at": scored_at
        }

        # Save score to weekend entry
        if "scores" not in weekend_entry:
            weekend_entry["scores"] = {}
        weekend_entry["scores"][pred_type] = score_data

        save_history(repo_path, history)
        print(f"\nScore: {total_score}/25 (Positions: {position_score}/20, Podium: {podium_score}/5)")

        # Generate score detail page
        short_name = _get_short_name(weekend_entry.get("circuit_id", ""), weekend_entry["race_name"])
        score_html = generate_score_html({
            "score": score_data,
            "race_name": weekend_entry["race_name"],
            "short_name": short_name,
            "date": race_date,
            "prediction_type": pred_type,
            "scored_at": scored_at,
            "prediction_generated": pred_data.get("generated_at", "Unknown")
        })
        score_path = Path(repo_path) / "weekends" / f"{race_date}_score_{pred_type}.html"
        with open(score_path, 'w') as f:
            f.write(score_html)
        print(f"Score detail page: {score_path}")

        # Regenerate weekend page (now includes score card)
        weekend_html = generate_weekend_html(weekend_entry)
        weekend_path = Path(repo_path) / "weekends" / f"{race_date}.html"
        with open(weekend_path, 'w') as f:
            f.write(weekend_html)

        # Regenerate index
        index_html = generate_index_html(history["weekends"])
        index_path = Path(repo_path) / "index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)

        print("Website regenerated with score card")

        # Publish if requested
        if args.publish:
            print("\nPublishing to GitHub...")
            publish_to_github(repo_path, race_name)

        print("\nDone!")
        sys.exit(0)

    # --score-sprint: score post-sprint-qualifying prediction against sprint race results
    if args.score_sprint:
        repo_path = args.repo

        # Load history and find an unscored post-sprint-qualifying prediction.
        # History is newest-first, so scan from the END: pick the OLDEST unscored
        # candidate (a future sprint with no results must not block past ones).
        history = load_history(repo_path)
        weekend_entry = None
        for w in reversed(history["weekends"]):
            has_prediction = "post-sprint-qualifying" in w.get("predictions", {})
            has_score = "post-sprint-qualifying" in w.get("scores", {})
            if has_prediction and not has_score:
                weekend_entry = w
                break

        if not weekend_entry:
            print("No unscored post-sprint-qualifying predictions found in history", file=sys.stderr)
            sys.exit(1)

        race_name = weekend_entry["race_name"]
        race_date = weekend_entry["date"]
        print(f"Scoring sprint weekend: {race_name} ({race_date})")

        # Find the F1DB slug for this race
        race_slug = find_race_slug(race_name)
        if not race_slug:
            print(f"Warning: Could not find slug for '{race_name}', trying all slugs...")
            all_slugs = get_all_race_slugs()
            for slug in all_slugs:
                if get_sprint_results(slug):
                    race_slug = slug
                    print(f"  Found slug with sprint results: {race_slug}")
                    break

        if not race_slug:
            print(f"Error: Could not find F1DB slug for {race_name}", file=sys.stderr)
            sys.exit(1)

        # Fetch sprint race result
        print(f"Fetching sprint race result for {race_slug}...")
        sprint_result = get_sprint_results(race_slug)
        if not sprint_result:
            print("Error: No sprint race result found — sprint may not have finished yet", file=sys.stderr)
            sys.exit(1)

        print(f"  Found {len(sprint_result)} classified drivers")

        # Get the prediction to score
        pred_data = weekend_entry["predictions"].get("post-sprint-qualifying")
        if not pred_data:
            print("Error: No post-sprint-qualifying prediction found for this weekend", file=sys.stderr)
            sys.exit(1)

        predictions = pred_data["predictions"]
        print(f"Scoring post-sprint-qualifying prediction ({len(predictions)} drivers)...")

        # Build actual results lookup: driver_name -> position
        # Also build a normalized lookup for fuzzy matching
        actual_results = {}
        actual_norm = {}
        for entry in sprint_result:
            driver = entry.get("name", "")
            pos = entry.get("position", "")
            if driver and pos:
                actual_results[driver] = int(pos) if isinstance(pos, str) and pos.isdigit() else pos
                # Normalized key: first + last name, lowercase, no suffixes/middle names
                parts = driver.lower().replace("jr.", "").replace("sr.", "").replace("jr", "").replace("sr", "").split()
                if len(parts) >= 2:
                    norm_key = f"{parts[0]} {parts[-1]}"
                    actual_norm[norm_key] = actual_results[driver]

        # Score the sprint prediction: 1pt per correct top-10 position + 5pt podium bonus, max 15
        position_score = 0
        position_details = []
        predicted_podium = []
        actual_podium = []

        # Get actual podium (top 3)
        for entry in sprint_result[:3]:
            driver = entry.get("name", "")
            if driver:
                actual_podium.append(driver)

        # Score each predicted position (1-10)
        for pred in predictions[:10]:
            pred_driver = pred.get("driver", "")
            pred_pos = pred.get("position", 0)

            # Try exact match first, then normalized (first+last name)
            actual_pos = actual_results.get(pred_driver)
            if actual_pos is None:
                norm_parts = pred_driver.lower().split()
                if len(norm_parts) >= 2:
                    norm_key = f"{norm_parts[0]} {norm_parts[-1]}"
                    actual_pos = actual_norm.get(norm_key)

            correct = (actual_pos == pred_pos)
            pts = 1 if correct else 0
            position_score += pts

            position_details.append({
                "position": pred_pos,
                "predicted": pred_driver,
                "actual": actual_pos if actual_pos is not None else "N/A",
                "correct": correct,
                "points": pts
            })

            if pred_pos <= 3:
                predicted_podium.append(pred_driver)

        # Podium score: 5 points if all 3 podium drivers predicted correctly (any order)
        podium_score = 0
        podium_correct = False
        if len(predicted_podium) == 3:
            # Normalize names for comparison (handle "Oscar Jack Piastri" vs "Oscar Piastri")
            def normalize_name(name):
                parts = name.lower().replace("jr.", "").replace("sr.", "").replace("jr", "").replace("sr", "").split()
                if len(parts) >= 2:
                    return f"{parts[0]} {parts[-1]}"
                return name.lower()

            pred_podium_norm = [normalize_name(p) for p in predicted_podium]
            actual_podium_norm = [normalize_name(a) for a in actual_podium]
            if all(ap in pred_podium_norm for ap in actual_podium_norm):
                podium_correct = True
                podium_score = 5

        total_sprint_score = position_score + podium_score

        scored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        score_data = {
            "sprint_score": total_sprint_score,
            "position_score": position_score,
            "podium_score": podium_score,
            "max_score": 15,
            "position_details": position_details,
            "podium_correct": podium_correct,
            "predicted_podium": predicted_podium,
            "actual_podium": actual_podium,
            "scored_at": scored_at
        }

        # Save score to weekend entry
        if "scores" not in weekend_entry:
            weekend_entry["scores"] = {}
        weekend_entry["scores"]["post-sprint-qualifying"] = score_data

        save_history(repo_path, history)
        print(f"\nSprint Score: {total_sprint_score}/15 (Positions: {position_score}/10, Podium: {podium_score}/5)")

        # Generate sprint score detail page
        short_name = _get_short_name(weekend_entry.get("circuit_id", ""), weekend_entry["race_name"])
        sprint_score_html = generate_sprint_score_html({
            "score": score_data,
            "race_name": weekend_entry["race_name"],
            "short_name": short_name,
            "date": race_date,
            "prediction_type": "post-sprint-qualifying",
            "scored_at": scored_at,
            "prediction_generated": pred_data.get("generated_at", "Unknown")
        })
        sprint_score_path = Path(repo_path) / "weekends" / f"{race_date}_score_sprint.html"
        with open(sprint_score_path, 'w') as f:
            f.write(sprint_score_html)
        print(f"Sprint score detail page: {sprint_score_path}")

        # Regenerate weekend page (now includes sprint score card)
        weekend_html = generate_weekend_html(weekend_entry)
        weekend_path = Path(repo_path) / "weekends" / f"{race_date}.html"
        with open(weekend_path, 'w') as f:
            f.write(weekend_html)

        # Regenerate index
        index_html = generate_index_html(history["weekends"])
        index_path = Path(repo_path) / "index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)

        print("Website regenerated with sprint score card")

        # Publish if requested
        if args.publish:
            print("\nPublishing to GitHub...")
            publish_to_github(repo_path, race_name)

        print("\nDone!")
        sys.exit(0)

    # Determine prediction type
    if args.detect:
        prediction_type = detect_prediction_type()
        print(f"Detected prediction type: {prediction_type}")
        sys.exit(0)
    elif args.type:
        prediction_type = args.type
    else:
        print("Error: Specify --type or --detect", file=sys.stderr)
        sys.exit(1)
    
    repo_path = args.repo
    
    # Get schedule info
    schedule = get_schedule()
    race_name = "Unknown"
    race_date = datetime.now().strftime("%Y-%m-%d")
    race_slug = None
    schedule_circuit_id = ""
    
    if schedule:
        for cat in schedule:
            if cat["category"] == "f1":
                race = cat.get("next_race", {})
                race_name = race.get("name", "Unknown")
                schedule_circuit_id = race.get("circuitId", "")
                for s in race.get("sessions", []):
                    dt = datetime.fromisoformat(s["datetime"])
                    race_date = dt.strftime("%Y-%m-%d")
                break
    
    # Find the F1DB slug for this race
    race_slug = find_race_slug(race_name)
    
    # Get race info from F1DB (circuit type, etc.)
    race_info = {}
    if race_slug:
        race_info = get_race_info(race_slug)
    
    print(f"Race: {race_name} ({race_date})")
    print(f"F1DB slug: {race_slug}")
    print(f"Prediction type: {prediction_type}")
    
    # Fetch data
    print("\nFetching standings...")
    standings = fetch_standings()
    print(f"  Found {len(standings)} drivers")
    
    print("Fetching completed race results...")
    races = fetch_race_results()
    print(f"  Found {len(races)} completed races")
    
    print("Fetching news...")
    news = fetch_news()
    print(f"  Found {len(news)} headlines")
    
    # Build session data for current weekend
    session_data = None
    sprint_weekend = race_slug and is_sprint_race(race_slug)
    
    if prediction_type in ["pre-sprint-qualifying", "post-sprint-qualifying", "pre-qualifying", "post-qualifying"] and race_slug:
        session_data = {
            "type": prediction_type,
            "race": race_name,
            "date": race_date,
            "sprint_weekend": sprint_weekend
        }
        
        if sprint_weekend:
            # Sprint weekend: only FP1 exists
            fp1_results = get_session_results(race_slug, "practice1")
            if fp1_results:
                session_data["practice1"] = fp1_results
            
            # Sprint qualifying data
            if prediction_type in ["post-sprint-qualifying", "pre-qualifying", "post-qualifying"]:
                sq_results = get_session_results(race_slug, "sprint_qualifying")
                if sq_results:
                    session_data["sprint_qualifying"] = sq_results
            
            # Sprint race data
            if prediction_type in ["pre-qualifying", "post-qualifying"]:
                sr_results = get_session_results(race_slug, "sprint_race")
                if sr_results:
                    session_data["sprint_race"] = sr_results
            
            # Main qualifying data
            if prediction_type == "post-qualifying":
                qual_results = get_session_results(race_slug, "qualifying")
                if qual_results:
                    session_data["qualifying"] = qual_results
        else:
            # Normal weekend: P1, P2, P3
            for practice in ["practice1", "practice2", "practice3"]:
                results = get_session_results(race_slug, practice)
                if results:
                    session_data[practice] = results
            
            # Fetch qualifying data for post-qualifying
            if prediction_type == "post-qualifying":
                qual_results = get_session_results(race_slug, "qualifying")
                if qual_results:
                    session_data["qualifying"] = qual_results
    
    # Fetch historical circuit performance
    historical_data = None
    if race_info:
        grand_prix_id = race_info.get("grandPrixId")
        if grand_prix_id:
            historical_data = get_historical_circuit_data(grand_prix_id, years=3)

    # Fetch official FIA starting grid (with penalties) for post-qualifying predictions
    fia_grid = None
    if prediction_type == "post-qualifying" and race_info:
        race_name = race_info.get("officialName", race_info.get("name", ""))
        if race_name:
            print("\nFetching official FIA starting grid (with penalties)...")
            fia_grid = fetch_fia_starting_grid(race_name)
            if fia_grid:
                grid_count = len(fia_grid.get('grid', []))
                penalty_count = len(fia_grid.get('penalties', []))
                print(f"  Fetched grid: {grid_count} drivers, {penalty_count} penalty notes")
            else:
                print("  Warning: Could not fetch FIA starting grid", file=sys.stderr)

    # Build prompt
    prompt = build_prediction_prompt(prediction_type, standings, races, news, session_data, race_info, historical_data, fia_grid)
    
    # Call LLM
    print("\nGenerating prediction (this may take a moment)...")
    llm_output = call_llm(prompt)
    
    if not llm_output:
        print("Error: LLM returned no output", file=sys.stderr)
        sys.exit(1)
    
    # Parse prediction
    predictions = parse_prediction(llm_output)
    if not predictions:
        print("Error: Could not parse prediction from LLM output", file=sys.stderr)
        print(f"LLM output: {llm_output[:500]}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Generated {len(predictions)} predictions")
    
    # Save to history
    history = load_history(repo_path)

    # Find or create weekend entry — match by date only (race names change between
    # schedule sources, e.g. "Barcelona-Catalunya" vs full F1DB official name)
    weekend_entry = None
    for w in history["weekends"]:
        if w["date"] == race_date:
            weekend_entry = w
            break

    if not weekend_entry:
        weekend_entry = {
            "date": race_date,
            "race_name": race_name,
            "circuit_id": schedule_circuit_id or race_info.get("circuitId", ""),
            "predictions": {},
            "standings": standings[:10] if standings else []
        }
        history["weekends"].insert(0, weekend_entry)
    else:
        # Update existing entry with latest data from schedule
        weekend_entry["race_name"] = race_name
        if not weekend_entry.get("circuit_id"):
            weekend_entry["circuit_id"] = schedule_circuit_id or race_info.get("circuitId", "")

    # Ensure circuit_id is set (may be missing from old entries)
    if not weekend_entry.get("circuit_id") and race_info.get("circuitId"):
        weekend_entry["circuit_id"] = race_info["circuitId"]
    
    # Track prediction types for this weekend
    if "prediction_types" not in weekend_entry:
        weekend_entry["prediction_types"] = []
    if prediction_type not in weekend_entry["prediction_types"]:
        weekend_entry["prediction_types"].append(prediction_type)
    
    # Store prediction
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    weekend_entry["predictions"][prediction_type] = {
        "type": prediction_type,
        "date": race_date,
        "generated_at": generated_at,
        "predictions": predictions,
        "race_name": race_name
    }
    
    save_history(repo_path, history)
    print(f"\nSaved prediction for {race_name}")
    
    # Generate website
    print("\nGenerating website...")
    
    # Index page
    index_html = generate_index_html(history["weekends"])
    index_path = Path(repo_path) / "index.html"
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    # Weekend pages
    weekends_dir = Path(repo_path) / "weekends"
    weekends_dir.mkdir(parents=True, exist_ok=True)
    
    for weekend in history["weekends"]:
        # Weekend page
        weekend_html = generate_weekend_html(weekend)
        weekend_path = weekends_dir / f"{weekend['date']}.html"
        with open(weekend_path, 'w') as f:
            f.write(weekend_html)
        
        # Prediction detail pages
        short_name = _get_short_name(weekend.get("circuit_id", ""), weekend["race_name"])
        for ptype, pdata in weekend["predictions"].items():
            pred_html = generate_prediction_html({
                "type": ptype,
                "predictions": pdata["predictions"],
                "race_name": weekend["race_name"],
                "short_name": short_name,
                "generated_at": pdata["generated_at"],
                "date": weekend["date"],
                "reasoning": ""
            })
            pred_path = weekends_dir / f"{weekend['date']}_{ptype}.html"
            with open(pred_path, 'w') as f:
                f.write(pred_html)
    
    print("Website generated")
    
    # Publish to GitHub
    if args.publish:
        print("\nPublishing to GitHub...")
        publish_to_github(repo_path, race_name)
    
    print("\nDone!")

def publish_to_github(repo_path, race_name):
    """Commit and push changes to GitHub."""
    os.chdir(repo_path)
    
    subprocess.run(["git", "add", "."], check=True)
    
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True
    )
    
    if result.returncode == 0:
        print("  No changes to commit")
        return
    
    msg = f"Predictions: {race_name} - {datetime.now().strftime('%Y-%m-%d')}"
    subprocess.run(["git", "commit", "-m", msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"  Published: {msg}")

if __name__ == "__main__":
    main()
