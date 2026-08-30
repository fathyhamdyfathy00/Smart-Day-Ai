#  SmartDay AI Agent v9.0 — Adaptive AI Scheduling System

An **adaptive, bilingual (Arabic/English) daily scheduling system** built entirely in Python, running as a local web app via **FastAPI**. The core idea: instead of building your schedule manually, the system takes all your tasks, proposes **several different candidate schedules**, and learns from your choices over time using a simple **Reinforcement Learning** algorithm — so recommendations get more accurate the more you use it.

---

##  Table of Contents

- [Key Features](#-key-features)
- [Tools & Stack Used](#-tools--stack-used)
- [Feature Guide](#-feature-guide)
- [Internal API Endpoints](#-internal-api-endpoints)
- [Installation](#️-installation)
- [Running the App](#️-running-the-app)
- [Files the App Generates](#-files-the-app-generates)
- [Supported Chatbot Commands](#-supported-chatbot-commands)
- [Important Notes](#️-important-notes)

---

##  Key Features

-  **Real recommendation system** using an **Epsilon-Greedy (80/20)** algorithm — 80% exploitation of the highest-scoring predicted schedule, and 20% exploration of alternative schedules so the system keeps learning.
-  Generates **4 to 6 genuinely diverse schedules** (not near-duplicates) using semi-random permutations plus deliberate bias.
-  An advanced **Utility Scoring** function that factors in: task priority, difficulty, deadline urgency, proximity to bedtime, and the user's energy level at a given time.
-  **Gradual reinforcement learning**: weights are updated incrementally (`learning_rate * reward`) with a small decay so the system never gets "stuck" on an old preference.
-  A **user/productivity profile** built automatically from behavior (task completion, postponement, productive hours) that influences future recommendations.
-  A **bilingual (Arabic/English) chatbot assistant** with natural-language understanding built on manual Regex/NLP — it understands commands like "I finished studying" or "postpone task to 5pm".
-  A **full audio system**: a "focus" sound when a task starts, an "alert" sound when its time is exceeded, plus text-to-speech for chatbot notifications.
-  A **fully interactive dashboard** in the browser: a visual timeline with connector arrows between tasks, task cards, live notifications, and a light/dark theme.
-  **Data stored in an Excel file** (`.xlsx`) instead of a database — easy to open and edit manually if you want.
-  **Built-in web search** inside the chatbot via the free DuckDuckGo API (no key required).
-  A **background scheduler** running every 5 seconds that watches tasks, sends notifications, and plays sounds at exactly the right time.
-  Automatically opens in **Google Chrome** (or the default browser) when the script starts.

---

##  Tools & Stack Used

### Backend (Python)
| Tool | Purpose |
|---|---|
| **FastAPI** | Core framework for the server and all API endpoints |
| **Uvicorn** | ASGI server used to run the FastAPI app |
| **openpyxl** | Reads/writes task data to an Excel file (`smartday_tasks.xlsx`) instead of a database |
| **pyttsx3** | Text-to-speech for the assistant's voice notifications |
| **pygame** (`pygame.mixer`) | Plays audio files (focus-start sound / task-end alert sound) |
| **requests** | Sends HTTP requests to the external search service (DuckDuckGo) |
| **numpy** | Helper numerical operations (optional — the app still runs without it) |
| **wave / struct** (Python standard library) | Automatically generates default WAV sound files (beeps) if none exist |
| **threading / ThreadPoolExecutor** | Runs the background scheduler loop and audio tasks in separate threads without blocking the server |
| **asyncio** | Handles some asynchronous operations |
| **re (Regex)** | All the chatbot's natural-language understanding — extracting time, duration, priority, and difficulty from user text |
| **json** | Stores/loads: RL weights (`rl_weights.json`), productivity profile, user preferences, settings, and schedule history |
| **datetime / timedelta** | All time and date calculations |
| **socket** | Detects the machine's local IP to display in the console (so you can access the app from other devices on the same network) |
| **webbrowser** | Automatically opens the browser (tries Google Chrome first) when the script starts |

### Frontend (embedded inside the same Python file, as HTML/CSS/JS)
| Tool | Purpose |
|---|---|
| **Vanilla JavaScript** | All UI logic (Dashboard, Timeline, Chat, Modals, Notifications) is hand-written with no framework (no React, no Vue) |
| **Vanilla CSS** (custom design system) | Full styling with CSS variables to support light/dark theme |
| **Google Fonts** (`Tajawal` + `JetBrains Mono`) | Arabic and English fonts used in the UI (loaded via CDN from `fonts.googleapis.com`) |
| **Fetch API** | All communication between the frontend and server (`/api/...`) goes through plain `fetch()`, no external library |

> Note: The entire frontend (HTML + CSS + JS) is written **inline** inside `SmartDay.py` as one long string (`HTML = """..."""`) served directly by FastAPI — there are no separate `.html` or `.js` files.

---

## 📖 Feature Guide

### 1) Smart Recommendation System (`IntelligentScheduler` + `RecommendationSystem`)
- The generator (`SmartScheduleGenerator`) produces several different task orderings using multiple strategies: by priority, by difficulty, by duration (short/long), by energy level, by deadline, and biased semi-random permutations.
- Each schedule is scored by `UtilityEvaluator`, which computes a score based on adjustable weights (`DEFAULT_WEIGHTS`).
- `RecommendationSystem` picks the suggested schedule using **Epsilon-Greedy**: usually (80%) it recommends the highest score, and sometimes (20%) it recommends a different one at random to "explore" and discover new user preferences.

### 2) Reinforcement Learning (`ReinforcementLearning`)
- Every time you pick a schedule from the suggestions, the system computes a "reward": +1 if you picked the top recommendation (meaning its suggestion was correct), or a lower/negative value otherwise.
- Weights are updated with: `new_weight = old_weight + learning_rate * reward`, plus a small decay (1%) that gradually pulls weights back toward their defaults so the system never stays "frozen" on an old decision.
- All of this is persisted in `rl_weights.json` and `schedule_history.json` so it survives restarts.

### 3) Productivity Profile (`ProductivityProfile`) and Talent Profile (`TalentProfile`)
- Tracks user behavior: does the user finish tasks on time? Do they postpone a lot? Which hours of the day are they most "productive" in?
- `TalentProfile` classifies tasks by type (e.g. "coding", "reading", etc.) and computes a "difficulty factor" and "completion speed" per type, using an **EMA (Exponential Moving Average)** to update values smoothly.

### 4) Smart Chatbot (`agent_process`)
- A bilingual (Arabic/English) chatbot that understands natural commands via advanced Regex, including:
  - Adding/deleting/editing tasks.
  - Extracting a task's time, duration, priority, and difficulty directly from the sentence.
  - Handling "partial completion" commands (if you finished only part of a task, not all of it).
  - Quick web lookups (via DuckDuckGo).
- The UI simulates a "typing" indicator and splits long replies across multiple messages, like a real conversation.

### 5) Scheduling and Notifications (`scheduler_loop`)
- A background thread always running (every 5 seconds) that watches for:
  - A task's start time being reached → plays "focus music" + sends a notification.
  - A task's end time being exceeded → plays an "alert sound" + sends a notification.
  - Bedtime approaching (25–35 minutes left) while tasks are still pending → a special warning.
  - Generating "smart insights" every two hours.
- `TaskAudioController` ensures every sound plays **exactly once** per task, even if it gets rescheduled.

### 6) Dashboard and Timeline
- A full visual layout of tasks as cards and a timeline with visual connector arrows linking tasks in order.
- A live notifications panel.
- Light/dark theme toggle.
- Keyboard shortcuts (Ctrl+1 through Ctrl+5) to navigate between panels.

### 7) Storage (no database)
- All tasks are stored in an **Excel** file (`smartday_tasks.xlsx`) via `openpyxl` — you can open and edit it manually anytime.
- Settings, user preferences, and RL weights are stored in simple, separate **JSON** files.

---

##  Internal API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/tasks` | Fetch all tasks |
| POST | `/api/tasks` | Add a new task |
| POST | `/api/tasks/batch` | Add multiple tasks at once |
| POST | `/api/tasks/{id}/complete` | Mark a task as completed |
| POST | `/api/tasks/{id}/complete-with-remaining` | Partially complete a task, specifying remaining time |
| POST | `/api/tasks/{id}/postpone/tomorrow` | Postpone a task to the next day |
| POST | `/api/tasks/{id}/postpone/time` | Postpone a task to a specific time |
| POST | `/api/tasks/{id}/update-time` | Update a task's time |
| POST | `/api/tasks/swap` | Swap the order of two tasks |
| GET | `/api/schedule/find-slots` | Find available free time slots |
| POST | `/api/schedule/optimize` | Automatically generate and optimize the schedule |
| GET | `/api/schedule/recommendations` | Fetch current smart recommendations |
| POST | `/api/schedule/select` | Select one of the recommendations (feeds the RL system) |
| GET/POST | `/api/settings` | Read/save app settings |
| POST | `/api/chat` | Send a message to the smart assistant |
| GET | `/api/history` | Previous scheduling history |
| GET | `/api/notifications` | Fetch new notifications |
| GET | `/api/insights` | Smart insights about user behavior |
| GET | `/api/talent` | Talent/performance profile |
| GET | `/api/status` | General server status |

---

##  Installation

```bash
git clone https://github.com/fathyhamdyfathy00/Smart-Day-Ai.git

cd Smart-Day-Ai

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Suggested `requirements.txt`
```
fastapi
uvicorn
openpyxl
pyttsx3
pygame
requests
numpy
```

> `openpyxl`, `pyttsx3`, `pygame`, `requests`, and `numpy` are all **effectively optional** — the app checks for their availability at startup (`try/except`) and runs with reduced features if any are missing (e.g. no sound if `pygame` isn't installed). For the full feature set, it's best to install all of them.

## Running the App

```bash
python SmartDay.py
```

On startup:
1. The server runs on `http://localhost:8000`.
2. The browser opens automatically (Google Chrome if found, otherwise the default browser).
3. You can also access it from another device on the same network using the local IP shown in the console.

##  Files the App Generates

These are created automatically the first time you run the script:

```
smartday_tasks.xlsx          # Task data (Excel)
rl_weights.json              # Reinforcement learning weights
productivity_profile.json    # User productivity profile
schedule_history.json        # History of past schedules and choices
app_settings.json            # General app settings
user_preferences.json        # Scheduling strategy preferences
mixkit-software-interface-*.wav   # Default sound files (auto-created if missing)
```

##  Supported Chatbot Commands

| Arabic | English |
|---|---|
| "خلصت [task]" | "complete [task]" |
| "ملحقتش [task]" | "not finished [task]" |
| "اجل [task] لـ [time]" | "postpone [task] to [time]" |
| "جدول ذكي" | "smart schedule" |
| "بدل [task1] مع [task2]" | "swap [task1] with [task2]" |

##  Important Notes

- The entire project lives in a single Python file (`SmartDay.py`) — backend and frontend (HTML/CSS/JS) are combined together.
- Audio (`pygame`) and `pyttsx3` run entirely locally on your machine — no internet connection is required for them.
- The `web_search` feature is the project's only internet-dependent call, and it's optional and time-limited (4 seconds) so the app doesn't hang on a slow or dropped connection.
- All user data is stored locally on the user's machine (Excel + JSON files) — no data is sent to any external servers.

