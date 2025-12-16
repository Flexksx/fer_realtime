# Emotion‑Aware LLM UI — Assessment Rubric (Grades Optional)

## What Students Build
- Emotion input: `happy | sad | angry | fear | surprise | disgust | neutral` (manual select is baseline; FER/webcam is bonus)
- Reactive behavior: on emotion change, call a backend API that prompts Ollama and returns a tailored response
- Output contract: LLM returns **strict JSON only** with keys exactly: `emotion`, `joke`, `tone`, `safety`
- UX: debounce rapid changes, ignore/cancel stale requests, show loading/streaming, handle errors gracefully
- Architecture: frontend → backend → Ollama (`/api/chat` or `/api/generate`); frontend does **not** call Ollama directly

## Rubric (use points or just checkmarks)

### 1) Core Functionality (optional points: ____ / 25)
- [ ] Emotion input works and uses the required emotion set
- [ ] On emotion change, app calls backend → Ollama and renders the response
- [ ] Stale-request handling (cancel/ignore outdated responses so UI matches current emotion)

### 2) Reactive UX Quality (optional points: ____ / 15)
- [ ] Debounce or rate-limit rapid emotion changes (e.g., 300–800ms)
- [ ] Clear loading/streaming indicator
- [ ] Graceful error state (message + retry path) without crashing

### 2b) Probability & Trigger Tuning (optional points: ____ / 10)
- [ ] Uses smoothed probabilities (e.g., moving average over last N frames) and explains the choice of N
- [ ] Tunes the trigger threshold to avoid spam and missed events, and justifies the chosen value
- [ ] Trigger metric implemented/understood as:
  - `delta_max = max_e | w(e) · ( p_t(e) − p_{t−1}(e) ) |`
  - where `p_t(e)` is the **smoothed** probability at time `t`, and `w(e)` comes from `emotion_weights`
- [ ] Explains that `neutral` is included in the max but typically set to `w(neutral)=0` by default (so it’s ignored unless students raise it)

### 3) Output Contract & Parsing (optional points: ____ / 20)
- [ ] Prompt enforces **JSON only** with keys exactly: `emotion`, `joke`, `tone`, `safety`
- [ ] Backend validates JSON schema and handles invalid output (reprompt / reject / fallback)
- [ ] Frontend handles parse/validation failures safely (no broken UI)

### 4) Backend Architecture (optional points: ____ / 15)
- [ ] Frontend never calls Ollama directly; backend proxies to Ollama
- [ ] Clean API design (e.g., `POST /generate` with `{ emotion }` → validated JSON response)

### 5) Prompt & Safety Rules (optional points: ____ / 15)
- [ ] Constraints applied (max ~2 sentences; classroom-appropriate; no insults; no self-harm; no politics)
- [ ] Backend enforces or corrects unsafe/invalid responses
- [ ] `safety` field is meaningful and consistent with the content

### 6) Documentation & Demo (optional points: ____ / 10)
- [ ] README includes setup/run steps for frontend, backend, and Ollama model
- [ ] Demo evidence (short video/gif/screenshots) shows emotion changes + loading + an error case

## Bonus (optional points: ____ / +10)
- [ ] FER/webcam integration provides the emotion signal and still meets debounce + stale-request + safety/JSON contract

## Notes / Feedback
- Strengths:
- Issues found:
- Suggested improvements:
