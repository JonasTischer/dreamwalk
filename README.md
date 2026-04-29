# Dreamwalk

**Voice-driven exploration of generated worlds.**

Talk to a voice agent. It paints scenes for you in real time, and you walk through them with your voice — "show me Sydney in 2050," then "walk closer to that bridge," then "make it sunset," then "show me the actual Tesla Cybertruck instead." The image is always the last thing you said, and the agent describes what it sees as it goes.

It's a single FastHTML app glued together with the OpenAI Realtime API, fal's flux-2 image models, and Brave Search.

## Demo flow

```text
[Start voice]   →   Hi, what are we building today?
> show me Sydney in 2050
[image: Sydney, 2050]
                →   here's downtown — there's a maglev rail above the harbour
> walk to the bridge
[image: Sydney harbour bridge, closer]
                →   we're on the bridge approach now
> make it night-time
[image: same scene, lit up]
                →   neon reflections in the water
> actually, show me the real Tesla Cybertruck
[image: real Cybertruck photo from web]
                →   here's the production unit
```

The agent decides which tool to use based on what you say. You don't pick from a menu.

## What's distinctive

Most "voice + image" demos are open-loop: you describe a scene, you get a picture, you start over. Dreamwalk closes the loop:

- **The agent sees what it just made.** Every generated image is downsampled and pushed back into the Realtime conversation as visual context. So when you say "go on top of *that* building," the agent knows which one.
- **Continuity across turns.** "Edit" mode passes the previous image as a reference to the next fal generation, so spatial moves feel continuous instead of teleporting to a brand-new world.
- **Real and imagined coexist.** `find_image` pulls actual photos from the web (Brave) and treats them as scenes you can dreamwalk *from* — say "show me UTS Startup Sydney" and then "now imagine it in 2080." The boundary between real-world and generated is one tool call.
- **Latency hiding.** The agent is instructed to start speaking *while* the image renders ("sure, taking you up there..."), so the 3-second wait is filled with audio and feels closer to instantaneous.

## Architecture

```text
┌──────────┐  WebRTC audio + data    ┌──────────────────────┐
│ browser  │ ◄─────────────────────► │  OpenAI Realtime     │
│          │                         │  (gpt-realtime)      │
│ <img>    │                         │  + 4 tools           │
│ orb UI   │                         └──────────┬───────────┘
└────┬─────┘                                    │ tool calls
     │ POST /api/{command,find-image,...}       ▼
     ▼                                  ┌──────────────────┐
┌──────────────┐  ┌──────────────────┐  │  routes them via │
│ FastHTML app │  │  fal flux-2/flash│  │  data channel    │
│              │  │  + edit          │  └──────────────────┘
│  /api/...    │──┤  fal flux/edit   │
│  WorldState  │  │  gpt-image-2     │
│  per session │  │  Brave Search    │
└──────────────┘  └──────────────────┘
```

Single Python file (`app.py`), single inline JS bundle, single inline CSS. No build step.

## The four tools the agent has

| Tool | What it does |
|---|---|
| **`update_scene`** | Generate a brand-new world or edit the current one. Picks `fal-ai/flux-2/flash` for fast photo-real (~3s) or `gpt-image-2` for high-quality with readable text (~15-30s, used only for diagrams/posters). |
| **`find_image`** | Brave image search → display the top real photo. For "show me X" where X is a real-world thing. |
| **`recall_image`** | Load a specific past frame back into the agent's vision context. Useful for "go back to the rooftop view." |
| **`search_web`** | Brave web search for grounding facts before generating ("today's weather in Tokyo" → grounded scene). |

The agent decides which to call from natural language. Instructions in `REALTIME_INSTRUCTIONS` (`app.py`) tell it when to prefer each.

## Run it

```bash
cd /path/to/dreamwalk
printf 'FAL_KEY=your_fal_key_here\nOPENAI_API_KEY=your_openai_key_here\nBRAVE_API_KEY=your_brave_key_here\n' > .env
uv run python app.py
```

Open <http://localhost:5001>, click **Start voice**, allow microphone, and talk.

- `FAL_KEY` — required for fast image generation (fal flux-2)
- `OPENAI_API_KEY` — required for voice and routing/vision
- `BRAVE_API_KEY` — required for `find_image` and `search_web`

Without `FAL_KEY` you get an SVG placeholder. Without `OPENAI_API_KEY` voice won't connect but typed commands still route locally.

## Useful environment variables

```bash
# Image generation
export FAL_IMAGE_MODEL="fal-ai/flux-2/flash"
export FAL_IMAGE_EDIT_MODEL="fal-ai/flux-2/flash/edit"
export FAL_IMAGE_SIZE="1024x576"
export FAL_IMAGE_OUTPUT_FORMAT="webp"
export FAL_IMAGE_GUIDANCE_SCALE="2.5"

# High-quality fallback
export OPENAI_IMAGE_MODEL="gpt-image-2"
export OPENAI_IMAGE_SIZE="1536x1024"
export OPENAI_IMAGE_QUALITY="medium"

# Routing / vision
export OPENAI_TEXT_MODEL="gpt-4.1-mini"

# Realtime voice
export OPENAI_REALTIME_MODEL="gpt-realtime"
export OPENAI_REALTIME_VOICE="marin"
```

## How it works under the hood

### Voice path

1. Browser opens a WebRTC peer connection.
2. `POST /api/realtime/session` forwards the SDP offer to OpenAI's `/v1/realtime/calls` along with the system instructions and tool schemas.
3. OpenAI returns the SDP answer. Browser sets the remote description and now has bidirectional audio + a data channel for events.
4. Tool calls arrive as `response.done` events with `function_call` items. The browser POSTs to the matching `/api/...` endpoint, gets a result, and sends `function_call_output` + `response.create` back over the data channel.

### Image generation path

```text
user voice "show me Sydney"
  → Realtime decides update_scene(mode="create", quality="fast", prompt="...")
  → browser POSTs /api/command
    → fal flux-2/flash returns a URL (~3s)
    → server records {image_id, url, prompt} on WorldState
    → returns image URL + manifest of recent images
  → browser shows image
  → browser POSTs /api/realtime-image to push downsampled image into agent's vision context
  → browser sends function_call_output + response.create to Realtime
  → agent speaks one short comment about the new image
```

Edit mode (`mode="edit"`) is identical except the previous image URL is passed to fal as a reference, so the new frame is a continuation, not a fresh world.

### State

Per-session `WorldState`:
- `last_image_url` — the most recent image (used as edit reference)
- `images: list[ImageRecord]` — history of {id, url, prompt, summary, created_at}, capped at 32 most recent
- `lock: asyncio.Lock` — per-session lock so concurrent tool calls don't race on `last_image_url`

Sessions are keyed by a browser-generated `session_id`. There's no auth and no persistence — sessions live in process memory.

### Concurrency

- All blocking I/O (fal calls, OpenAI calls, Brave) is wrapped in `asyncio.to_thread` so the event loop never stalls during a 10-second generation.
- The Realtime data channel allows only one active response at a time. The client tracks active response IDs and gates `response.create` sends behind in-flight responses to avoid `conversation_already_has_active_response` errors.

### Structured outputs

Both the route_text_command (typed-command router) and the agent's tool schemas use OpenAI's `response_format={"type": "json_schema", "strict": True}`. No regex JSON parsing, no markdown-fence fallbacks — the API rejects malformed output at the boundary.

## File layout

```text
app.py            # the entire app: FastHTML routes + inline JS + inline CSS
pyproject.toml    # uv-managed deps
README.md         # you are here
AGENTS.md         # (placeholder)
llms-ctx.txt      # FastHTML context for LLMs
```

`app.py` is one file because the project fits in one file — the cost of splitting it would be more than the cost of scrolling.

## What was built during the hackathon

Everything past the initial "voice + one tool that calls fal" skeleton was built in this session:

- **Multi-tool voice agent** (`update_scene`, `find_image`, `recall_image`, `search_web`) with a shared instruction set that teaches the model when to pick each.
- **Closed-loop vision context** — `/api/realtime-image` downsamples each frame to 640px JPEG and pushes it back into the agent's vision context as an `input_image`, so the agent literally sees what it just generated and can ground the next turn.
- **Two image backends** — fal flux-2/flash for the fast immersive path (~3s) and OpenAI gpt-image-2 for the rare hi-res/text-readable case. Selected automatically by the router based on what the user asked for.
- **Brave Search integration** for both real-photo lookups (`find_image`) and text grounding (`search_web`).
- **Latency-hiding pattern** — agent instructed to speak *during* image generation, so a 3-second fal call feels like a continuous conversation instead of a wait.
- **Concurrency hardening** — per-session asyncio lock + `to_thread` for blocking I/O + a response.create gate in JS that prevents Realtime's "active response" errors.
- **Structured outputs** for the routing decision and the agent's tool schemas — no JSON regex parsing.
- **Voice transcript / status split** — operation status (what's happening) and agent transcript (what's being said) live on separate UI rows so neither overwrites the other mid-action.

Several features were prototyped, evaluated, and **explicitly cut** before submission: real-time annotation overlays, region-specific edits with feathered PIL composites, an upscaler stage with a stale-result guard, and a YouTube livestream loader (yt-dlp + ffmpeg). They worked, but each added failure surface that wasn't paying for itself in a 5-minute demo. The git-less commit-history of this single file goes both ways — features were added and removed, and what's here now is the spine that's worth showing.

## Tech stack

- **[FastHTML](https://fastht.ml/)** — Python web framework, no build step
- **[OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)** — `gpt-realtime` for voice, `gpt-4.1-mini` for routing, `gpt-image-2` for hi-res images
- **[fal](https://fal.ai/)** — `flux-2/flash` and `flux-2/flash/edit` for fast photo-real generation
- **[Brave Search API](https://brave.com/search/api/)** — image and web search
- **[Pillow](https://pillow.readthedocs.io/)** — image downsampling for vision context
- **[uv](https://docs.astral.sh/uv/)** — dep management
- **WebRTC + DataChannel** — browser ↔ OpenAI audio + event transport, no library

## Known limitations

- **Single browser tab per session.** No multi-user state isolation beyond browser-generated session IDs.
- **No persistence.** Restarting the server forgets all worlds.
- **fal queue cold starts.** First image of the day can take 5-10s instead of ~3s.
- **Realtime model occasionally misroutes tools.** With four tools the decision surface isn't infinite, but `find_image` vs `update_scene` for "show me X" is genuinely ambiguous and the agent guesses wrong sometimes. The instruction set tries to disambiguate ("real-world named entity → find_image; imagine/show me a... → update_scene") but it's not perfect.
- **No auth.** Demo only.
