import asyncio
import base64
import io
import json
import logging
import os
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field

import fal_client
import httpx
from dotenv import load_dotenv
from fasthtml.common import *
from openai import OpenAI
from PIL import Image
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("dreamwalk")

FAL_IMAGE_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux-2/flash")
FAL_IMAGE_EDIT_MODEL = os.getenv("FAL_IMAGE_EDIT_MODEL", "fal-ai/flux-2/flash/edit")
FAL_IMAGE_SIZE = os.getenv("FAL_IMAGE_SIZE", "1024x576")
FAL_IMAGE_OUTPUT_FORMAT = os.getenv("FAL_IMAGE_OUTPUT_FORMAT", "webp")
FAL_IMAGE_GUIDANCE_SCALE = float(os.getenv("FAL_IMAGE_GUIDANCE_SCALE", "2.5"))
FAL_ENABLE_PROMPT_EXPANSION = os.getenv("FAL_ENABLE_PROMPT_EXPANSION", "false").lower() == "true"
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1536x1024")
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "marin")


@dataclass
class ImageRecord:
    id: str
    url: str
    prompt: str
    summary: str
    created_at: float


@dataclass
class WorldState:
    last_image_url: str = ""
    images: list[ImageRecord] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)


def summarize_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    first = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    text = first or cleaned
    if len(text) <= 140:
        return text
    cut = text[:140].rsplit(" ", 1)[0]
    return (cut or text[:140]) + "..."


MAX_IMAGES_PER_SESSION = 32


def add_image_record(state: WorldState, url: str, prompt: str) -> ImageRecord:
    record = ImageRecord(
        id=uuid.uuid4().hex[:8],
        url=url,
        prompt=prompt,
        summary=summarize_prompt(prompt),
        created_at=time.time(),
    )
    state.images.append(record)
    if len(state.images) > MAX_IMAGES_PER_SESSION:
        del state.images[: len(state.images) - MAX_IMAGES_PER_SESSION]
    state.last_image_url = url
    return record


def recent_image_manifest(state: WorldState, limit: int = 8) -> list[dict]:
    now = time.time()
    return [
        {
            "id": r.id,
            "summary": r.summary,
            "seconds_ago": int(now - r.created_at),
        }
        for r in state.images[-limit:]
    ]


SESSIONS: dict[str, WorldState] = {}


def get_state(session_id: str) -> WorldState:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = WorldState()
    return SESSIONS[session_id]


def fal_image_size():
    if re.fullmatch(r"\d+x\d+", FAL_IMAGE_SIZE):
        width, height = FAL_IMAGE_SIZE.split("x", 1)
        return {"width": int(width), "height": int(height)}
    return FAL_IMAGE_SIZE


def placeholder_image(prompt: str, missing: str = "API key") -> str:
    short = prompt[:420].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    svg = textwrap.dedent(
        f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1536 1024">
          <rect width="1536" height="1024" fill="#050505"/>
          <rect x="72" y="72" width="1392" height="880" fill="#111" stroke="#333" stroke-width="2"/>
          <text x="120" y="150" fill="#f2f2f2" font-family="Arial" font-size="44">{missing} required</text>
          <foreignObject x="120" y="210" width="1296" height="650">
            <div xmlns="http://www.w3.org/1999/xhtml" style="color:#bbb;font-family:Arial;font-size:28px;line-height:1.35">
              {short}
            </div>
          </foreignObject>
        </svg>
        """
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def generate_fal_image(prompt: str, reference_image_url: str = "") -> str:
    if not os.getenv("FAL_KEY"):
        logger.warning("FAL_KEY missing; returning placeholder")
        return placeholder_image(prompt, missing="FAL_KEY")

    use_edit = bool(reference_image_url and reference_image_url.startswith("http"))
    model = FAL_IMAGE_EDIT_MODEL if use_edit else FAL_IMAGE_MODEL

    arguments = {
        "prompt": prompt,
        "guidance_scale": FAL_IMAGE_GUIDANCE_SCALE,
        "image_size": fal_image_size(),
        "num_images": 1,
        "enable_prompt_expansion": FAL_ENABLE_PROMPT_EXPANSION,
        "enable_safety_checker": False,
        "output_format": FAL_IMAGE_OUTPUT_FORMAT,
    }
    if use_edit:
        arguments["image_urls"] = [reference_image_url]

    started = time.perf_counter()
    logger.info(
        "fal %s model=%s prompt_chars=%s reference=%s",
        "edit" if use_edit else "create",
        model,
        len(prompt),
        reference_image_url or "none",
    )
    result = fal_client.subscribe(model, arguments=arguments)
    elapsed = time.perf_counter() - started
    images = result.get("images") or []
    if not images or not images[0].get("url"):
        raise RuntimeError(f"fal returned no image: {result}")
    url = images[0]["url"]
    logger.info("fal done in %.2fs image=%s", elapsed, url)
    return url


def generate_openai_image(prompt: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY missing; returning placeholder")
        return placeholder_image(prompt, missing="OPENAI_API_KEY")

    started = time.perf_counter()
    logger.info(
        "openai create model=%s size=%s quality=%s prompt_chars=%s",
        OPENAI_IMAGE_MODEL, OPENAI_IMAGE_SIZE, OPENAI_IMAGE_QUALITY, len(prompt),
    )
    client = OpenAI()
    result = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=OPENAI_IMAGE_SIZE,
        quality=OPENAI_IMAGE_QUALITY,
        n=1,
    )
    elapsed = time.perf_counter() - started
    image_b64 = result.data[0].b64_json or ""

    if os.getenv("FAL_KEY"):
        try:
            png_bytes = base64.b64decode(image_b64)
            url = fal_client.upload(png_bytes, content_type="image/png", file_name="openai.png")
            logger.info("openai done in %.2fs uploaded=%s", elapsed, url)
            return url
        except Exception:
            logger.exception("failed to upload openai image to fal; falling back to data URL")

    logger.info("openai done in %.2fs base64_chars=%s (no fal upload)", elapsed, len(image_b64))
    return f"data:image/png;base64,{image_b64}"


def generate_image(prompt: str, quality: str, reference_image_url: str = "") -> str:
    """Pick the right image backend based on quality.

    quality="fast" → fal flux-2/flash (uses reference for edit if available)
    quality="high" → openai gpt-image (always from scratch, supports text/diagrams)
    """
    if quality == "high":
        return generate_openai_image(prompt)
    return generate_fal_image(prompt, reference_image_url=reference_image_url)


def realtime_image_data_uri(image_url: str) -> str:
    """Convert a session image to a compact JPEG data URI for Realtime vision input."""
    if image_url.startswith("data:image/"):
        match = re.match(r"data:image/[^;]+;base64,(.+)$", image_url, flags=re.I | re.S)
        if not match:
            raise RuntimeError("image data URI must be base64 encoded")
        image_bytes = base64.b64decode(match.group(1), validate=True)
    elif image_url.startswith(("http://", "https://")):
        response = httpx.get(image_url, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
        image_bytes = response.content
    else:
        raise RuntimeError("image_url must be http(s) or an image data URI")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((640, 640), Image.LANCZOS)

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=68, optimize=True)
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def search_images(query: str, count: int = 5) -> list[dict]:
    """Brave Image Search. Returns up to `count` results as [{url, title, source, thumb}]."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY is not set")
    started = time.perf_counter()
    response = httpx.get(
        "https://api.search.brave.com/res/v1/images/search",
        params={"q": query, "count": count, "safesearch": "strict"},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for r in data.get("results") or []:
        props = r.get("properties") or {}
        url = props.get("url") or (r.get("thumbnail") or {}).get("src")
        if not url:
            continue
        results.append({
            "url": url,
            "title": (r.get("title") or "").strip(),
            "source": (r.get("source") or "").strip(),
            "thumb": (r.get("thumbnail") or {}).get("src", ""),
        })
    logger.info(
        "brave images in %.2fs query=%r results=%d",
        time.perf_counter() - started, query, len(results),
    )
    return results[:count]


def search_web(query: str, count: int = 5) -> list[dict]:
    """Brave Search. Returns up to `count` results as [{title, url, snippet}]."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY is not set")
    started = time.perf_counter()
    response = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": count},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for r in (data.get("web") or {}).get("results") or []:
        results.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "snippet": (r.get("description") or "").strip(),
        })
    logger.info(
        "brave search in %.2fs query=%r results=%d",
        time.perf_counter() - started, query, len(results),
    )
    return results[:count]


ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["create", "edit"]},
        "quality": {"type": "string", "enum": ["fast", "high"]},
        "prompt": {"type": "string"},
    },
    "required": ["mode", "quality", "prompt"],
    "additionalProperties": False,
}


def route_text_command(state: WorldState, text: str) -> dict:
    """For typed commands (no voice). One LLM call picks create vs edit + writes the prompt.
    The current image (if any) is attached so the LLM can see what the user is referring to."""
    has_image = bool(state.last_image_url and state.last_image_url.startswith("http"))
    if not os.getenv("OPENAI_API_KEY"):
        return {"mode": "edit" if has_image else "create", "quality": "fast", "prompt": text}

    client = OpenAI()
    user_content: list = [{"type": "input_text", "text": text}]
    if has_image:
        user_content.append(
            {"type": "input_image", "image_url": state.last_image_url, "detail": "low"}
        )

    response = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You convert a user's natural-language request into a structured decision.\n"
                    "Pick `mode`:\n"
                    "- 'create': brand-new world, e.g. 'show me Sydney at sunrise'.\n"
                    "- 'edit': continue from the current frame — movement, lighting, style, add/remove things.\n"
                    "Pick `quality`:\n"
                    "- 'fast' (default, ~3s, photo-real): immersive scenes.\n"
                    "- 'high' (~15-30s): only for infographics, charts, posters, maps, or where text "
                    "  must be readable. 'high' forces mode='create'.\n"
                    "`prompt` is a concrete 1-3 sentence English image-generation instruction.\n"
                    "When an image is attached, ground references in what is actually visible. "
                    "For edits, name what changes AND what stays the same.\n"
                    f"State: {'an image is attached — prefer edit unless user clearly wants a new world' if has_image else 'no image yet, must be create'}."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        text={"format": {"type": "json_schema", "name": "router", "schema": ROUTER_SCHEMA, "strict": True}},
    )
    decision = json.loads(response.output_text)

    if decision["mode"] not in ("create", "edit"):
        decision["mode"] = "edit" if has_image else "create"
    if decision["quality"] not in ("fast", "high"):
        decision["quality"] = "fast"
    if decision["quality"] == "high":
        decision["mode"] = "create"
    if not decision.get("prompt"):
        decision["prompt"] = text
    return decision


REALTIME_INSTRUCTIONS = """
You are Dreamwalk, a warm, concise voice guide for a generated visual world.
The user is looking at the latest generated image. You decide what happens on every turn.

ALWAYS speak in English, regardless of any other language that may appear in the
audio (echo, background voices, mid-word artifacts). Never switch to another language.

You have these tools:
- update_scene: regenerate or edit the image. Use for new worlds, movement,
  lighting changes, style changes, adding/removing things — anything visual.
  Do NOT call for questions or chit-chat.
- search_web: search the live web for current facts, recent news, or specific real-world
  details you cannot already know. Use ONLY for time-sensitive or specific factual lookups,
  NEVER for generic world-building. After the results return, typically call update_scene
  next with a prompt grounded in what you learned.
- recall_image: load a specific PAST image into your visual context. After every generation
  you receive a `recent_images` manifest of {id, summary, seconds_ago}. The latest image is
  already attached. Use recall_image only when the user references an earlier scene.
- find_image: search the web and display a REAL photo. Use when the user wants an actual,
  specific real-world thing ("show me the new Tesla Cybertruck", "show me Tom Cruise").
  The top web image is shown directly. Do NOT use for imagined or stylized scenes.

update_scene has two modes:
- "create": generate a brand-new image. Use for the very first scene
  ("show me Sydney harbour", "a quiet forest at dawn"), or when the user clearly
  asks for a different world.
- "edit": modify the most recent image. Use for any continuation: walking closer,
  looking around, going on top of a building, changing lighting, swapping the style,
  adding or removing things. The previous image is used as a visual reference, so
  small spatial moves stay continuous.

It also takes a quality parameter:
- "fast" (default): fast photo-real image model (~3s). Use for immersive scenes,
  exploration, atmospheric worlds.
- "high": high-quality model (~15-30s, much slower). Use ONLY when the user asks
  for an infographic, diagram, chart, poster, map, or anything where text, labels,
  or signs must be readable. The fast model cannot render text reliably; the high
  model can. When you choose "high", you MUST use mode="create" (no continuation).
  When you call with quality="high", warn the user briefly that it'll take a moment.

The "prompt" parameter is a full English image-generation instruction in concrete
visual language, 1-3 sentences. Translate the user's casual speech (in any language)
into that. For edits, name what changes AND what stays the same.

After every image is generated, the freshly rendered image is appended to this
conversation as a user-side input image. Look at it. Ground your next prompts and
your answers in what is actually visible there — refer to specific landmarks,
objects, lighting, and layout you can see, not generic placeholders.

When the user says "that bridge", "the building on the left", "go on top of it",
resolve the reference against the current image you can see. Only call
update_scene with mode="edit" when continuing from the visible frame.

LATENCY HIDING (very important):
When you call update_scene, in the SAME response also speak one short phrase
(3-6 words) like "Okay, going closer now", "Sure, taking you up there",
"On it — heading over". The user will hear this while the image renders, so the
wait feels much shorter. Do NOT wait for the tool result before speaking.

After the tool result returns, mention one specific thing you can see in the
new image in under 10 words, then stop. Be terse — this is a live visual demo.
"""


SCENE_TOOL = {
    "type": "function",
    "name": "update_scene",
    "description": (
        "Generate a new image or edit the existing image. Only call when the user wants "
        "the visual scene to change. Do not call for questions or chit-chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["create", "edit"],
                "description": "'create' for a brand-new world. 'edit' to modify the most recent image. Must be 'create' when quality is 'high'.",
            },
            "quality": {
                "type": "string",
                "enum": ["fast", "high"],
                "description": "'fast' (default, ~3s, photo-real) for immersive scenes. 'high' (~15-30s, slower but renders text/diagrams) ONLY for infographics, charts, posters, maps, or images that need readable text.",
            },
            "prompt": {
                "type": "string",
                "description": "Full English image-generation prompt in concrete visual language. For edits, describe the new frame and what stays the same.",
            },
        },
        "required": ["mode", "quality", "prompt"],
    },
}


RECALL_TOOL = {
    "type": "function",
    "name": "recall_image",
    "description": (
        "Load a specific previously-generated image into your visual context. After this returns, "
        "the chosen image is appended to the conversation as an input_image you can see. Use this "
        "when the user references an earlier scene ('go back to the rooftop', 'the one before that') "
        "or when you need to look at a past frame to answer a question. The latest image is already "
        "in your context — only call this for non-latest images. Use an id from the recent_images "
        "manifest you receive after every generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string",
                "description": "The 8-char hex id of the image to load.",
            },
        },
        "required": ["image_id"],
    },
}


FIND_IMAGE_TOOL = {
    "type": "function",
    "name": "find_image",
    "description": (
        "Search the web for a REAL photographic image and display it as the current scene. "
        "Use this when the user wants to see an actual real-world thing — a specific place, "
        "building, product, person, or event ('show me UTS Startup Sydney', 'show me the new "
        "Apple Vision Pro'). The top web result is automatically displayed. Do NOT use for "
        "fictional/imagined/stylized scenes — use update_scene for those. Briefly say what "
        "you're searching for while the lookup runs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Image search query, 2-8 words, specific.",
            },
        },
        "required": ["query"],
    },
}


SEARCH_TOOL = {
    "type": "function",
    "name": "search_web",
    "description": (
        "Search the live web for current facts, recent news, or specific real-world details. "
        "Use ONLY when you need information you cannot reasonably already know — current events, "
        "today's weather/prices, the look of a recently launched product, real-world facts. "
        "Do NOT search for general world-building or creative scenes. Returns the top web "
        "snippets which you should then use to ground a follow-up update_scene call or to "
        "answer the user accurately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused web search query, 3-10 words.",
            },
        },
        "required": ["query"],
    },
}


def realtime_session_config() -> dict:
    return {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "instructions": textwrap.dedent(REALTIME_INSTRUCTIONS).strip(),
        "audio": {"output": {"voice": REALTIME_VOICE}},
        "tools": [SCENE_TOOL, SEARCH_TOOL, RECALL_TOOL, FIND_IMAGE_TOOL],
        "tool_choice": "auto",
    }


CSS = """
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #050505;
  color: #f4f4f1;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background: #050505;
  overflow: hidden;
}

.stage {
  position: relative;
  min-height: 100vh;
  background: #050505;
}

.scene-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0;
  transform: scale(1.03);
  transition: opacity 700ms ease, transform 1400ms ease;
}

.scene-img.ready {
  opacity: 1;
  transform: scale(1);
}

.scene-img.walking {
  transform: scale(1.08);
  filter: saturate(1.05) contrast(1.03);
}

.vignette {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(180deg, rgb(0 0 0 / 0.52), transparent 24%, transparent 62%, rgb(0 0 0 / 0.72)),
    radial-gradient(circle at center, transparent 45%, rgb(0 0 0 / 0.44));
}

.hud {
  position: absolute;
  left: 24px;
  right: 24px;
  bottom: 24px;
}

.panel {
  max-width: 820px;
  border: 1px solid rgb(255 255 255 / 0.15);
  background: rgb(5 5 5 / 0.72);
  backdrop-filter: blur(18px);
  border-radius: 8px;
  padding: 12px;
}

.status {
  min-height: 24px;
  color: #d8d8d2;
  font-size: 14px;
  line-height: 1.4;
  margin-bottom: 10px;
}

.controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 8px;
}

input {
  width: 100%;
  min-width: 0;
  border: 1px solid rgb(255 255 255 / 0.18);
  background: rgb(255 255 255 / 0.08);
  color: #fff;
  border-radius: 6px;
  padding: 12px 14px;
  font: inherit;
  outline: none;
}

input:focus { border-color: #9ad7ff; }

button {
  border: 1px solid rgb(255 255 255 / 0.18);
  background: #f4f4f1;
  color: #080808;
  border-radius: 6px;
  padding: 0 14px;
  min-height: 46px;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}

button.secondary {
  background: rgb(255 255 255 / 0.08);
  color: #f4f4f1;
}

button.connected {
  background: #9ad7ff;
  color: #061018;
}

button[disabled] {
  opacity: 0.6;
  cursor: wait;
}

.loading {
  position: absolute;
  top: 24px;
  right: 24px;
  display: none;
  border: 1px solid rgb(255 255 255 / 0.16);
  background: rgb(5 5 5 / 0.68);
  color: #f4f4f1;
  border-radius: 999px;
  padding: 9px 12px;
  font-size: 13px;
}

.loading.visible { display: block; }

.subtitles {
  position: absolute;
  left: 50%;
  bottom: 130px;
  transform: translateX(-50%);
  max-width: 760px;
  padding: 8px 18px;
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  border-radius: 6px;
  font-size: 17px;
  line-height: 1.4;
  text-align: center;
  pointer-events: none;
  backdrop-filter: blur(10px);
  opacity: 0;
  transition: opacity 220ms ease;
  font-family: Inter, system-ui, sans-serif;
}

.subtitles:not(:empty) {
  opacity: 1;
}

.transcript {
  position: absolute;
  left: 50%;
  bottom: 184px;
  transform: translateX(-50%);
  max-width: 760px;
  padding: 6px 14px;
  background: rgb(0 0 0 / 0.4);
  color: #d8d8d2;
  border-radius: 6px;
  font-size: 14px;
  line-height: 1.35;
  text-align: center;
  pointer-events: none;
  backdrop-filter: blur(8px);
  opacity: 0;
  transition: opacity 220ms ease;
  font-family: Inter, system-ui, sans-serif;
  font-style: italic;
}

.transcript:not(:empty) {
  opacity: 1;
}

.orb-wrap {
  position: absolute;
  right: 28px;
  bottom: 28px;
  display: flex;
  flex-direction: column-reverse;
  align-items: center;
  gap: 6px;
  pointer-events: auto;
  z-index: 10;
}

.orb {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  cursor: pointer;
  user-select: none;
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,0.7), transparent 48%),
    radial-gradient(circle, #555562 0%, #1a1a22 70%);
  box-shadow:
    0 0 calc(20px + var(--lvl, 0) * 70px) var(--orb-glow, rgba(255,255,255,0.18)),
    inset 0 -6px 14px rgba(0,0,0,0.45),
    inset 0 4px 10px rgba(255,255,255,0.10);
  transform: scale(calc(1 + var(--lvl, 0) * 0.28));
  transition: background 600ms ease, box-shadow 120ms linear;
}

.orb-wrap[data-state="idle"] .orb {
  --orb-glow: rgba(255,255,255,0.18);
  animation: orbBreath 3.6s ease-in-out infinite;
}

@keyframes orbBreath {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.06); }
}

.orb-wrap[data-state="listening"] .orb {
  --orb-glow: rgba(154, 215, 255, 0.55);
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,0.7), transparent 48%),
    radial-gradient(circle, #4aa5e8 0%, #0c3a66 75%);
}

.orb-wrap[data-state="speaking"] .orb {
  --orb-glow: rgba(255, 200, 130, 0.55);
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,0.7), transparent 48%),
    radial-gradient(circle, #f0a85a 0%, #4a2308 75%);
}

.orb-wrap[data-state="busy"] .orb {
  --orb-glow: rgba(200, 150, 255, 0.55);
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,0.6), transparent 48%),
    radial-gradient(circle, #b88af0 0%, #2a1247 75%);
  animation: orbThink 1.6s ease-in-out infinite;
}

@keyframes orbThink {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.10); }
}

.orb-caption {
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: rgb(255 255 255 / 0.55);
  white-space: nowrap;
}

.log-toggle {
  position: absolute;
  top: 24px;
  right: 24px;
  z-index: 60;
  background: rgb(5 5 5 / 0.7);
  border: 1px solid rgb(255 255 255 / 0.16);
  color: #f4f4f1;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  cursor: pointer;
  backdrop-filter: blur(12px);
  min-height: 0;
}

.log-panel {
  position: fixed;
  top: 0;
  right: -440px;
  width: 420px;
  max-width: 90vw;
  height: 100vh;
  background: rgb(8 8 12 / 0.94);
  backdrop-filter: blur(20px);
  border-left: 1px solid rgb(255 255 255 / 0.10);
  transition: right 300ms ease;
  z-index: 50;
  display: flex;
  flex-direction: column;
  font-family: ui-monospace, "SF Mono", Menlo, Monaco, monospace;
  font-size: 11.5px;
  color: #e4e4e0;
}

.log-panel.open { right: 0; }

.log-panel-header {
  padding: 14px 16px;
  border-bottom: 1px solid rgb(255 255 255 / 0.08);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: Inter, system-ui, sans-serif;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #d8d8d2;
}

.log-panel-clear {
  background: transparent;
  border: 1px solid rgb(255 255 255 / 0.18);
  color: #d8d8d2;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  cursor: pointer;
  letter-spacing: 0.04em;
  text-transform: none;
  min-height: 0;
  font-weight: 400;
}

.log-content {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.log-entry {
  border-left: 3px solid #444;
  padding: 6px 9px;
  background: rgb(255 255 255 / 0.03);
  border-radius: 0 4px 4px 0;
  word-break: break-word;
  white-space: pre-wrap;
  line-height: 1.4;
}

.log-entry .time {
  color: #777;
  font-size: 10px;
  display: inline-block;
  margin-right: 8px;
}

.log-entry .label {
  color: #ccc;
  font-weight: 700;
  margin-right: 6px;
}

.log-entry .body {
  display: block;
  margin-top: 4px;
  color: #d8d8d2;
}

.log-entry.user      { border-left-color: #9ad7ff; }
.log-entry.ai        { border-left-color: #f0a85a; }
.log-entry.tool-call { border-left-color: #b88af0; }
.log-entry.tool-out  { border-left-color: #6cd484; }
.log-entry.error     { border-left-color: #ff6b6b; }
.log-entry.system    { border-left-color: #888; }

.log-entry.user .label      { color: #9ad7ff; }
.log-entry.ai .label        { color: #f0a85a; }
.log-entry.tool-call .label { color: #b88af0; }
.log-entry.tool-out .label  { color: #6cd484; }
.log-entry.error .label     { color: #ff6b6b; }

@media (max-width: 720px) {
  body { overflow: auto; }
  .stage { min-height: 100svh; }
  .hud { left: 12px; right: 12px; bottom: 12px; }
  .controls { grid-template-columns: 1fr; }
  button { width: 100%; }
}
"""


JS = """
const sessionId = (typeof crypto !== "undefined" && crypto.randomUUID)
  ? crypto.randomUUID()
  : "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
const img = document.querySelector("#scene");
const statusEl = document.querySelector("#status");
const loadingEl = document.querySelector("#loading");
const orbWrap = document.querySelector("#orb-wrap");
const orbEl = document.querySelector("#orb");
const orbCaption = document.querySelector("#orb-caption");
const logPanel = document.querySelector("#log-panel");
const logContent = document.querySelector("#log-content");
const logToggle = document.querySelector("#log-toggle");
const logClear = document.querySelector("#log-clear");

function isHttpUrl(u) {
  return typeof u === "string" && /^https?:\\/\\//.test(u);
}

let toolQueue = Promise.resolve();
function enqueueFunctionCall(call) {
  toolQueue = toolQueue
    .catch(() => {})
    .then(() => handleFunctionCall(call))
    .catch((err) => console.error("[dreamwalk:realtime] tool failed", err));
}

let aiTranscriptBuf = "";

function fmtBody(body) {
  if (body == null) return "";
  if (typeof body === "string") return body;
  try { return JSON.stringify(body, null, 2); }
  catch { return String(body); }
}

function logEvent(kind, label, body) {
  if (!logContent) return;
  const entry = document.createElement("div");
  entry.className = `log-entry ${kind}`;
  const time = new Date().toLocaleTimeString();

  const head = document.createElement("div");
  const t = document.createElement("span"); t.className = "time"; t.textContent = time;
  const l = document.createElement("span"); l.className = "label"; l.textContent = label;
  head.appendChild(t); head.appendChild(l);
  entry.appendChild(head);

  const text = fmtBody(body);
  if (text) {
    const b = document.createElement("span"); b.className = "body"; b.textContent = text;
    entry.appendChild(b);
  }
  logContent.prepend(entry);
}

logToggle.addEventListener("click", () => {
  const open = logPanel.classList.toggle("open");
  logToggle.textContent = open ? "HIDE" : "EVENTS";
});
logClear.addEventListener("click", () => { logContent.innerHTML = ""; });

let audioCtx = null;
let micAnalyser = null;
let outAnalyser = null;
let orbRAF = null;
let smoothMic = 0;
let smoothOut = 0;

function ensureAudioCtx() {
  if (!audioCtx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctor();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function attachAnalyser(stream) {
  if (!stream) return null;
  const ctx = ensureAudioCtx();
  const src = ctx.createMediaStreamSource(stream);
  const a = ctx.createAnalyser();
  a.fftSize = 64;
  a.smoothingTimeConstant = 0.7;
  src.connect(a);
  return a;
}

function streamLevel(a) {
  if (!a) return 0;
  const data = new Uint8Array(a.frequencyBinCount);
  a.getByteFrequencyData(data);
  let sum = 0;
  for (const v of data) sum += v;
  return Math.min(1, (sum / data.length / 255) * 1.8);
}

function startOrbLoop() {
  if (orbRAF) return;
  function tick() {
    const m = streamLevel(micAnalyser);
    const o = streamLevel(outAnalyser);
    smoothMic = smoothMic * 0.78 + m * 0.22;
    smoothOut = smoothOut * 0.78 + o * 0.22;

    let state = "idle";
    if (busy) state = "busy";
    else if (smoothOut > 0.06) state = "speaking";
    else if (smoothMic > 0.06) state = "listening";

    orbWrap.dataset.state = state;
    const lvl = state === "speaking" ? smoothOut : state === "listening" ? smoothMic : 0;
    orbEl.style.setProperty("--lvl", lvl.toFixed(3));

    if (state === "listening") orbCaption.textContent = "listening";
    else if (state === "speaking") orbCaption.textContent = "speaking";
    else if (state === "busy") orbCaption.textContent = "rendering";
    else orbCaption.textContent = "tap to mute";

    orbRAF = requestAnimationFrame(tick);
  }
  tick();
}

function stopOrbLoop() {
  if (orbRAF) cancelAnimationFrame(orbRAF);
  orbRAF = null;
  smoothMic = 0;
  smoothOut = 0;
  micAnalyser = null;
  outAnalyser = null;
  orbEl.style.setProperty("--lvl", "0");
  orbWrap.dataset.state = "idle";
  orbCaption.textContent = "tap to start voice";
}

let busy = false;
let realtime = null;

function setBusy(nextBusy, walking = false) {
  busy = nextBusy;
  loadingEl.classList.toggle("visible", nextBusy);
  img.classList.toggle("walking", walking);
}

function showImage(url) {
  const next = new Image();
  next.onload = () => {
    img.classList.remove("ready");
    requestAnimationFrame(() => {
      img.src = url;
      img.classList.add("ready");
    });
  };
  next.src = url;
}

async function postCommand(body) {
  const response = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Command failed");
  return data;
}

async function addVisibleImageToRealtime(imageUrl, text) {
  if (!isRealtimeConnected()) return false;
  try {
    let realtimeImageUrl = "";
    if (imageUrl) {
      const response = await fetch("/api/realtime-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, image_url: imageUrl }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not prepare image context");
      realtimeImageUrl = data.image_url;
    }

    const content = [{ type: "input_text", text }];
    if (/^data:image\\/(png|jpeg);base64,/i.test(realtimeImageUrl || "")) {
      content.push({ type: "input_image", image_url: realtimeImageUrl, detail: "low" });
    }

    sendRealtime({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content },
    });
    return true;
  } catch (error) {
    console.warn("[dreamwalk:realtime] image context skipped", error);
    logEvent("error", "image context skipped", error.message || String(error));
    try {
      sendRealtime({
        type: "conversation.item.create",
        item: {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: text + " Image context could not be attached." }],
        },
      });
    } catch {}
    return false;
  }
}

async function runSceneUpdate({ mode, quality, prompt }) {
  if (!prompt.trim() || busy) return null;
  setBusy(true, mode === "edit");
  const label = quality === "high" ? "Rendering hi-res (slower): " : (mode === "create" ? "Creating: " : "Editing: ");
  statusEl.textContent = label + prompt;
  try {
    const data = await postCommand({ mode, quality, prompt });
    showImage(data.image);
    statusEl.textContent = "Done.";
    return data;
  } catch (error) {
    statusEl.textContent = error.message;
    throw error;
  } finally {
    setBusy(false, false);
  }
}

async function runTypedCommand(text) {
  if (!text.trim() || busy) return;
  setBusy(true, true);
  statusEl.textContent = text;
  logEvent("user", "you typed", text);
  try {
    const data = await postCommand({ text });
    showImage(data.image);
    statusEl.textContent = "Done.";
    logEvent("tool-out", "image", { mode: data.mode, quality: data.quality, image_id: data.image_id });
  } catch (error) {
    statusEl.textContent = error.message;
    logEvent("error", "command failed", error.message);
  } finally {
    setBusy(false, false);
  }
}

function isRealtimeConnected() {
  return Boolean(realtime?.dc && realtime.dc.readyState === "open");
}

// Track active response IDs from response.created / response.done events.
// Realtime allows only one active response at a time — sending response.create
// while one is in flight gives "conversation_already_has_active_response" and
// the agent goes silent for the rest of the turn.
const activeResponses = new Set();

function rawSend(event) {
  if (!isRealtimeConnected()) throw new Error("Voice is not connected");
  console.info("[dreamwalk:realtime] send", event.type);
  realtime.dc.send(JSON.stringify(event));
}

// Queue response.create sends behind any in-flight response. If several pile up,
// the latest one wins (its instructions reflect the latest tool result).
let pendingResponseCreate = null;
async function flushPendingResponseCreate() {
  if (!pendingResponseCreate) return;
  const start = Date.now();
  while (activeResponses.size > 0 && Date.now() - start < 4000) {
    await new Promise((r) => setTimeout(r, 50));
  }
  if (activeResponses.size > 0) {
    console.warn("[dreamwalk:realtime] response-clear wait timed out, sending anyway");
  }
  const ev = pendingResponseCreate;
  pendingResponseCreate = null;
  if (isRealtimeConnected()) rawSend(ev);
}

function sendRealtime(event) {
  if (event?.type === "response.create" && activeResponses.size > 0) {
    if (pendingResponseCreate) {
      console.info("[dreamwalk:realtime] coalescing response.create");
    } else {
      console.info("[dreamwalk:realtime] queue response.create (active=", activeResponses.size, ")");
    }
    pendingResponseCreate = event;
    flushPendingResponseCreate();
    return;
  }
  if (!isRealtimeConnected()) throw new Error("Voice is not connected");
  console.info("[dreamwalk:realtime] send", event.type);
  realtime.dc.send(JSON.stringify(event));
  if (event?.item?.type === "function_call_output") {
    let parsed = event.item.output;
    try { parsed = JSON.parse(event.item.output); } catch {}
    const kind = parsed && parsed.ok === false ? "error" : "tool-out";
    logEvent(kind, "← tool result", parsed);
  }
}

function setVoiceConnected(connected) {
  // No-op: orb is the connection indicator now (data-state on orbWrap).
  void connected;
}

async function handleSearchCall(call, args) {
  const query = (args.query || "").trim();
  if (!query) {
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: "empty query" }) },
    });
    sendRealtime({ type: "response.create" });
    return;
  }
  statusEl.textContent = `Searching: ${query}`;
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Search failed");
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: true, results: data.results || [] }) },
    });
    sendRealtime({
      type: "response.create",
      response: { instructions: "Use the search results to inform the next action. If the user wants a visual, call update_scene with a prompt grounded in what you learned. Otherwise answer briefly." },
    });
  } catch (error) {
    statusEl.textContent = error.message;
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: error.message }) },
    });
    sendRealtime({ type: "response.create", response: { instructions: "Briefly say the search failed." } });
  }
}

async function handleFindImageCall(call, args) {
  const query = (args.query || "").trim();
  if (!query) {
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: "empty query" }) },
    });
    sendRealtime({ type: "response.create" });
    return;
  }
  setBusy(true, false);
  statusEl.textContent = `Finding: ${query}`;
  try {
    const response = await fetch("/api/find-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "find_image failed");
    showImage(data.image);
    statusEl.textContent = "";
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({
        ok: true,
        image_id: data.image_id,
        title: data.title,
        source: data.source,
        alternatives: data.alternatives || [],
        recent_images: data.recent_images || [],
      }) },
    });
    await addVisibleImageToRealtime(data.image, `(Real web image now visible: ${data.title || query})`);
    sendRealtime({
      type: "response.create",
      response: { instructions: "In under 10 words, briefly say what real image you found. Then stop." },
    });
  } catch (error) {
    statusEl.textContent = error.message;
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: error.message }) },
    });
    sendRealtime({ type: "response.create", response: { instructions: "Briefly say the image search failed." } });
  } finally {
    setBusy(false, false);
  }
}

async function handleRecallCall(call, args) {
  const imageId = (args.image_id || "").trim();
  if (!imageId) {
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: "empty image_id" }) },
    });
    sendRealtime({ type: "response.create" });
    return;
  }
  try {
    const response = await fetch("/api/recall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, image_id: imageId }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Recall failed");
    if (data.url) showImage(data.url);
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: true, image_id: data.id, summary: data.summary, seconds_ago: data.seconds_ago }) },
    });
    await addVisibleImageToRealtime(data.url, `(Recalled image ${imageId}: ${data.summary})`);
    sendRealtime({
      type: "response.create",
      response: { instructions: "Use the image you just loaded to answer or to ground your next action. Be brief." },
    });
  } catch (error) {
    statusEl.textContent = error.message;
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: error.message }) },
    });
    sendRealtime({ type: "response.create", response: { instructions: "Briefly say the recall failed." } });
  }
}

async function handleFunctionCall(call) {
  let args = {};
  try { args = JSON.parse(call.arguments || "{}"); }
  catch (e) { console.error("[dreamwalk:realtime] bad args", e); }

  logEvent("tool-call", `→ ${call.name}`, args);

  if (call.name === "find_image") {
    return handleFindImageCall(call, args);
  }
  if (call.name === "recall_image") {
    return handleRecallCall(call, args);
  }
  if (call.name === "search_web") {
    return handleSearchCall(call, args);
  }
  if (call.name !== "update_scene") {
    console.warn("[dreamwalk:realtime] unknown tool", call.name);
    sendRealtime({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: call.call_id, output: JSON.stringify({ ok: false, error: `unknown tool ${call.name}` }) },
    });
    sendRealtime({ type: "response.create" });
    return;
  }

  const quality = args.quality === "high" ? "high" : "fast";
  const mode = quality === "high" ? "create" : (args.mode === "edit" ? "edit" : "create");
  const prompt = (args.prompt || "").trim();

  if (!prompt) {
    sendRealtime({
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: call.call_id,
        output: JSON.stringify({ ok: false, error: "empty prompt" }),
      },
    });
    sendRealtime({ type: "response.create" });
    return;
  }

  try {
    const data = await runSceneUpdate({ mode, quality, prompt });
    sendRealtime({
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: call.call_id,
        output: JSON.stringify({ ok: true, mode, quality, image_id: data?.image_id, recent_images: data?.recent_images || [] }),
      },
    });
    await addVisibleImageToRealtime(data?.image, "(This is the image now visible to me.)");
    sendRealtime({
      type: "response.create",
      response: { instructions: "In under 10 words, mention one specific thing you can see in the new image. Then stop." },
    });
  } catch (error) {
    sendRealtime({
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: call.call_id,
        output: JSON.stringify({ ok: false, error: error.message }),
      },
    });
    sendRealtime({
      type: "response.create",
      response: { instructions: "Briefly say the image update failed." },
    });
  }
}

function handleRealtimeEvent(event) {
  if (event.type === "error") {
    statusEl.textContent = event.error?.message || "Realtime error";
    logEvent("error", "realtime", event.error || event);
    return;
  }
  if (event.type === "session.created" || event.type === "session.updated") {
    statusEl.textContent = "Voice connected";
    return;
  }
  if (event.type === "input_audio_buffer.speech_started") {
    statusEl.textContent = "Listening...";
    return;
  }
  if (event.type === "conversation.item.input_audio_transcription.completed") {
    if (event.transcript?.trim()) logEvent("user", "you said", event.transcript.trim());
    return;
  }
  if (event.type === "response.output_audio_transcript.delta") {
    if (event.delta) {
      aiTranscriptBuf += event.delta;
      if (event.delta.trim()) statusEl.textContent = aiTranscriptBuf.slice(-160);
    }
    return;
  }
  if (event.type === "response.output_audio_transcript.done") {
    const text = (event.transcript || aiTranscriptBuf || "").trim();
    if (text) logEvent("ai", "ai said", text);
    aiTranscriptBuf = "";
    setTimeout(() => { statusEl.textContent = ""; }, 2500);
    return;
  }
  if (event.type === "response.created") {
    if (event.response?.id) activeResponses.add(event.response.id);
    console.info("[dreamwalk:realtime] response.created", event.response?.id, "active=", activeResponses.size);
    return;
  }
  if (event.type === "response.cancelled" || event.type === "response.failed") {
    if (event.response?.id) activeResponses.delete(event.response.id);
    logEvent("error", event.type, event.response || event);
    console.warn("[dreamwalk:realtime]", event.type, event.response || event);
    return;
  }
  if (event.type === "response.done") {
    if (event.response?.id) activeResponses.delete(event.response.id);
    if (aiTranscriptBuf.trim()) {
      logEvent("ai", "ai said", aiTranscriptBuf.trim());
      aiTranscriptBuf = "";
    }
    const status = event.response?.status;
    if (status && status !== "completed") {
      logEvent("error", `response.done status=${status}`, event.response?.status_details || event.response);
      console.warn("[dreamwalk:realtime] response.done non-completed", status, event.response);
    }
    const calls = (event.response?.output || []).filter((item) => item.type === "function_call");
    for (const call of calls) enqueueFunctionCall(call);
    return;
  }
}

async function connectRealtime() {
  if (isRealtimeConnected()) return;
  statusEl.textContent = "Connecting voice...";

  const pc = new RTCPeerConnection();
  const dc = pc.createDataChannel("oai-events");
  const audio = document.createElement("audio");
  audio.autoplay = true;

  pc.ontrack = (event) => {
    audio.srcObject = event.streams[0];
    try { outAnalyser = attachAnalyser(event.streams[0]); } catch (e) { console.warn("out analyser failed", e); }
  };
  dc.addEventListener("message", (event) => handleRealtimeEvent(JSON.parse(event.data)));

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const track of stream.getTracks()) pc.addTrack(track, stream);
  try { micAnalyser = attachAnalyser(stream); } catch (e) { console.warn("mic analyser failed", e); }
  startOrbLoop();

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const sdpResponse = await fetch("/api/realtime/session", {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: offer.sdp,
  });
  if (!sdpResponse.ok) throw new Error((await sdpResponse.text()) || "Failed to connect voice");

  await pc.setRemoteDescription({ type: "answer", sdp: await sdpResponse.text() });
  await new Promise((resolve, reject) => {
    if (dc.readyState === "open") return resolve();
    const t = setTimeout(() => reject(new Error("Realtime data channel timed out")), 10000);
    dc.addEventListener("open", () => { clearTimeout(t); resolve(); }, { once: true });
  });

  realtime = { pc, dc, stream, audio };
  setVoiceConnected(true);
  statusEl.textContent = "";
  logEvent("system", "voice connected", "");

  sendRealtime({
    type: "response.create",
    response: { instructions: "Greet the user exactly: Hi, what are we building today?" },
  });
}

function disconnectRealtime() {
  if (!realtime) return;
  realtime.stream?.getTracks().forEach((track) => track.stop());
  realtime.dc?.close();
  realtime.pc?.close();
  realtime.audio?.remove();
  realtime = null;
  activeResponses.clear();
  pendingResponseCreate = null;
  setVoiceConnected(false);
  statusEl.textContent = "Voice disconnected";
  stopOrbLoop();
  logEvent("system", "voice disconnected", "");
}

orbEl.addEventListener("click", async () => {
  try {
    if (isRealtimeConnected()) { disconnectRealtime(); return; }
    await connectRealtime();
  } catch (error) {
    console.error("[dreamwalk:realtime] connect failed", error);
    setVoiceConnected(false);
    statusEl.textContent = error.message;
    stopOrbLoop();
  }
});

img.classList.add("ready");
setVoiceConnected(false);
"""


app, rt = fast_app(
    hdrs=(
        Meta(charset="utf-8"),
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Style(CSS),
    )
)


@rt("/")
def get():
    logger.info("Serving Dreamwalk UI")
    return (
        Title("Dreamwalk"),
        Main(
            Img(id="scene", cls="scene-img", alt="", src=""),
            Div(cls="vignette"),
            NotStr(
                '<div id="orb-wrap" class="orb-wrap" data-state="idle">'
                '<div id="orb" class="orb" role="button" aria-label="Toggle voice"></div>'
                '<div id="orb-caption" class="orb-caption">tap to start voice</div>'
                '</div>'
            ),
            NotStr(
                '<button id="log-toggle" class="log-toggle" type="button">EVENTS</button>'
                '<aside id="log-panel" class="log-panel">'
                '<div class="log-panel-header">'
                '<span>Event log</span>'
                '<button id="log-clear" class="log-panel-clear" type="button">Clear</button>'
                '</div>'
                '<div id="log-content" class="log-content"></div>'
                '</aside>'
            ),
            Div("Generating...", id="loading", cls="loading"),
            Div("", id="status", cls="subtitles"),
            Div("", id="transcript", cls="transcript"),
            Script(JS),
            cls="stage",
        ),
    )


@rt("/api/command", methods=["POST"])
async def command(request: Request):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        payload = await request.json()
        session_id = payload.get("session_id") or str(uuid.uuid4())
        mode = payload.get("mode")
        quality = payload.get("quality") or "fast"
        prompt = (payload.get("prompt") or "").strip()
        text = (payload.get("text") or "").strip()

        state = get_state(session_id)
        async with state.lock:
            # Typed-command path: route the raw text through one LLM call.
            if not mode and text:
                logger.info("[%s] Routing typed command text=%r", request_id, text)
                decision = await asyncio.to_thread(route_text_command, state, text)
                mode = decision["mode"]
                quality = decision.get("quality", "fast")
                prompt = decision.get("prompt") or ""
                logger.info(
                    "[%s] Routed mode=%s quality=%s prompt=%r",
                    request_id, mode, quality, prompt,
                )

            if mode not in ("create", "edit"):
                return JSONResponse({"error": "mode must be 'create' or 'edit'"}, status_code=400)
            if quality not in ("fast", "high"):
                quality = "fast"
            if quality == "high":
                mode = "create"
            if not prompt:
                return JSONResponse({"error": "prompt is required"}, status_code=400)

            reference = state.last_image_url if mode == "edit" else ""
            if mode == "edit" and not (reference and reference.startswith("http")):
                logger.info("[%s] mode=edit without http reference; falling back to create", request_id)
                reference = ""

            logger.info("[%s] %s quality=%s prompt_chars=%s", request_id, mode, quality, len(prompt))
            image = await asyncio.to_thread(generate_image, prompt, quality, reference)
            record = add_image_record(state, image, prompt)
            manifest = recent_image_manifest(state)

        elapsed = time.perf_counter() - started
        logger.info("[%s] command done in %.2fs id=%s", request_id, elapsed, record.id)
        return JSONResponse({
            "kind": "image",
            "image": image,
            "mode": mode,
            "quality": quality,
            "image_id": record.id,
            "summary": record.summary,
            "recent_images": manifest,
        })
    except Exception as exc:
        logger.exception("[%s] command failed", request_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rt("/api/realtime-image", methods=["POST"])
async def realtime_image(request: Request):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        payload = await request.json()
        session_id = payload.get("session_id") or ""
        image_url = (payload.get("image_url") or "").strip()
        if not image_url:
            return JSONResponse({"error": "image_url is required"}, status_code=400)

        state = get_state(session_id)
        session_urls = {state.last_image_url}
        session_urls.update(r.url for r in state.images)
        if image_url not in session_urls:
            return JSONResponse({"error": "image_url is not part of this session"}, status_code=403)

        data_uri = await asyncio.to_thread(realtime_image_data_uri, image_url)
        elapsed = time.perf_counter() - started
        logger.info("[%s] realtime-image done in %.2fs chars=%d", request_id, elapsed, len(data_uri))
        return JSONResponse({"image_url": data_uri})
    except Exception as exc:
        logger.exception("[%s] realtime-image failed", request_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rt("/api/recall", methods=["POST"])
async def recall(request: Request):
    request_id = uuid.uuid4().hex[:8]
    try:
        payload = await request.json()
        session_id = payload.get("session_id") or ""
        image_id = (payload.get("image_id") or "").strip()
        if not image_id:
            return JSONResponse({"error": "image_id is required"}, status_code=400)
        state = get_state(session_id)
        record = next((r for r in state.images if r.id == image_id), None)
        if not record:
            return JSONResponse({"error": f"image_id {image_id} not found"}, status_code=404)
        state.last_image_url = record.url
        logger.info("[%s] recall image_id=%s url=%s", request_id, image_id, record.url)
        return JSONResponse({
            "id": record.id,
            "url": record.url,
            "summary": record.summary,
            "prompt": record.prompt,
            "seconds_ago": int(time.time() - record.created_at),
        })
    except Exception as exc:
        logger.exception("[%s] recall failed", request_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rt("/api/find-image", methods=["POST"])
async def find_image(request: Request):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        payload = await request.json()
        session_id = payload.get("session_id") or ""
        query = (payload.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)

        results = search_images(query, count=5)
        if not results:
            return JSONResponse({"error": f"no images found for {query!r}"}, status_code=404)

        chosen = results[0]
        state = get_state(session_id)
        record = add_image_record(state, chosen["url"], f"web image: {query}")
        elapsed = time.perf_counter() - started
        logger.info(
            "[%s] find-image done in %.2fs query=%r picked=%s",
            request_id, elapsed, query, chosen["url"],
        )
        return JSONResponse({
            "image": chosen["url"],
            "image_id": record.id,
            "title": chosen["title"],
            "source": chosen["source"],
            "alternatives": [
                {"url": r["url"], "title": r["title"], "source": r["source"]}
                for r in results[1:]
            ],
            "recent_images": recent_image_manifest(state),
        })
    except Exception as exc:
        logger.exception("[%s] find-image failed", request_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rt("/api/search", methods=["POST"])
async def search(request: Request):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        payload = await request.json()
        query = (payload.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)
        results = await asyncio.to_thread(search_web, query, 5)
        elapsed = time.perf_counter() - started
        logger.info(
            "[%s] search done in %.2fs query=%r results=%d",
            request_id, elapsed, query, len(results),
        )
        return JSONResponse({"results": results})
    except Exception as exc:
        logger.exception("[%s] search failed", request_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rt("/api/realtime/session", methods=["POST"])
async def realtime_session(request: Request):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("[%s] Realtime session requested without OPENAI_API_KEY", request_id)
        return PlainTextResponse("OPENAI_API_KEY is not set", status_code=500)

    offer_sdp = (await request.body()).decode("utf-8")
    if not offer_sdp.strip():
        return PlainTextResponse("Missing SDP offer", status_code=400)

    session_config = realtime_session_config()
    logger.info(
        "[%s] Creating Realtime WebRTC session model=%s voice=%s sdp_chars=%s",
        request_id, REALTIME_MODEL, REALTIME_VOICE, len(offer_sdp),
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "sdp": (None, offer_sdp),
                    "session": (None, json.dumps(session_config)),
                },
            )
        elapsed = time.perf_counter() - started
        logger.info("[%s] Realtime session status=%s in %.2fs", request_id, response.status_code, elapsed)
        if response.status_code >= 400:
            logger.error("[%s] Realtime session failed: %s", request_id, response.text)
            return PlainTextResponse(response.text, status_code=response.status_code)
        return PlainTextResponse(response.text, media_type="application/sdp")
    except Exception:
        logger.exception("[%s] Realtime session creation failed", request_id)
        return PlainTextResponse("Failed to create Realtime session", status_code=500)


if __name__ == "__main__":
    serve(port=5001)
