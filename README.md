# Dreamwalk

**Speak a world into existence. Walk through it with your voice.**

Talk to a voice agent. It paints scenes for you in real time, and you keep going — *"show me Sydney in 2050,"* then *"walk to the bridge,"* then *"make it night,"* then *"actually, show me the real Cybertruck."* The image is always the last thing you said. The agent describes what it sees as it goes.

The trick: **the agent sees what it just made.** Every frame goes back into its vision context, so *"go on top of that building"* knows which building.

## What you can do with it

**Daydream a trip.** *"Show me a fishing village in northern Japan in autumn."* → *"Walk to the harbour."* → *"Now show me a real photo of Hakodate."* You drift between imagined and real without breaking flow.

**Worldbuild out loud.** Pitching a setting for a story or game? Talk through it. *"A desert city built into a canyon."* → *"Now from inside the market."* → *"Make it dawn."* Continuity holds across turns — it's the same city, not three new ones.

**Mood-board a space.** *"My dream studio apartment, lots of plants, big window."* → *"Add a reading nook by the window."* → *"More natural light."* Edits compound instead of starting over.

**Reach for a memory.** *"Show me what UTS Startup Sydney looks like."* (real photo) → *"Now imagine it in 2080."* (generated). The boundary is one sentence.

**Explain something visual mid-conversation.** *"Draw me a diagram of how a heat pump works"* — switches automatically to the slower, text-readable model when it matters.

## Try it

```bash
printf 'OPENAI_API_KEY=...\nFAL_KEY=...\nBRAVE_API_KEY=...\n' > .env
uv run python app.py
```

Open <http://localhost:5001>, hit **Start voice**, and talk.

## Built with

OpenAI Realtime API · fal flux-2 · Brave Search · FastHTML

One Python file. No build step. The whole app is in `app.py`.
