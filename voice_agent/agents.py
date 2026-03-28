"""Agents, confirmation loop, and dispatcher."""

from utils import log, speak, record_short, transcribe
from llm import interpret, clarify

# ── Affirmation detection ────────────────────────────────────────────
_AFFIRM_WORDS = {
    "alright", "okay", "proceed", "yes", "do it", "go ahead",
    "confirmed", "correct", "that's right", "perfect", "go for it",
    "sounds good", "yep", "sure", "affirmative",
}

CONFIRM_CHUNK = 5  # seconds to listen for affirmation/correction


def _is_affirmation(text: str) -> bool:
    """Check if text is an affirmation (not a correction)."""
    lower = text.lower().strip(" .,!?")
    # Short responses that match an affirmation phrase
    for phrase in _AFFIRM_WORDS:
        if phrase in lower:
            return True
    return False


def _format_plan(plan: dict) -> str:
    """Format plan dict into spoken/logged text."""
    understood = plan.get("understood", "I'm not sure what you want.")
    intent = plan.get("intent", "unknown")
    steps = plan.get("next_steps", [])

    lines = [f"I understood: {understood}"]
    lines.append(f"Intent: {intent}")
    if steps:
        lines.append("Next steps:")
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
    return "\n".join(lines)


# ── Confirmation loop ────────────────────────────────────────────────
def confirm_loop(transcript: str) -> dict | None:
    """
    Send transcript to LLM → speak understanding → listen for
    affirmation or correction → loop until affirmed → return plan.
    """
    # First interpretation
    plan = interpret(transcript)
    log("PLAN", "\n" + _format_plan(plan))
    speak_text = (
        f"I understood: {plan.get('understood', '?')}. "
        f"Next steps: {', '.join(plan.get('next_steps', []))}. "
        f"Say alright to proceed, or tell me what I got wrong."
    )
    speak(speak_text)

    # Confirmation loop
    while True:
        log("CONFIRM", "Listening for affirmation or correction…")
        audio = record_short(CONFIRM_CHUNK)
        response = transcribe(audio, model="active")
        log("CONFIRM:HEARD", f'"{response}"')

        if not response.strip():
            log("CONFIRM", "No speech detected — listening again…")
            continue

        if _is_affirmation(response):
            log("CONFIRM", "✅ Affirmed! Proceeding.")
            speak("Got it. Proceeding.")
            return plan

        # User is correcting — clarify with LLM
        log("CONFIRM", "Correction detected — re-interpreting…")
        plan = clarify(plan, response)
        log("PLAN", "\n" + _format_plan(plan))
        speak_text = (
            f"Updated understanding: {plan.get('understood', '?')}. "
            f"Next steps: {', '.join(plan.get('next_steps', []))}. "
            f"Say alright to proceed, or correct me again."
        )
        speak(speak_text)


# ── Stub agents ──────────────────────────────────────────────────────
def outreach_agent(plan: dict):
    """Stub: will be replaced with real email/outreach API."""
    log("AGENT:OUTREACH", f"Would execute plan: {plan}")


def calendar_agent(plan: dict):
    """Stub: will be replaced with real calendar API."""
    log("AGENT:CALENDAR", f"Would execute plan: {plan}")


# ── Dispatcher ───────────────────────────────────────────────────────
_AGENTS = {
    "outreach": outreach_agent,
    "calendar": calendar_agent,
}


def dispatch(plan: dict):
    """Route to the correct stub agent based on the LLM plan."""
    intent = plan.get("intent", "unknown")
    agent = _AGENTS.get(intent)
    if agent:
        log("DISPATCH", f"→ {agent.__name__} (intent: {intent})")
        agent(plan)
    else:
        log("DISPATCH", f"No agent for intent '{intent}'. Plan: {plan}")
