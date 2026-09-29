"""
Clinical Chart Assistant: Gradio front end.

Deterministic data (allergies, problems, medications, lab trends) is rendered straight from the
structured record, never through the language model. The model is used only for free-text
questions, and every answer is shown next to the exact sources it cites.
"""
import html
import inspect
import os
import threading

# Hugging Face's free tier now runs Gradio Spaces on ZeroGPU, which refuses to start unless the app
# registers at least one @spaces.GPU function. This app needs no GPU (models run at OpenAI), so we
# register a no-op placeholder that is never called. The `spaces` package exists only on Hugging Face,
# so locally and in CI this block does nothing. Import it before anything else, as ZeroGPU requires.
if os.getenv("SPACE_ID"):
    try:
        import spaces

        @spaces.GPU(duration=1)
        def _zerogpu_placeholder():
            """Never called; satisfies ZeroGPU's startup check."""
            return None
    except ImportError:
        pass

import gradio as gr
import pandas as pd

from chartrag import config
from chartrag.ingest import ensure_index
from chartrag.records import get_patients

PATIENTS = get_patients()
PATIENT_CHOICES = [(p.label, p.id) for p in PATIENTS.values()]
DEFAULT_PID = PATIENT_CHOICES[0][1]

# The index and the assistant are created on the first question, not at import time. Hosts such as
# Hugging Face ZeroGPU fork worker processes after import, and a database client opened before a fork
# can hang in the child. Lazy creation also lets the page load even if the API key is missing.
STARTUP_ERROR = ("OPENAI_API_KEY is not set" if config.LLM_MODE == "openai" and not os.getenv("OPENAI_API_KEY")
                 else None)
_ASSISTANT = None
_LOCK = threading.Lock()


def get_assistant():
    global _ASSISTANT
    with _LOCK:
        if _ASSISTANT is None:
            from chartrag.pipeline import ChartAssistant
            ensure_index()
            _ASSISTANT = ChartAssistant(patients=PATIENTS)
    return _ASSISTANT

esc = html.escape

# ------------------------------------------------------------------ styling
CSS = """
:root {
  --bg: #0D1117; --surface: #161B22; --surface-2: #1C2330; --rule: #2A3441; --text: #E6EDF3; --muted: #9AA7B8;
  --accent: #7C8CFF; --violet: #B794F6; --cyan: #4FD1E8; --coral: #FF7A7A; --coral-bg: rgba(255,122,122,0.12);
  --amber: #FFB547; --amber-bg: rgba(255,181,71,0.12); --green: #3DDC97; --green-bg: rgba(61,220,151,0.12);
}
body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }
.gradio-container { max-width: 1440px !important; width: 100% !important; margin: 0 auto !important;
  font-size: 16px !important; }
#masthead { padding: 10px 2px 16px; margin-bottom: 14px; display: flex; flex-wrap: wrap;
  align-items: baseline; gap: 6px 18px; border-bottom: 2px solid transparent;
  border-image: linear-gradient(90deg, var(--cyan), var(--accent), var(--violet)) 1; }
#masthead h1 { font-size: 1.9rem !important; font-weight: 700; margin: 0; letter-spacing: -0.02em;
  color: var(--text) !important; }
#masthead p { color: var(--muted); margin: 0; font-size: 1rem; }
#masthead .notice { margin-left: auto; font-size: 0.88rem; color: var(--amber); background: var(--amber-bg);
  padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(255,181,71,0.35); }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; color: var(--text); }
.card h2 { font-size: 1.45rem; margin: 0 0 4px; color: var(--text); }
.card .meta { color: var(--muted); font-size: 0.95rem; margin-bottom: 14px; line-height: 1.45; }
.card h3 { font-size: 1rem; font-weight: 600; margin: 16px 0 6px; }
.card h3.problems { color: var(--violet); } .card h3.meds { color: var(--cyan); }
.card ul { margin: 0; padding-left: 20px; }
.card li { margin: 4px 0; font-size: 0.98rem; line-height: 1.45; color: var(--text); }
.card .code { color: var(--muted); font-size: 0.85rem; margin-left: 4px; }
.band { border-left: 5px solid var(--coral); background: var(--coral-bg); padding: 11px 14px; border-radius: 6px; }
.band.none { border-left-color: var(--green); background: var(--green-bg); }
.band strong { color: var(--coral); font-size: 1rem; }
.band.none strong { color: var(--green); }
.band div { font-size: 0.98rem; color: var(--text); margin-top: 3px; }
.tag { display: inline-block; font-size: 0.78rem; padding: 0 7px; border-radius: 999px; margin-left: 8px;
  border: 1px solid currentColor; }
.tag.intol { color: var(--amber); } .tag.env { color: var(--muted); }
.status { padding: 11px 14px; border-radius: 8px; font-weight: 600; font-size: 1.02rem; margin-bottom: 10px;
  border: 1px solid transparent; }
.status.answered { background: var(--green-bg); color: var(--green); border-color: rgba(61,220,151,0.3); }
.status.not_documented, .status.blocked { background: var(--amber-bg); color: var(--amber); border-color: rgba(255,181,71,0.3); }
.status.out_of_scope { background: rgba(124,140,255,0.12); color: var(--accent); border-color: rgba(124,140,255,0.3); }
.status.error { background: var(--coral-bg); color: var(--coral); border-color: rgba(255,122,122,0.3); }
.verify { font-size: 0.95rem; margin: 0 0 12px; color: var(--muted); }
.verify.warn { color: var(--amber); }
.src { border: 1px solid var(--rule); border-left: 3px solid var(--cyan); border-radius: 8px; padding: 12px 14px;
  margin-bottom: 10px; background: var(--surface); }
.src .head { font-size: 0.98rem; color: var(--cyan); font-weight: 600; }
.src .sub { font-size: 0.85rem; color: var(--muted); margin: 2px 0 8px; }
.src .body { font-size: 0.95rem; line-height: 1.55; color: var(--text); white-space: pre-wrap; }
details { font-size: 0.9rem; color: var(--muted); margin-top: 10px; }
details summary { cursor: pointer; }
details table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin-top: 6px; }
details td, details th { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--rule); color: var(--text); }
.empty { color: var(--muted); font-size: 1rem; padding: 18px 4px; line-height: 1.6; }
.panel-title { font-size: 1.05rem; font-weight: 600; color: var(--text); }
.suggest button { font-size: 0.92rem !important; }
#evidence { max-height: 760px; overflow-y: auto; }
#chat .message-content, #chat .message-content * { font-size: 1rem !important; line-height: 1.6 !important; }
"""

# One dark palette for both the light and the dark variant of every Gradio variable, so the app looks
# the same whether the viewer's operating system is in light or dark mode.
_PALETTE = {
    "body_background_fill": "#0D1117", "background_fill_primary": "#0D1117", "background_fill_secondary": "#161B22",
    "block_background_fill": "#161B22", "block_border_color": "#2A3441", "border_color_primary": "#2A3441",
    "border_color_accent": "#7C8CFF", "border_color_accent_subdued": "#3B4470",
    "body_text_color": "#E6EDF3", "body_text_color_subdued": "#9AA7B8",
    "block_label_text_color": "#9AA7B8", "block_title_text_color": "#E6EDF3", "block_info_text_color": "#9AA7B8",
    "input_background_fill": "#1C2330", "input_background_fill_focus": "#1C2330", "input_border_color": "#2A3441",
    "input_border_color_focus": "#7C8CFF", "input_placeholder_color": "#6B7A8F",
    "button_primary_background_fill": "#7C8CFF", "button_primary_background_fill_hover": "#96A3FF",
    "button_primary_text_color": "#0D1117", "button_primary_border_color": "#7C8CFF",
    "button_secondary_background_fill": "#1C2330", "button_secondary_background_fill_hover": "#253044",
    "button_secondary_text_color": "#E6EDF3", "button_secondary_border_color": "#2A3441",
    "color_accent": "#7C8CFF", "color_accent_soft": "#232B4A", "link_text_color": "#4FD1E8",
    "link_text_color_hover": "#8BE4F2", "link_text_color_visited": "#4FD1E8", "link_text_color_active": "#4FD1E8",
    "table_even_background_fill": "#161B22", "table_odd_background_fill": "#1C2330", "table_border_color": "#2A3441",
    "code_background_fill": "#1C2330", "panel_background_fill": "#161B22", "panel_border_color": "#2A3441",
    "checkbox_background_color": "#1C2330", "checkbox_border_color": "#2A3441",
    "shadow_drop": "none", "shadow_drop_lg": "none",
}
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
    primary_hue=gr.themes.colors.indigo, neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md, text_size=gr.themes.sizes.text_lg,
).set(**{k: v for name, value in _PALETTE.items() for k, v in ((name, value), (name + "_dark", value))
        if k in inspect.signature(gr.themes.Base.set).parameters})

# Also switch Gradio into its dark mode before the page renders, so built-in components (charts,
# dropdown menus, code blocks) pick their dark styling regardless of the operating system setting.
FORCE_THEME = """<script>
(function () {
  try {
    var url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "dark") {
      url.searchParams.set("__theme", "dark");
      window.location.replace(url.toString());
    }
  } catch (e) {}
})();
</script>"""


# ------------------------------------------------------------ patient panel
def patient_card(pid: str) -> str:
    p = PATIENTS[pid]
    drug = [a for a in p.allergies if a["type"] in ("Allergy", "Intolerance")]
    other = [a for a in p.allergies if a["type"] == "Environmental"]
    if drug:
        items = "".join(
            f"<div>{esc(a['substance'])}: {esc(a['reaction'].lower())}"
            + (f"<span class='tag intol'>intolerance</span>" if a["type"] == "Intolerance" else "") + "</div>"
            for a in drug)
        band = f"<div class='band'><strong>Allergies and intolerances</strong>{items}"
    else:
        band = "<div class='band none'><strong>No known drug allergies</strong>"
    band += "".join(f"<div>{esc(a['substance'])}: {esc(a['reaction'].lower())}<span class='tag env'>environmental</span></div>"
                    for a in other) + "</div>"
    problems = "".join(f"<li>{esc(x['condition'])} <span class='code'>{esc(x['icd10'])}</span></li>" for x in p.active_problems)
    meds = "".join(f"<li>{esc(m['name'])} {esc(m['dose'])}, {esc(m['frequency'])}</li>" for m in p.active_medications)
    return f"""<div class='card'>
      <h2>{esc(p.name)}</h2>
      <div class='meta'>{p.age()}-year-old {esc(p.sex.lower())}, born {esc(p.dob)}. MRN {esc(p.mrn)}. PCP {esc(p.data.get('primary_care', ''))}.</div>
      {band}
      <h3 class='problems'>Active problems</h3><ul>{problems}</ul>
      <h3 class='meds'>Active medications</h3><ul>{meds}</ul>
    </div>"""


def lab_names(pid):
    return [lab["test"] for lab in PATIENTS[pid].labs]


def lab_frame(pid, test):
    lab = next((l for l in PATIENTS[pid].labs if l["test"] == test), None)
    if not lab:
        return pd.DataFrame({"date": [], "value": [], "series": []})
    return pd.DataFrame({"date": pd.to_datetime([r["date"] for r in lab["results"]]),
                         "value": [r["value"] for r in lab["results"]], "series": lab["test"]})


def lab_caption(pid, test):
    lab = next((l for l in PATIENTS[pid].labs if l["test"] == test), None)
    return f"{lab['test']} in {lab['unit']}. Reference range {lab['ref_range']}. LOINC {lab['loinc']}." if lab else ""


# --------------------------------------------------------- evidence panel
EMPTY_EVIDENCE = ("<div class='empty'>Sources appear here after you ask a question. Each answer links its "
                  "statements to numbered sources, [S1], [S2], so you can check them against the chart.</div>")

STATUS_TEXT = {
    "not_documented": "Not documented in this record",
    "blocked": "Answer withheld: citations failed validation",
    "error": "Error",
}
SCOPE_TEXT = {"clinical_advice": "Outside scope: treatment or dosing advice",
              "other_patient": "Outside scope: a different patient", "off_topic": "Outside scope",
              "empty": "Outside scope"}


def evidence_html(result) -> str:
    if result.status == "answered":
        label = f"Answered from {len(result.cited)} source{'s' if len(result.cited) != 1 else ''}"
    elif result.status == "out_of_scope":
        label = SCOPE_TEXT.get(result.reason, "Outside scope")
    else:
        label = STATUS_TEXT.get(result.status, result.status)
    out = [f"<div class='status {result.status}'>{esc(label)}</div>"]

    v = result.verification
    if result.status == "answered" and v:
        if v.get("error"):
            out.append("<p class='verify warn'>Automatic verification was unavailable for this answer. "
                       "Check the sources below.</p>")
        elif v["total"] and v["supported"] == v["total"]:
            out.append(f"<p class='verify'>All {v['total']} statements were verified against the cited sources.</p>")
        elif v["total"]:
            missing = "".join(f"<li>{esc(c)}</li>" for c in v["unsupported"])
            out.append(f"<div class='verify warn'>{v['total'] - v['supported']} of {v['total']} statements could "
                       f"not be verified against the sources:<ul>{missing}</ul></div>")

    if result.status == "answered":
        for n, cid in result.citation_map.items():
            c = next(x for x in result.cited if x.id == cid)
            out.append(f"<div class='src'><div class='head'>[S{n}] {esc(c.title)}</div>"
                       f"<div class='sub'>{esc(c.doc_type)}, {esc(c.date or 'undated')}, section: {esc(c.section)}</div>"
                       f"<div class='body'>{esc(c.body)}</div></div>")
    elif result.retrieved:
        out.append("<p class='verify'>Closest sections searched in this patient's record:</p>")
        for c in result.retrieved[:3]:
            out.append(f"<div class='src'><div class='head'>{esc(c.title)}</div>"
                       f"<div class='sub'>{esc(c.doc_type)}, {esc(c.date or 'undated')}, section: {esc(c.section)}</div>"
                       f"<div class='body'>{esc(c.body[:400])}</div></div>")

    if result.retrieved:
        rows = "".join(
            f"<tr><td>{esc(c.id.split('/', 1)[1])}</td><td>{'' if c.vector_score is None else f'{c.vector_score:.2f}'}</td>"
            f"<td>{'' if c.bm25_score is None else f'{c.bm25_score:.1f}'}</td></tr>" for c in result.retrieved)
        out.append(f"<details><summary>Retrieval details</summary><p>Search query: {esc(result.standalone_question)}</p>"
                   f"<table><tr><th>Chunk</th><th>Vector</th><th>BM25</th></tr>{rows}</table></details>")
    out.append(f"<details><summary>Request</summary>Request id {esc(result.request_id)}, {result.latency_s:.1f} s, "
               f"citation coverage {'' if result.citation_coverage is None else f'{result.citation_coverage:.0%}'}. "
               f"Model {esc(config.CHAT_MODEL)}, pipeline {esc(config.PIPELINE_VERSION)}.</details>")
    return "".join(out)


# ------------------------------------------------------------------ events
def select_patient(pid):
    tests = lab_names(pid)
    first = tests[0] if tests else None
    qs = (PATIENTS[pid].suggested_questions + [""] * 4)[:4]
    return ([patient_card(pid), gr.update(choices=tests, value=first), lab_frame(pid, first), lab_caption(pid, first),
             [], EMPTY_EVIDENCE] + [gr.update(value=q, visible=bool(q)) for q in qs])


def select_lab(pid, test):
    return lab_frame(pid, test), lab_caption(pid, test)


def add_question(question, history):
    question = (question or "").strip()
    if not question:
        return "", history
    return "", history + [{"role": "user", "content": question}]


def message_text(content) -> str:
    """Gradio 6 stores chat content as a list of parts; the pipeline wants plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content or "")


def answer(pid, history):
    if not history or history[-1]["role"] != "user":
        return history, gr.update()
    plain = [{"role": m["role"], "content": message_text(m["content"])} for m in history]
    question, prior = plain[-1]["content"], plain[:-1]
    try:
        assistant = get_assistant()
    except Exception as exc:
        msg = f"The assistant is not available: {type(exc).__name__}: {exc}"[:400]
        return history + [{"role": "assistant", "content": msg}], f"<div class='status error'>{esc(msg)}</div>"
    result = assistant.ask(pid, question, prior)
    text = result.answer
    v = result.verification
    if result.status == "answered" and v and v.get("total") and v["supported"] < v["total"]:
        text += f"\n\n_{v['total'] - v['supported']} statement(s) could not be verified. See the sources panel._"
    return history + [{"role": "assistant", "content": text}], evidence_html(result)


def read_doc(path, missing):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return missing


# ---------------------------------------------------------------------- UI
def build_ui():
    with gr.Blocks(title="Clinical Chart Assistant") as ui:
        gr.HTML("""<div id='masthead'><h1>Chart assistant</h1>
          <p>Ask about one patient's record. Every answer cites the notes it came from.</p>
          <span class='notice'>Synthetic patients only. Not for clinical use.</span></div>""")
        if STARTUP_ERROR:
            gr.HTML(f"<div class='status error'>Startup problem: {esc(STARTUP_ERROR)}. If this Space was just "
                    "created, add OPENAI_API_KEY under Settings, then restart.</div>")

        with gr.Tab("Chart review"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300):
                    patient = gr.Dropdown(PATIENT_CHOICES, value=DEFAULT_PID, label="Patient")
                    card = gr.HTML(patient_card(DEFAULT_PID))
                    lab = gr.Dropdown(lab_names(DEFAULT_PID), value=lab_names(DEFAULT_PID)[0], label="Lab trend")
                    plot = gr.LinePlot(lab_frame(DEFAULT_PID, lab_names(DEFAULT_PID)[0]), x="date", y="value",
                                       x_title="Date", y_title="Result", height=240, show_label=False,
                                       color="series", color_map={l["test"]: "#4FD1E8" for p in PATIENTS.values() for l in p.labs},
                                       colors_in_legend=[])
                    caption = gr.Markdown(lab_caption(DEFAULT_PID, lab_names(DEFAULT_PID)[0]))

                with gr.Column(scale=5, min_width=340):
                    chat = gr.Chatbot(height=560, show_label=False, elem_id="chat", placeholder=(
                        "Ask about this patient's history, results, medications or procedures. "
                        "Switching patients starts a new conversation."))
                    with gr.Row():
                        suggestions = [gr.Button(q, size="sm", elem_classes="suggest")
                                       for q in (PATIENTS[DEFAULT_PID].suggested_questions + [""] * 4)[:4]]
                    with gr.Row(equal_height=True):
                        box = gr.Textbox(placeholder="Ask a question about this patient", show_label=False,
                                         scale=8, lines=1, max_lines=4, autofocus=True)
                        ask = gr.Button("Ask", variant="primary", scale=1, min_width=80)
                    clear = gr.Button("Start new conversation", size="sm")

                with gr.Column(scale=4, min_width=300):
                    gr.HTML("<div class='panel-title'>Sources for the latest answer</div>")
                    evidence = gr.HTML(EMPTY_EVIDENCE, elem_id="evidence")

        with gr.Tab("Evaluation"):
            gr.Markdown(read_doc(config.REPORTS_DIR / "eval_report.md",
                                 "No evaluation report yet. Run `python -m eval.run_eval` and commit `reports/`."))
        with gr.Tab("Model card"):
            gr.Markdown(read_doc(config.DOCS_DIR / "MODEL_CARD.md", "Model card not found."))
        with gr.Tab("Risk analysis"):
            gr.Markdown(read_doc(config.DOCS_DIR / "RISK_ANALYSIS.md", "Risk analysis not found."))

        patient.change(select_patient, [patient], [card, lab, plot, caption, chat, evidence, *suggestions])
        lab.change(select_lab, [patient, lab], [plot, caption])
        for trigger in (box.submit, ask.click):
            trigger(add_question, [box, chat], [box, chat]).then(answer, [patient, chat], [chat, evidence])
        for b in suggestions:
            b.click(add_question, [b, chat], [box, chat]).then(answer, [patient, chat], [chat, evidence])
        clear.click(lambda: ([], EMPTY_EVIDENCE), outputs=[chat, evidence])
    return ui


# Hugging Face's Gradio runner looks for a module-level Blocks object named `demo`.
demo = build_ui()

if __name__ == "__main__":
    auth = (os.getenv("APP_USER"), os.getenv("APP_PASSWORD")) if os.getenv("APP_USER") and os.getenv("APP_PASSWORD") else None
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        auth=auth, theme=THEME, css=CSS, head=FORCE_THEME,
        # Serve the normal client-rendered app. Gradio's experimental server-side rendering adds a
        # Node.js layer on Spaces that can leave the page unstyled if the browser app fails to load.
        ssr_mode=False,
    )