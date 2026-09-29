"""
Clinical Chart Assistant: Gradio front end.

Deterministic data (allergies, problems, medications, lab trends) is rendered straight from the
structured record, never through the language model. The model is used only for free-text
questions, and every answer is shown next to the exact sources it cites.
"""
import html
import os

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

ASSISTANT, STARTUP_ERROR = None, None
try:
    ensure_index()
    from chartrag.pipeline import ChartAssistant
    ASSISTANT = ChartAssistant(patients=PATIENTS)
except Exception as exc:  # show the problem in the UI instead of crashing the Space
    STARTUP_ERROR = f"{type(exc).__name__}: {exc}"

esc = html.escape

# ------------------------------------------------------------------ styling
CSS = """
:root {
  --paper: #FAFBFC; --ink: #16283A; --blue: #1F5A8C; --rule: #D7DEE6; --muted: #5B6B7B;
  --red: #B42318; --red-bg: #FDECEA; --amber: #8A5A00; --amber-bg: #FFF4DB; --green: #1E7A4C; --green-bg: #E7F5EE;
}
.gradio-container { max-width: 1440px !important; background: var(--paper) !important; }
#masthead { padding: 8px 2px 14px; border-bottom: 2px solid var(--ink); margin-bottom: 12px;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px; }
#masthead h1 { font-size: 1.65rem; font-weight: 700; color: var(--ink); margin: 0; letter-spacing: -0.01em; }
#masthead p { color: var(--muted); margin: 0; font-size: 0.95rem; }
#masthead .notice { margin-left: auto; font-size: 0.82rem; color: var(--amber); background: var(--amber-bg);
  padding: 3px 10px; border-radius: 4px; }
.card { background: #fff; border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; color: var(--ink); }
.card h2 { font-size: 1.3rem; margin: 0 0 2px; color: var(--ink); }
.card .meta { color: var(--muted); font-size: 0.86rem; margin-bottom: 12px; }
.card h3 { font-size: 0.9rem; font-weight: 600; color: var(--blue); margin: 14px 0 6px; }
.card ul { margin: 0; padding-left: 18px; }
.card li { margin: 2px 0; font-size: 0.9rem; line-height: 1.4; }
.card .code { color: var(--muted); font-size: 0.8rem; }
.band { border-left: 6px solid var(--red); background: var(--red-bg); padding: 10px 12px; border-radius: 4px; }
.band.none { border-left-color: var(--muted); background: #EEF2F6; }
.band strong { color: var(--red); font-size: 0.92rem; }
.band.none strong { color: var(--ink); }
.band div { font-size: 0.9rem; color: var(--ink); margin-top: 2px; }
.tag { display: inline-block; font-size: 0.75rem; padding: 0 6px; border-radius: 3px; margin-left: 6px;
  border: 1px solid currentColor; }
.tag.intol { color: var(--amber); } .tag.env { color: var(--muted); }
.status { padding: 10px 12px; border-radius: 4px; font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; }
.status.answered { background: var(--green-bg); color: var(--green); }
.status.not_documented, .status.blocked { background: var(--amber-bg); color: var(--amber); }
.status.out_of_scope { background: #EEF2F6; color: var(--ink); }
.status.error { background: var(--red-bg); color: var(--red); }
.verify { font-size: 0.88rem; margin: 0 0 12px; color: var(--ink); }
.verify.warn { color: var(--amber); }
.src { border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; margin-bottom: 10px; background: #fff; }
.src .head { font-size: 0.86rem; color: var(--blue); font-weight: 600; }
.src .sub { font-size: 0.78rem; color: var(--muted); margin-bottom: 6px; }
.src .body { font-size: 0.86rem; line-height: 1.5; color: var(--ink); white-space: pre-wrap; }
details { font-size: 0.82rem; color: var(--muted); margin-top: 8px; }
details table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
details td, details th { text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--rule); }
.empty { color: var(--muted); font-size: 0.92rem; padding: 18px 4px; line-height: 1.5; }
.suggest button { font-size: 0.84rem !important; text-align: left !important; justify-content: flex-start !important; }
#evidence { max-height: 720px; overflow-y: auto; }
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_sm,
).set(**{k: v for pair in [
    ("body_background_fill", "#FAFBFC"), ("body_text_color", "#16283A"), ("block_background_fill", "#FFFFFF"),
    ("block_border_color", "#D7DEE6"), ("border_color_primary", "#D7DEE6"), ("input_background_fill", "#FFFFFF"),
    ("button_primary_background_fill", "#1F5A8C"), ("button_primary_background_fill_hover", "#174873"),
    ("button_primary_text_color", "#FFFFFF"), ("block_label_text_color", "#5B6B7B"),
    ("body_text_color_subdued", "#5B6B7B"), ("color_accent_soft", "#E6EEF6"),
] for k, v in ((pair[0], pair[1]), (pair[0] + "_dark", pair[1]))})  # same palette in OS dark mode


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
      <h3>Active problems</h3><ul>{problems}</ul>
      <h3>Active medications</h3><ul>{meds}</ul>
    </div>"""


def lab_names(pid):
    return [lab["test"] for lab in PATIENTS[pid].labs]


def lab_frame(pid, test):
    lab = next((l for l in PATIENTS[pid].labs if l["test"] == test), None)
    if not lab:
        return pd.DataFrame({"date": [], "value": []})
    return pd.DataFrame({"date": pd.to_datetime([r["date"] for r in lab["results"]]),
                         "value": [r["value"] for r in lab["results"]]})


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
    if ASSISTANT is None:
        msg = f"The assistant is not available: {STARTUP_ERROR}"
        return history + [{"role": "assistant", "content": msg}], f"<div class='status error'>{esc(msg)}</div>"
    result = ASSISTANT.ask(pid, question, prior)
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
                                       x_title="Date", y_title="Result", height=220, show_label=False)
                    caption = gr.Markdown(lab_caption(DEFAULT_PID, lab_names(DEFAULT_PID)[0]))

                with gr.Column(scale=5, min_width=340):
                    chat = gr.Chatbot(height=520, show_label=False, placeholder=(
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
                    gr.Markdown("**Sources for the latest answer**")
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


if __name__ == "__main__":
    auth = (os.getenv("APP_USER"), os.getenv("APP_PASSWORD")) if os.getenv("APP_USER") and os.getenv("APP_PASSWORD") else None
    build_ui().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        auth=auth, theme=THEME, css=CSS,
    )
