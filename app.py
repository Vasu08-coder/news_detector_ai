import os
import re
from uuid import uuid4
from io import BytesIO

import requests
from flask import Flask, jsonify, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from agent import image_agent, meta_agent, text_agent, url_agent, verifier_agent, video_agent
from history import get_last_five_results, init_db, save_result
from url_detector import fetch_url_text

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
init_db()


def save_uploaded_file(file_storage):
    original_name = secure_filename(file_storage.filename or "")
    if not original_name:
        raise ValueError("Invalid file name.")
    unique_name = f"{uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file_storage.save(filepath)
    return unique_name, filepath


def extract_claim_highlights(text):
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    suspicious_keywords = {
        "breaking", "shocking", "viral", "rumor", "secret", "exclusive",
        "conspiracy", "aliens", "miracle", "banned", "exposed", "fake",
    }
    highlights = []

    for sentence in sentences[:8]:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            continue

        lowered = clean_sentence.lower()
        matched = [keyword for keyword in suspicious_keywords if keyword in lowered]
        if matched:
            status = "warning"
            reason = f"Contains attention-heavy language: {', '.join(matched[:3])}."
        elif len(clean_sentence.split()) > 16:
            status = "info"
            reason = "Long claim that may need source verification."
        else:
            status = "normal"
            reason = "Looks like a regular claim with no immediate warning keywords."

        highlights.append(
            {
                "sentence": clean_sentence,
                "status": status,
                "reason": reason,
            }
        )

    return highlights


def build_pdf_report(analysis_context):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    lines = [
        "Multi-Modal Fake News Detector Report",
        "",
        f"Final Result: {analysis_context.get('result', 'UNKNOWN')}",
        f"Confidence: {analysis_context.get('confidence', 0)}%",
        "",
        "Final Explanation:",
        analysis_context.get("explanation", ""),
        "",
    ]

    for label, key in [
        ("Text Agent", "text_result"),
        ("URL Agent", "url_result"),
        ("Image Agent", "image_result"),
        ("Video Agent", "video_result"),
        ("Verifier Agent", "verifier_result"),
    ]:
        result = analysis_context.get(key)
        if result:
            lines.append(f"{label}: {result.get('label', 'UNKNOWN')} ({result.get('confidence', 0)}%)")
            lines.append(result.get("explanation", ""))
            lines.append("")

    for line in lines:
        chunks = [line[index:index + 95] for index in range(0, len(line), 95)] or [""]
        for chunk in chunks:
            pdf.drawString(40, y, chunk)
            y -= 18
            if y < 50:
                pdf.showPage()
                y = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer


def call_llm_response(message, analysis_context):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not analysis_context:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    agent_summaries = []
    for key, label in [
        ("text_result", "Text"),
        ("url_result", "URL"),
        ("image_result", "Image"),
        ("video_result", "Video"),
        ("verifier_result", "Verifier"),
    ]:
        value = analysis_context.get(key)
        if value:
            agent_summaries.append(
                f"{label}: {value.get('label', 'UNKNOWN')} at {value.get('confidence', 0)}% confidence. {value.get('explanation', '')}"
            )

    prompt = (
        "You are an assistant for a fake news detection system. "
        "Answer clearly, briefly, and practically using the analysis context below.\n\n"
        f"Final result: {analysis_context.get('result', 'UNKNOWN')}\n"
        f"Final confidence: {analysis_context.get('confidence', 0)}%\n"
        f"Final explanation: {analysis_context.get('explanation', '')}\n"
        f"Agent details: {' | '.join(agent_summaries)}\n\n"
        f"User question: {message}"
    )

    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You explain fake-news detection results for users."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
            },
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def build_analysis_context(
    final_result,
    confidence,
    explanation,
    text_result=None,
    url_result=None,
    image_result=None,
    video_result=None,
    verifier_result=None,
    claim_highlights=None,
):
    return {
        "result": final_result,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "text_result": text_result,
        "url_result": url_result,
        "image_result": image_result,
        "video_result": video_result,
        "verifier_result": verifier_result,
        "claim_highlights": claim_highlights or [],
    }


def build_chat_response(message, analysis_context):
    if not analysis_context:
        return "Run an analysis first, then I can explain the result, compare agent signals, and suggest what to verify next."

    llm_reply = call_llm_response(message, analysis_context)
    if llm_reply:
        return llm_reply

    message_lower = (message or "").strip().lower()
    final_result = analysis_context.get("result", "UNKNOWN")
    confidence = analysis_context.get("confidence", 0)
    text_result = analysis_context.get("text_result")
    url_result = analysis_context.get("url_result")
    image_result = analysis_context.get("image_result")
    video_result = analysis_context.get("video_result")
    verifier_result = analysis_context.get("verifier_result")

    active_agents = []
    for name, result in [
        ("text", text_result),
        ("url", url_result),
        ("image", image_result),
        ("video", video_result),
    ]:
        if result and result.get("label") not in {None, "UNKNOWN"}:
            active_agents.append((name, result))

    strongest_agent = None
    if active_agents:
        strongest_agent = max(
            active_agents,
            key=lambda item: float(item[1].get("confidence", 0) or 0),
        )

    if any(keyword in message_lower for keyword in ["why", "reason", "explain"]):
        lead = f"The final result is {final_result} with {confidence}% confidence."
        if strongest_agent:
            lead += f" The strongest signal came from the {strongest_agent[0]} agent."
        return f"{lead} {analysis_context.get('explanation', '')}".strip()

    if any(keyword in message_lower for keyword in ["summary", "summarize", "short"]):
        summary_parts = [f"Result: {final_result}", f"Confidence: {confidence}%"]
        if strongest_agent:
            summary_parts.append(
                f"Strongest agent: {strongest_agent[0].title()} ({strongest_agent[1].get('label', 'UNKNOWN')})"
            )
        return ". ".join(summary_parts) + "."

    if any(keyword in message_lower for keyword in ["agent", "agents", "signal", "signals"]):
        agent_lines = []
        for name, result in active_agents:
            agent_lines.append(
                f"{name.title()} agent: {result.get('label', 'UNKNOWN')} with {result.get('confidence', 0)}% confidence."
            )
        if verifier_result and verifier_result.get("label") not in {None, "UNKNOWN"}:
            agent_lines.append(
                f"Verifier agent: {verifier_result.get('label', 'UNKNOWN')} with {verifier_result.get('confidence', 0)}% confidence."
            )
        return " ".join(agent_lines) if agent_lines else "No active agent details are available yet."

    if any(keyword in message_lower for keyword in ["verify", "trust", "credible", "source", "next"]):
        suggestions = []
        if not url_result:
            suggestions.append("add a source URL")
        if not image_result:
            suggestions.append("upload a related image")
        if not video_result:
            suggestions.append("upload a supporting video")
        if not suggestions:
            suggestions.append("compare the claim with reliable news sources")
        return (
            f"To verify this further, {', '.join(suggestions)}. "
            "You can also compare the headline with trusted publishers and check whether the visuals match the text claim."
        )

    return (
        f"I can help explain this analysis. The current result is {final_result} with {confidence}% confidence. "
        "Ask me why it was classified that way, which agent influenced the decision most, or what to verify next."
    )


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        result=None,
        confidence=None,
        explanation=None,
        text_result=None,
        url_result=None,
        image_result=None,
        video_result=None,
        verifier_result=None,
        claim_highlights=[],
        active_inputs=[],
        history=get_last_five_results(),
        chat_history=session.get("chat_history", []),
        text="",
        url_value="",
        image_url=None,
        video_url=None,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        text = request.form.get("text", "").strip()
        url_value = request.form.get("url", "").strip()
        if not text:
            text = request.form.get("news", "").strip()
        image = request.files.get("image")
        video = request.files.get("video")
        image_url = None
        video_url = None
        text_result = None
        url_result = None
        image_result = None
        video_result = None
        verifier_result = None
        claim_highlights = []
        extracted_url_text = None

        if image and image.filename != "":
            filename, filepath = save_uploaded_file(image)

            image_result = {
                **image_agent(filepath),
            }
            image_url = url_for("static", filename="uploads/" + filename)

        if video and video.filename != "":
            filename, filepath = save_uploaded_file(video)

            video_result = {
                **video_agent(filepath),
            }
            video_url = url_for("static", filename="uploads/" + filename)

        if url_value:
            extracted_url_text = fetch_url_text(url_value)
            url_result = {
                **url_agent(url_value, extracted_text=extracted_url_text),
            }

        if text:
            text_result = {
                **text_agent(text),
            }
            claim_highlights = extract_claim_highlights(text)
        elif extracted_url_text:
            claim_highlights = extract_claim_highlights(extracted_url_text)

        active_inputs = []
        if text_result:
            active_inputs.append("Text")
        if url_result:
            active_inputs.append("URL")
        if image_result:
            active_inputs.append("Image")
        if video_result:
            active_inputs.append("Video")

        verifier_result = verifier_agent(
            text_result=text_result,
            url_result=url_result,
            image_result=image_result,
            video_result=video_result,
        )

        if text_result is None and image_result is None and video_result is None and url_result is None:
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                explanation="Please enter some news text, a URL, or upload an image or video.",
                text_result=None,
                url_result=None,
                image_result=None,
                video_result=None,
                verifier_result=None,
                claim_highlights=[],
                active_inputs=[],
                history=get_last_five_results(),
                chat_history=session.get("chat_history", []),
                text="",
                url_value=url_value,
                image_url=None,
                video_url=None
            )

        final_result, confidence, explanation = meta_agent(
            text_result,
            image_result,
            video_result,
            url_result=url_result,
            verifier_result=verifier_result,
        )
        analysis_context = build_analysis_context(
            final_result,
            confidence,
            explanation,
            text_result=text_result,
            url_result=url_result,
            image_result=image_result,
            video_result=video_result,
            verifier_result=verifier_result,
            claim_highlights=claim_highlights,
        )
        session["latest_analysis"] = analysis_context
        session["chat_history"] = [
            {
                "role": "assistant",
                "text": (
                    f"Analysis complete. The final result is {final_result} with {round(confidence, 2)}% confidence. "
                    "Ask me why the system reached this decision or which agent influenced it most."
                ),
            }
        ]
        save_result(final_result, round(confidence, 2))

        return render_template(
            "index.html",
            result=final_result,
            confidence=round(confidence, 2),
            explanation=explanation,
            text_result=text_result,
            url_result=url_result,
            image_result=image_result,
            video_result=video_result,
            verifier_result=verifier_result,
            claim_highlights=claim_highlights,
            active_inputs=active_inputs,
            history=get_last_five_results(),
            chat_history=session.get("chat_history", []),
            text=text,
            url_value=url_value,
            image_url=image_url,
            video_url=video_url
        )
    except (ValueError, ConnectionError) as error:
        return render_template(
            "index.html",
            result=None,
            confidence=None,
            explanation=str(error),
            text_result=None,
            url_result=None,
            image_result=None,
            video_result=None,
            verifier_result=None,
            claim_highlights=[],
            active_inputs=[],
            history=get_last_five_results(),
            chat_history=session.get("chat_history", []),
            text=request.form.get("text", "").strip() or request.form.get("news", "").strip(),
            url_value=request.form.get("url", "").strip(),
            image_url=None,
            video_url=None
        )
    except Exception as error:
        return render_template(
            "index.html",
            result=None,
            confidence=None,
            explanation=f"Something went wrong: {error}",
            text_result=None,
            url_result=None,
            image_result=None,
            video_result=None,
            verifier_result=None,
            claim_highlights=[],
            active_inputs=[],
            history=get_last_five_results(),
            chat_history=session.get("chat_history", []),
            text=request.form.get("text", "").strip() or request.form.get("news", "").strip(),
            url_value=request.form.get("url", "").strip(),
            image_url=None,
            video_url=None
        )


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Ask a question about the latest analysis and I will help explain it."})

    analysis_context = session.get("latest_analysis")
    reply = build_chat_response(message, analysis_context)

    history = session.get("chat_history", [])
    history.append({"role": "user", "text": message})
    history.append({"role": "assistant", "text": reply})
    session["chat_history"] = history[-10:]

    return jsonify({"reply": reply, "history": session["chat_history"]})


@app.route("/export-report", methods=["GET"])
def export_report():
    analysis_context = session.get("latest_analysis")
    if not analysis_context:
        return jsonify({"error": "No analysis available to export."}), 400

    try:
        pdf_buffer = build_pdf_report(analysis_context)
    except ImportError:
        return jsonify({"error": "PDF export requires reportlab. Install dependencies from requirements.txt first."}), 500

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="fake_news_report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
