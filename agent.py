from image_detector import detect_image
from text_detector import detect_text
from url_detector import fetch_url_text
from verifier import verify_news
from video_detector import detect_video


class MetaAgentResult(dict):
    def __iter__(self):
        yield self["result"]
        yield self["confidence"]
        yield self["explanation"]


def _safe_probability(confidence):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.5

    if value > 1:
        value = value / 100.0

    return max(0.0, min(value, 1.0))


def _normalize_text_detection(raw_result):
    if isinstance(raw_result, dict):
        label = str(raw_result.get("label", "UNKNOWN")).upper()
        confidence = raw_result.get("confidence", 0)
        explanation = raw_result.get("explanation", "Prediction based on trained ML model.")
    elif isinstance(raw_result, (tuple, list)) and len(raw_result) >= 3:
        label = str(raw_result[0]).upper()
        confidence = raw_result[1]
        explanation = raw_result[2]
    elif isinstance(raw_result, (tuple, list)) and len(raw_result) >= 2:
        label = str(raw_result[0]).upper()
        confidence = raw_result[1]
        explanation = "Prediction based on trained ML model."
    else:
        label = "UNKNOWN"
        confidence = 0
        explanation = "Prediction based on trained ML model."

    probability = _safe_probability(confidence)
    return label, probability, str(explanation)


def text_agent(text):
    if not text:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": "Text not provided.",
        }

    raw_result = detect_text(text)
    label, probability, explanation = _normalize_text_detection(raw_result)
    lowered_text = str(text).lower()
    common_keywords = {
        "government", "president", "minister", "election", "court", "police",
        "report", "official", "statement", "international", "india", "world",
        "economy", "health", "education", "technology", "market", "sports",
        "policy", "agency", "cabinet", "parliament",
    }
    suspicious_keywords = {
        "shocking", "viral", "rumor", "secret", "conspiracy", "aliens",
        "miracle", "exposed", "banned", "hoax", "clickbait", "unbelievable",
    }
    common_count = sum(1 for word in common_keywords if word in lowered_text)
    suspicious_count = sum(1 for word in suspicious_keywords if word in lowered_text)

    if label == "REAL":
        score = probability
    elif label == "FAKE":
        score = 1.0 - probability
    else:
        score = 0.5
        label = "UNKNOWN"

    try:
        verification_score, verification_explanation = verify_news(text)
        verification_score = max(0.0, min(float(verification_score), 1.0))
    except Exception:
        verification_score = 0.5
        verification_explanation = "Verification support was not available."

    # Let the trained model lead, and let verification only nudge borderline cases.
    verification_adjustment = (verification_score - 0.5) * 0.12
    final_score = score + verification_adjustment

    # If the wording looks like normal news and verification supports it,
    # stop borderline FAKE bias from dominating the final text decision.
    if label == "FAKE" and verification_score >= 0.65 and suspicious_count == 0 and common_count >= 1:
        final_score = max(final_score, 0.58)

    final_score = max(0.0, min(round(final_score, 2), 1.0))

    if final_score > 0.5:
        final_label = "REAL"
    elif final_score < 0.5:
        final_label = "FAKE"
    else:
        final_label = label

    final_confidence = round(max(50.0, min(95.0, (0.85 * probability + 0.15 * verification_score) * 100)), 2)
    final_explanation = (
        f"The model predicted {final_label} with {final_confidence}% confidence. "
        f"{explanation} {verification_explanation}"
    ).strip()

    return {
        "label": final_label,
        "score": final_score,
        "confidence": final_confidence,
        "explanation": final_explanation,
    }


def url_agent(url, extracted_text=None):
    if not url:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": "URL not provided.",
        }

    try:
        if extracted_text is None:
            extracted_text = fetch_url_text(url)
        text_result = text_agent(extracted_text)

        trusted_domains = ("bbc", "reuters", "apnews", "thehindu", "indianexpress", "ndtv", "nytimes", "theguardian")
        caution_domains = ("blogspot", "wordpress", "substack", "telegram", "rumble")
        lower_url = url.lower()
        if any(domain in lower_url for domain in trusted_domains):
            domain_bonus = 0.08
            domain_note = "The source domain looks relatively credible."
            credibility_score = 82
        elif any(domain in lower_url for domain in caution_domains):
            domain_bonus = -0.06
            domain_note = "The source domain looks less established, so credibility is lower."
            credibility_score = 42
        else:
            domain_bonus = 0.0
            domain_note = "The source domain is neutral because it is not in the trusted or caution lists."
            credibility_score = 60

        base_score = text_result["score"] if text_result["score"] is not None else 0.5
        score = max(0.0, min(round(base_score + domain_bonus, 2), 1.0))

        if score > 0.5:
            label = "REAL"
        elif score < 0.5:
            label = "FAKE"
        else:
            label = text_result["label"]

        confidence = min(95, round(text_result["confidence"] + (domain_bonus * 100), 2))
        explanation = (
            f"URL content was extracted and checked. Domain credibility score is {credibility_score}/100. "
            f"{domain_note} {text_result['explanation']}"
        )

        return {
            "label": label,
            "score": score,
            "confidence": confidence,
            "explanation": explanation,
        }
    except Exception as error:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": f"URL analysis failed: {error}",
        }


def image_agent(image_path):
    if not image_path:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": "Image not provided.",
        }

    try:
        label, confidence, raw_explanation = detect_image(image_path)
        probability = _safe_probability(confidence)
        detail = str(raw_explanation).lower()
        model_driven = "trained visual model" in detail

        if str(label).upper() == "REAL":
            score = 0.55 + min((probability - 0.5) * 0.4, 0.25) if model_driven else 0.5 + min((probability - 0.5) * 0.2, 0.1)
            final_label = "REAL"
        elif str(label).upper() == "FAKE":
            score = 0.45 - min((probability - 0.5) * 0.4, 0.25) if model_driven else 0.5 - min((probability - 0.5) * 0.2, 0.1)
            final_label = "FAKE"
        else:
            score = 0.5
            final_label = "UNKNOWN"

        notes = []
        if "resolution" in detail or "quality" in detail:
            if "low" in detail or "weak" in detail:
                notes.append("the image quality looks weak")
            else:
                notes.append("the image resolution looks acceptable")
        if "metadata" in detail:
            if "missing" in detail:
                notes.append("metadata is missing")
            else:
                notes.append("metadata is present")
        if "noise" in detail or "variance" in detail or "compression" in detail:
            if "suspicious" in detail or "artifact" in detail or "missing" in detail:
                notes.append("the visual pattern looks slightly suspicious")
            else:
                notes.append("the visual pattern looks natural")

        if not notes:
            notes.append("basic quality checks were completed")

        if final_label == "FAKE":
            prefix = "The visual agent found suspicious evidence because " if model_driven else "Image looks suspicious because "
            explanation = prefix + ", ".join(notes) + "."
        elif final_label == "REAL":
            prefix = "The visual agent found supportive evidence because " if model_driven else "Image looks more trustworthy because "
            explanation = prefix + ", ".join(notes) + "."
        else:
            explanation = "Image analysis was inconclusive."

        score = max(0.3, min(round(score, 2), 0.7)) if model_driven else max(0.4, min(round(score, 2), 0.6))
        final_confidence = round(probability * 100, 2)

        return {
            "label": final_label,
            "score": score,
            "confidence": final_confidence,
            "explanation": explanation,
        }
    except Exception as error:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": f"Image analysis failed: {error}",
        }


def video_agent(video_path):
    if not video_path:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": "Video not provided.",
        }

    try:
        label, confidence, raw_explanation = detect_video(video_path)
        probability = _safe_probability(confidence)
        detail = str(raw_explanation).lower()
        model_driven = "trained video model" in detail

        if str(label).upper() == "REAL":
            score = 0.55 + min((probability - 0.5) * 0.4, 0.25) if model_driven else 0.5 + min((probability - 0.5) * 0.2, 0.1)
            final_label = "REAL"
        elif str(label).upper() == "FAKE":
            score = 0.45 - min((probability - 0.5) * 0.4, 0.25) if model_driven else 0.5 - min((probability - 0.5) * 0.2, 0.1)
            final_label = "FAKE"
        else:
            score = 0.5
            final_label = "UNKNOWN"

        notes = []
        if "frames" in detail:
            notes.append("multiple frames were reviewed")
        if "blur" in detail or "blurry" in detail or "sharp" in detail:
            if "sharp" in detail:
                notes.append("frame clarity looks normal")
            else:
                notes.append("some frames show blur")
        if "brightness" in detail:
            if "natural" in detail or "normal" in detail or "stable" in detail:
                notes.append("brightness changes look stable")
            else:
                notes.append("brightness consistency looks unusual")

        if not notes:
            notes.append("basic frame checks were completed")

        if final_label == "FAKE":
            prefix = "The video agent found suspicious evidence because " if model_driven else "Video looks suspicious because "
            explanation = prefix + ", ".join(notes) + "."
        elif final_label == "REAL":
            prefix = "The video agent found supportive evidence because " if model_driven else "Video looks more trustworthy because "
            explanation = prefix + ", ".join(notes) + "."
        else:
            explanation = "Video analysis was inconclusive."

        score = max(0.3, min(round(score, 2), 0.7)) if model_driven else max(0.4, min(round(score, 2), 0.6))
        final_confidence = round(probability * 100, 2)

        return {
            "label": final_label,
            "score": score,
            "confidence": final_confidence,
            "explanation": explanation,
        }
    except Exception as error:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": f"Video analysis failed: {error}",
        }


def verifier_agent(text_result=None, url_result=None, image_result=None, video_result=None):
    available = [
        ("text", text_result),
        ("url", url_result),
        ("image", image_result),
        ("video", video_result),
    ]
    active = [(name, result) for name, result in available if result is not None and result.get("score") is not None]

    if not active:
        return {
            "label": "UNKNOWN",
            "score": None,
            "confidence": 0,
            "explanation": "No agent outputs available for verification.",
        }

    real_votes = 0
    fake_votes = 0
    labels = []

    for name, result in active:
        label = result.get("label", "UNKNOWN")
        labels.append(f"{name}:{label}")
        if label == "REAL":
            real_votes += 1
        elif label == "FAKE":
            fake_votes += 1

    if real_votes > fake_votes:
        label = "REAL"
        score = 0.65
    elif fake_votes > real_votes:
        label = "FAKE"
        score = 0.35
    else:
        label = "UNKNOWN"
        score = 0.5

    confidence = round((max(real_votes, fake_votes) / max(len(active), 1)) * 100, 2)
    explanation = "Verifier agent compared the active agents: " + ", ".join(labels) + "."

    if real_votes and fake_votes:
        explanation += " Some disagreement was detected between agents."
    else:
        explanation += " The active agents were mostly aligned."

    return {
        "label": label,
        "score": score,
        "confidence": confidence,
        "explanation": explanation,
    }


def meta_agent(text_result=None, image_result=None, video_result=None, url_result=None, verifier_result=None):
    sources = [
        ("Text", text_result, 0.6),
        ("URL", url_result, 0.2),
        ("Image", image_result, 0.25),
        ("Video", video_result, 0.15),
    ]

    active_sources = []
    for name, result, weight in sources:
        if result is not None and result.get("score") is not None:
            active_sources.append((name, result, weight))

    if not active_sources:
        return MetaAgentResult(
            {
                "result": "UNKNOWN",
                "confidence": 0,
                "explanation": "No input provided.",
            }
        )

    total_weight = sum(weight for _, _, weight in active_sources)
    weighted_average = 0.0
    explanation_parts = []
    influence_scores = []
    active_labels = []
    confidence_sum = 0.0

    for name, result, weight in active_sources:
        numeric_score = max(0.0, min(float(result["score"]), 1.0))
        normalized_weight = weight / total_weight
        weighted_average += numeric_score * normalized_weight
        explanation_parts.append(f"{name}: {result.get('explanation', 'Not analyzed')}")
        active_labels.append((name, result.get("label", "UNKNOWN"), numeric_score))
        confidence_sum += float(result.get("confidence", 0)) * normalized_weight
        influence_scores.append(
            {
                "name": name,
                "weighted_distance": abs(numeric_score - 0.5) * normalized_weight,
                "explanation": result.get("explanation", "Not analyzed"),
            }
        )

    if weighted_average > 0.5:
        final_result = "REAL"
    elif weighted_average < 0.5:
        final_result = "FAKE"
    else:
        majority_real = sum(1 for _, label, _ in active_labels if label == "REAL")
        majority_fake = sum(1 for _, label, _ in active_labels if label == "FAKE")
        if majority_real > majority_fake:
            final_result = "REAL"
        elif majority_fake > majority_real:
            final_result = "FAKE"
        else:
            final_result = "REAL"

    final_confidence = confidence_sum if confidence_sum > 0 else (55 + abs(weighted_average - 0.5) * 50)

    conflict_messages = []
    if (
        text_result is not None
        and text_result.get("score") is not None
        and image_result is not None
        and image_result.get("score") is not None
    ):
        text_score = max(0.0, min(float(text_result["score"]), 1.0))
        image_score = max(0.0, min(float(image_result["score"]), 1.0))

        if text_score > 0.7 and image_score < 0.4:
            final_confidence -= 10
            conflict_messages.append("Conflict between text and image analysis")

    if (
        text_result is not None
        and image_result is not None
        and text_result.get("label") == "FAKE"
        and image_result.get("label") == "REAL"
    ):
        final_confidence -= 5
        conflict_messages.append("Conflict between text and visual analysis")

    if (
        text_result is not None
        and text_result.get("score") is not None
        and video_result is not None
        and video_result.get("score") is not None
    ):
        text_score = max(0.0, min(float(text_result["score"]), 1.0))
        video_score = max(0.0, min(float(video_result["score"]), 1.0))

        if (text_score > 0.6 and video_score < 0.4) or (text_score < 0.4 and video_score > 0.6):
            final_confidence -= 8
            conflict_messages.append("Conflict between text and video analysis")

    if verifier_result is not None and verifier_result.get("explanation"):
        explanation_parts.append("Verifier: " + verifier_result["explanation"])
        if "disagreement" in verifier_result["explanation"].lower():
            final_confidence -= 5

    support_count = 0
    disagreement_count = 0
    for _, label, _ in active_labels:
        if label == final_result:
            support_count += 1
        elif label in {"REAL", "FAKE"} and label != final_result:
            disagreement_count += 1

    if support_count >= 2:
        final_confidence += min(6, support_count * 2)

    if disagreement_count:
        final_confidence -= min(12, disagreement_count * 4)
        conflict_messages.append("Multiple agents reported conflicting signals")

    final_confidence = max(50, min(95, round(final_confidence, 2)))

    strongest_source = max(
        influence_scores,
        key=lambda item: item["weighted_distance"],
    )["name"].lower()

    if strongest_source == "text":
        strongest_phrase = "text analysis was the strongest signal"
    elif strongest_source == "image":
        strongest_phrase = "image analysis was the strongest signal"
    elif strongest_source == "url":
        strongest_phrase = "source analysis was the strongest signal"
    else:
        strongest_phrase = "video analysis was the strongest signal"

    if final_result == "REAL":
        summary = f"The news appears real because {strongest_phrase}. The final confidence is {final_confidence}%."
    else:
        summary = f"The news appears fake because {strongest_phrase}. The final confidence is {final_confidence}%."

    missing_inputs = []
    if image_result is None or image_result.get("score") is None:
        missing_inputs.append("image")
    if video_result is None or video_result.get("score") is None:
        missing_inputs.append("video")

    if missing_inputs:
        if len(missing_inputs) == 2:
            summary += " No visual data available."
        else:
            summary += f" No {missing_inputs[0]} data available."

    if conflict_messages:
        summary += " Conflicting signals detected."

    if final_confidence > 80:
        summary += " Highly reliable prediction."
    elif final_confidence >= 60:
        summary += " Moderate confidence."
    else:
        summary += " Low confidence, result may not be accurate."

    final_explanation = summary + " " + " | ".join(explanation_parts)

    return MetaAgentResult(
        {
            "result": final_result,
            "confidence": final_confidence,
            "explanation": final_explanation.strip(),
        }
    )


def final_decision(text_result, image_result, video_result):
    result = meta_agent(text_result, image_result, video_result)
    return result["result"], result["confidence"], result["explanation"]
