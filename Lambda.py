import json
import boto3
import base64
import os
import uuid
import urllib.request
import urllib.error
import hashlib
import time
import re

# ---------- ENV ----------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
S3_BUCKET      = os.environ.get("S3_BUCKET")
MODEL          = "gemini-2.5-flash"
DEBUG          = os.environ.get("DEBUG", "false")
# Bump this string whenever prompts or answer format changes to bust stale cache
CACHE_VERSION  = "v3"

# ---------- AWS CLIENTS ----------

polly     = boto3.client("polly")
textract  = boto3.client("textract")
s3        = boto3.client("s3")
dynamodb  = boto3.resource("dynamodb")

cache_table = dynamodb.Table("civic_ai_cache")

HTTP = urllib.request.build_opener()

HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
    "Content-Type":                 "application/json"
}

VOICE_MAP = {
    "english": "Joanna",
    "hindi":   "Aditi",
    "tamil":   "Raveena"
}

# ---------- UTIL ----------

def clean_input(text):
    """
    FIX: Original truncated hard at char boundary, destroying eligibility numbers.
    Now splits on sentence/paragraph boundaries so context is preserved.
    """
    text = str(text).strip()
    if len(text) <= 6000:
        return text

    # Keep first 3000 chars (scheme description / title / criteria)
    # and last 2000 chars (application steps / contacts) — split at whitespace
    head = text[:3000]
    tail = text[-2000:]

    # Walk back head to last whitespace so we don't cut a word/number
    cut = head.rfind(" ")
    if cut > 2000:
        head = head[:cut]

    return head + "\n...[truncated]...\n" + tail


def detect_file_type(raw_bytes):
    """
    FIX: Backend had no way to distinguish PDF from image.
    Detect by magic bytes so routing is unambiguous.
    """
    if raw_bytes[:4] == b"%PDF":
        return "pdf"
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


# ---------- JSON EXTRACTION ----------

def extract_json(text):
    if not text:
        return None
    try:
        text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        text = text.strip()

        start = text.find("{")
        end   = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

    except Exception as e:
        print("JSON parse error:", e)
        print("RAW GEMINI OUTPUT:", text)

    return None


# ---------- CACHE ----------

def generate_cache_key(document_text, mode, answers, language):
    # CACHE_VERSION is mixed into the hash — bumping it instantly invalidates
    # all previously cached results without touching DynamoDB manually.
    payload = (
        CACHE_VERSION +
        document_text +
        mode +
        language +
        MODEL +
        json.dumps(answers, sort_keys=True)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cache(cache_key):
    try:
        response = cache_table.get_item(Key={"cache_key": cache_key})
        if "Item" in response:
            return response["Item"]["data"]
    except Exception as e:
        print("Cache read error:", e)
    return None


def store_cache(cache_key, data):
    try:
        cache_table.put_item(Item={
            "cache_key": cache_key,
            "data":      data,
            "timestamp": int(time.time())
        })
    except Exception as e:
        print("Cache write error:", e)


def is_valid_result(mode, result):
    if not isinstance(result, dict):
        return False
    if mode == "questions":
        return bool(result.get("questions"))
    if mode == "personalize":
        summary = result.get("summary", {})
        return bool(summary.get("summary"))
    return False


# ---------- GEMINI ----------

def call_gemini(prompt, max_tokens=600):
    """
    FIX: Single shared token limit of 700 caused personalize JSON to be
    truncated mid-response, breaking JSON parse → "Unable to determine eligibility".
    Now each call site specifies the budget it actually needs.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     0.3
        }
    }

    data = json.dumps(payload).encode()

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            response = HTTP.open(req, timeout=25)
            result   = json.loads(response.read())

            parts = (
                result.get("candidates", [{}])[0]
                      .get("content", {})
                      .get("parts", [])
            )

            # gemini-2.5-flash is a thinking model: parts[0] is always
            # {"thought": true, "text": "<internal reasoning>"}.
            # Taking parts[0] blindly returns the thinking blob, not the answer.
            # Fix: skip any part with thought=True and join the remaining text.
            text = " ".join(
                p.get("text", "")
                for p in parts
                if not p.get("thought", False)
            ).strip()

            finish = (
                result.get("candidates", [{}])[0]
                      .get("finishReason", "UNKNOWN")
            )
            print(f"GEMINI finishReason={finish} textLen={len(text)}", flush=True)
            print("GEMINI RAW:", text[:500], flush=True)

            if finish == "MAX_TOKENS":
                print("WARNING: Gemini response was truncated — increase max_tokens", flush=True)

            return text

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            print(f"Gemini HTTP error: code={e.code} body={body}", flush=True)
            if e.code == 429:
                time.sleep(2 ** attempt)
            else:
                time.sleep(2)
        except Exception as e:
            print(f"Gemini call failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(2 ** attempt)

    print("Gemini: all 3 attempts failed", flush=True)
    return ""


# ---------- DOCUMENT TYPE ----------

def detect_document_type(text):
    prompt = f"""
Classify this government document.

Return ONLY JSON.

{{ "document_type":"" }}

DOCUMENT:
{text}
"""
    parsed = extract_json(call_gemini(prompt, max_tokens=80))
    return parsed if parsed else {"document_type": "policy"}


# ---------- FACTORS ----------

def extract_factors(text):
    prompt = f"""
Extract eligibility factors from this government scheme document.

Return ONLY JSON. Use ONLY these exact factor strings:
- "income"
- "age"
- "student"
- "location"
- "category"

Example output: {{ "factors": ["income", "location", "category"] }}

DOCUMENT:
{text}
"""
    parsed = extract_json(call_gemini(prompt, max_tokens=150))
    factors = list(parsed.get("factors", [])) if parsed else []

    # Hard fallback: scan raw text for keywords Gemini might have missed.
    # This ensures SC/ST/OBC schemes always get the category question,
    # scholarship schemes always get the student question, etc.
    tl = text.lower()
    keyword_map = {
        "income":   ["income", "annual", "salary", "earning", "rupee", "lakh"],
        "student":  ["student", "scholar", "class", "graduation", "course", "school", "college", "enrolled"],
        "age":      ["age", "years old", "born", "year of birth"],
        "location": ["state", "district", "domicile", "resident", "residing"],
        "category": ["sc", "st", "obc", "ews", "scheduled caste", "scheduled tribe",
                     "other backward", "caste", "category", "social"]
    }
    for factor, keywords in keyword_map.items():
        if factor not in factors:
            if any(kw in tl for kw in keywords):
                factors.append(factor)

    return {"factors": factors if factors else ["income", "student"]}


# ---------- QUESTIONS ----------

def generate_questions(factors, language):
    questions_map = {
        "english": {
            "income":   "What is your annual household income?",
            "student":  "Are you currently a student?",
            "age":      "What is your age?",
            "location": "Which state do you live in?",
            "category": "Which social category do you belong to? (SC/ST/OBC/General/EWS)"
        },
        "hindi": {
            "income":   "आपके परिवार की वार्षिक आय कितनी है?",
            "student":  "क्या आप वर्तमान में छात्र हैं?",
            "age":      "आपकी उम्र क्या है?",
            "location": "आप किस राज्य में रहते हैं?",
            "category": "आप किस सामाजिक श्रेणी से हैं? (SC/ST/OBC/सामान्य/EWS)"
        },
        "tamil": {
            "income":   "உங்கள் குடும்பத்தின் ஆண்டு வருமானம் என்ன?",
            "student":  "நீங்கள் தற்போது மாணவரா?",
            "age":      "உங்கள் வயது என்ன?",
            "location": "நீங்கள் எந்த மாநிலத்தில் வசிக்கிறீர்கள்?",
            "category": "நீங்கள் எந்த சமூக பிரிவை சேர்ந்தவர்? (SC/ST/OBC/General/EWS)"
        }
    }

    lang = language.lower()
    if lang not in questions_map:
        lang = "english"

    questions = []

    for f in (factors or []):
        f = f.lower()
        if "income" in f:
            questions.append(questions_map[lang]["income"])
        elif "student" in f:
            questions.append(questions_map[lang]["student"])
        elif "age" in f:
            questions.append(questions_map[lang]["age"])
        elif "location" in f or "state" in f:
            questions.append(questions_map[lang]["location"])
        elif any(k in f for k in ("category", "caste", "sc", "st", "obc", "ews")):
            questions.append(questions_map[lang]["category"])

    if not questions:
        questions.append(questions_map[lang]["income"])
        questions.append(questions_map[lang]["student"])

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:5]


# ---------- PERSONALIZED ANALYSIS ----------

def personalized_analysis(text, answers, language):
    lang_map = {
        "english": "English",
        "hindi":   "Hindi",
        "tamil":   "Tamil"
    }
    output_lang = lang_map.get(language.lower(), "English")

    # FIX 1: Use 1500 tokens — the full JSON with 5 fields easily exceeds 700,
    #         causing truncation that breaks json.loads → silent fallback.
    # FIX 2: eligibility_result was a placeholder string "Eligible / Not Eligible /
    #         Possibly Eligible" — Gemini was returning it literally. Now the
    #         instruction is explicit and separate from the JSON template.
    # Pre-format answers outside the f-string — nested f-strings inside
    # f-string expressions are unreliable in Python 3.10 (Lambda runtime)
    if answers:
        answers_text = "\n".join(f"- {q}: {a}" for q, a in answers.items())
    else:
        answers_text = "No answers provided"

    prompt = f"""
You are a government scheme eligibility engine. Analyse the document and user data below.

RULES:
- Reply with ONLY a single valid JSON object. No markdown, no backticks, no explanation.
- eligibility_result MUST be exactly one of: "Eligible", "Not Eligible", "Possibly Eligible"
- All text fields must be written in simple {output_lang}.
- Keep each text field under 120 words.
- similar_schemes must have 2 entries.

JSON FORMAT (fill every field, do not leave any empty):
{{
  "summary": "2-3 sentence plain-language overview of what this scheme offers",
  "benefits": "specific financial or material benefits the user would receive",
  "next_steps": "step-by-step action the user should take to apply",
  "eligibility_result": "Eligible",
  "similar_schemes": [
    {{"scheme": "scheme name", "reason": "why it is relevant to this user"}},
    {{"scheme": "scheme name", "reason": "why it is relevant to this user"}}
  ]
}}

DOCUMENT:
{text}

USER ANSWERS:
{answers_text}
"""
    response = call_gemini(prompt, max_tokens=2500)
    parsed   = extract_json(response)

    if parsed:
        return parsed

    return {
        "summary":             "",
        "benefits":            "",
        "next_steps":          "",
        "eligibility_result":  "Unable to determine eligibility",
        "similar_schemes":     []
    }


# ---------- AUDIO ----------

def make_audio(text, language):
    if not text:
        return None

    voice = VOICE_MAP.get(language.lower(), "Joanna")

    try:
        response = polly.synthesize_speech(
            Text=text[:2000],
            OutputFormat="mp3",
            VoiceId=voice
        )
        audio = response["AudioStream"].read()
    except Exception as e:
        print("Polly error:", e)
        return None

    filename = f"audio-{uuid.uuid4()}.mp3"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=filename,
        Body=audio,
        ContentType="audio/mpeg"
    )

    return f"https://{S3_BUCKET}.s3.amazonaws.com/{filename}"


# ---------- OCR — IMAGE ----------

def ocr_image(image_bytes):
    """
    FIX: Original passed raw bytes without checking type.
    Now validates magic bytes first; raises a clear error if unsupported.
    Textract supports JPEG / PNG / TIFF for synchronous detection.
    """
    ftype = detect_file_type(image_bytes)
    if ftype not in ("jpeg", "png", "tiff"):
        raise ValueError(
            f"Unsupported image type '{ftype}' for synchronous OCR. "
            "Use ocr_pdf() for PDF files."
        )

    try:
        response = textract.detect_document_text(
            Document={"Bytes": image_bytes}
        )
    except Exception as e:
        print("Textract image error:", e)
        return ""

    lines = [
        block["Text"]
        for block in response["Blocks"]
        if block["BlockType"] == "LINE"
    ]
    return "\n".join(lines)


# ---------- OCR — PDF ----------
# FIX: Completely new function. Textract cannot read PDF bytes directly;
# the document must be in S3. We upload, poll until done, then delete.

def ocr_pdf(pdf_bytes):
    tmp_key = f"tmp-ocr-{uuid.uuid4()}.pdf"

    try:
        # 1. Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=tmp_key,
            Body=pdf_bytes,
            ContentType="application/pdf"
        )

        # 2. Start async detection
        job = textract.start_document_text_detection(
            DocumentLocation={
                "S3Object": {"Bucket": S3_BUCKET, "Name": tmp_key}
            }
        )
        job_id = job["JobId"]

        # 3. Poll (max ~60 s, well within Lambda default 3 min)
        for _ in range(30):
            result = textract.get_document_text_detection(JobId=job_id)
            status = result["JobStatus"]

            if status == "SUCCEEDED":
                lines = []
                pages = [result]

                # Paginate if Textract result is multi-page
                next_token = result.get("NextToken")
                while next_token:
                    page = textract.get_document_text_detection(
                        JobId=job_id, NextToken=next_token
                    )
                    pages.append(page)
                    next_token = page.get("NextToken")

                for p in pages:
                    for block in p["Blocks"]:
                        if block["BlockType"] == "LINE":
                            lines.append(block["Text"])

                return "\n".join(lines)

            elif status == "FAILED":
                print("Textract async job failed:", result.get("StatusMessage"))
                return ""

            time.sleep(2)

        print("Textract PDF job timed out")
        return ""

    except Exception as e:
        print("PDF OCR error:", e)
        return ""

    finally:
        # 4. Always clean up the temporary S3 object
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=tmp_key)
        except Exception as cleanup_err:
            print("S3 cleanup error:", cleanup_err)


# ---------- HANDLER ----------

def lambda_handler(event, context):

    # Handle CORS preflight
    method = (
        event.get("requestContext", {})
             .get("http", {})
             .get("method", "")
    )
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": HEADERS, "body": ""}

    try:
        body = event.get("body") or {}
        if isinstance(body, str):
            body = json.loads(body)

        mode          = body.get("mode", "questions")
        document_text = body.get("document_text")
        answers       = body.get("answers", {})
        language      = body.get("language", "english")
        file_b64      = body.get("file")            # FIX: unified field for image OR pdf
        file_type_hint = body.get("file_type", "")  # FIX: "image" or "pdf" from frontend

        # ----- OCR routing -----
        if file_b64:
            raw_bytes   = base64.b64decode(file_b64)
            actual_type = detect_file_type(raw_bytes)

            if actual_type == "pdf" or file_type_hint == "pdf":
                document_text = ocr_pdf(raw_bytes)
            else:
                # image path (jpeg / png / tiff)
                document_text = ocr_image(raw_bytes)

        if not document_text:
            return {
                "statusCode": 400,
                "headers":    HEADERS,
                "body":       json.dumps({"error": "Document required (text, image, or PDF)"})
            }

        document_text = clean_input(document_text)

        # ----- Cache lookup -----
        cache_key = generate_cache_key(document_text, mode, answers, language)

        if DEBUG != "true":
            cached = get_cache(cache_key)
            if cached:
                print("CACHE HIT")
                return {"statusCode": 200, "headers": HEADERS, "body": json.dumps(cached)}

        print("CACHE MISS")

        # ----- Mode routing -----
        if mode == "questions":
            doc_type  = detect_document_type(document_text)["document_type"]
            factors   = extract_factors(document_text)["factors"]
            questions = generate_questions(factors, language)

            result = {
                "document_type":  doc_type,
                "questions":      questions,
                "document_text":  document_text  # FIX: return OCR'd text so frontend stores it
            }

        elif mode == "personalize":
            try:
                analysis = personalized_analysis(document_text, answers, language)
            except Exception as e:
                print("analysis error:", e)
                analysis = {}
            

            summary = {
                "summary":    analysis.get("summary",    ""),
                "benefits":   analysis.get("benefits",   ""),
                "next_steps": analysis.get("next_steps", "")
            }

            # Only generate audio when there is actual summary text
            audio_text = summary.get("summary", "")
            audio      = make_audio(audio_text, language) if audio_text else None

            result = {
                "summary":            summary,
                "audio_url":          audio,
                "eligibility_result": analysis.get("eligibility_result"),
                "similar_schemes":    analysis.get("similar_schemes", [])
            }

        else:
            result = {"error": "Invalid mode. Use 'questions' or 'personalize'."}

        if is_valid_result(mode, result):
            store_cache(cache_key, result)

        return {
            "statusCode": 200,
            "headers":    HEADERS,
            "body":       json.dumps(result)
        }

    except Exception as e:
        print("Lambda error:", str(e))
        return {
            "statusCode": 500,
            "headers":    HEADERS,
            "body":       json.dumps({"error": "Processing failed", "detail": str(e)})
        }
