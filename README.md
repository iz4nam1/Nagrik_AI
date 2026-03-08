# Nagrik AI – Civic Scheme Assistant

Nagrik AI is an AI-powered civic assistant that helps citizens understand government schemes and check their eligibility instantly.

The system analyzes scheme documents using AI, extracts eligibility factors, asks personalized questions, and determines whether the user qualifies for the scheme.

Built for the AI for Bharat Hackathon by Hack2Skill.

---

## Problem

India has thousands of welfare schemes, but most citizens struggle to understand:

• Eligibility criteria  
• Required documents  
• Application process  

Scheme documents are often long and difficult to interpret.

Nagrik AI simplifies this process.

---

## Solution

Users upload a government scheme document (PDF or image), and the system:

1. Extracts text using OCR  
2. Uses AI to identify eligibility criteria  
3. Asks the user relevant questions  
4. Determines eligibility  
5. Provides next steps and similar schemes  
6. Generates an audio summary for accessibility

---

## Architecture

Frontend
→ HTML + JavaScript interface

Backend
→ AWS Lambda (Python)

Services used

• Google Gemini (AI analysis)  
• AWS Textract (OCR)  
• AWS Polly (Text-to-Speech)  
• AWS S3 (audio storage)  
• AWS DynamoDB (response caching)  
• AWS API Gateway (API endpoint)

---

## Features

• Upload PDF or image scheme documents  
• AI-powered eligibility analysis  
• Personalized question generation  
• Multi-language support (English, Hindi, Tamil)  
• Audio summary using text-to-speech  
• Similar scheme recommendations  
• Intelligent caching to reduce AI calls

---

## Tech Stack

Frontend
HTML, CSS, JavaScript

Backend
Python (AWS Lambda)

Cloud Services
AWS Textract  
AWS Polly  
AWS DynamoDB  
AWS S3  
AWS API Gateway

AI
Google Gemini 2.5 Flash

---

## How it Works

1. User uploads scheme document
2. Textract performs OCR
3. AI extracts eligibility factors
4. System asks personalized questions
5. AI evaluates eligibility
6. Results and recommendations are displayed

---

## Demo Flow

1. Upload a scheme document
2. AI extracts eligibility questions
3. User answers questions
4. System determines eligibility and benefits
5. Audio summary is generated

---

## Environment Variables

The Lambda function requires:
GEMINI_API_KEY
S3_BUCKET
DEBUG


---

## Future Improvements

• Vector search for large documents  
• More language support  
• Government scheme database integration  
• Mobile app interface  

---

## License

MIT License
