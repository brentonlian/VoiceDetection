import sys
import os
import wave
import time
import json
import numpy as np
import soundcard as sc
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TOXIC_KEYWORDS = [
    "kill yourself", "retard", "trash", "noob", "stupid", "idiot", "I hate you", "loser", "you're stupid","meathead","evil","quit",
    "dumb", "f***", "b****", "n****", "c****", "kys", "die" "idiot", "moron", "imbecile", "dumbass", "dipshit", "dunce", "simpleton", "fool", "halfwit", "nitwit", "dullard", "ignoramus", "bonehead", "knucklehead", "blockhead", "pea-brain", "fucking idiot", "waste of space", "useless", "asshole", "jerk", "bastard", "son of a bitch", "dick", "prick", "douchebag", "scumbag", "shithead", "motherfucker", "dirtbag", "lowlife", "scum", "snake", "weasel", "rat", "pig", "sleazebag", "cunt", "bitch", "coward", "liar", "hypocrite", "narcissist", "psycho", "sociopath", "ugly", "hideous", "fatso", "lardass", "pig", "cow", "whale", "skeleton", "scrawny", "skinny bitch", "disgusting", "slob", "skank", "trashy", "freak", "mutant", "shit", "fuck", "fuckwit", "shitstain", "cumstain", "piss-ant", "dickweed", "asswipe", "fuckface", "shit-for-brains", "tool", "simp", "incel", "cuck", "Karen", "neckbeard", "thot", "basic", "clown", "bozo", "nonce", "annoying", "insufferable", "pathetic", "worthless", "incompetent", "lazy", "good-for-nothing", "two-faced", "backstabbing", "manipulative", "clingy", "needy", "desperate", "cringey", "cringe", "try-hard", "wannabe", "poser", "nr", "ft", "kke", "spc", "ch*nk", "whore", "slut", "slag", "bimbo", "retard", "cripple"
]

TOXIC_PHRASES = [
    "kill yourself","commit suicide","you should die", "you should end it all", "You were an accident","Make me a sandwich","Go back to the kitchen","waste of oxygen", "You're a bot"
]

class AudioRecorder:
    def __init__(self, sample_rate=44100, buffer_seconds=15):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.speaker = sc.get_microphone(sc.default_microphone().name)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ToxiGuard_Output')
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self):
        print(f"[ToxiGuard] Monitoring system audio via: {self.speaker.name}")
        print(f"[ToxiGuard] Output files will be saved to: {self.output_dir}")

    def get_last_seconds(self, seconds):
        numframes = int(self.sample_rate * seconds)
        with self.speaker.recorder(samplerate=self.sample_rate) as rec:
            data = rec.record(numframes=numframes)
        return data

    def save_to_wav(self, data, filename):
        if np.any(data):
            max_val = np.max(np.abs(data))
            scaled = np.int16(data / max_val * 32767) if max_val > 0 else np.zeros_like(data, dtype=np.int16)
        else:
            scaled = np.zeros_like(data, dtype=np.int16)
        filepath = os.path.join(self.output_dir, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(data.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(scaled.tobytes())
        print(f"[ToxiGuard] Saved audio to: {filepath}")
        return filepath

class ToxiGuardBackend:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.stt_model = whisper.load_model("base")

        # Context-aware NLP
        self.nlp_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        self.nlp_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
        self.nlp_model.eval()

    def transcribe_audio(self, data):
        tmp_path = os.path.join(self.recorder.output_dir, "_temp.wav")
        self.recorder.save_to_wav(data, "_temp.wav")
        print("[ToxiGuard] Transcribing audio...")
        result = self.stt_model.transcribe(tmp_path)
        print(f"[ToxiGuard] Transcription result: {result['text']}")
        return result['text']

    def check_toxicity(self, text):
        return [word for word in TOXIC_PHRASES if word.lower() in text.lower()],[word for word in TOXIC_KEYWORDS if word.lower() in text.lower()]

    def context_toxicity_score(self, text):
        inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            subtracting_score=(scores[0][1].item()*100)**0.7
            if subtracting_score>1:
                subtracting_score=1
            return  1-subtracting_score # Probability of toxic

    def transcribe_and_score(self, data, text):
        transcription_path = os.path.join(self.recorder.output_dir, "transcription.txt")
        report_path = os.path.join(self.recorder.output_dir, "toxicity_report.json")

        with open(transcription_path, "w") as f:
            f.write(text)

        found_keywords = []
        keyword_score = 0
        text_lower = text.lower().replace(".", "")
        for word in TOXIC_KEYWORDS:
            word_lower = word.lower()
            if word_lower in text_lower:
                mult2=text_lower.count(word_lower)
                multiplier = 2
                if "you " + word_lower in text_lower or "you are a " + word_lower or "you're a " or "you're so " + word_lower or "you're such a " + word_lower in text_lower:
                    multiplier = 15
                keyword_score += multiplier
                found_keywords.append(word)
        for word in TOXIC_PHRASES:
            word_lower = word.lower()
            if word_lower in text_lower:
                mult2=text_lower.count(word_lower)
                multiplier = 10
                multiplier=multiplier*mult2
                keyword_score += multiplier
                found_keywords.append(word) 
        x=len(text_lower) if len(text_lower)>0 else 1
        x=x if x<200 else 200   
        keyword_score = (keyword_score*2/x)**1
        if keyword_score>1:
            keyword_score=1
        context_score = self.context_toxicity_score(text)
        final_score = 0.5 * keyword_score + 0.5 * context_score

        report = {
            "transcription": text,
            "keyword_score": keyword_score,
            "context_score": context_score,
            "toxicity_score": final_score,
            "flagged_words": found_keywords
        }

        # Load existing reports
        try:
            with open(report_path, "r") as f:
                reports = json.load(f)
                if not isinstance(reports, list):
                    reports = []
        except (FileNotFoundError, json.JSONDecodeError):
            reports = []

        # Clear if 4 entries
        if len(reports) >= 4:
            reports = []

        # Append new report
        reports.append(report)



        # Save reports list
        with open(report_path, "w") as f:
            json.dump(reports, f, indent=2)

        print(f"[ToxiGuard] Transcription saved to: {transcription_path}")
        print(f"[ToxiGuard] Toxicity report saved to: {report_path}")

    def run_monitor_loop(self, interval=5):
        self.recorder.start_recording()
        print("[ToxiGuard] Starting background monitoring...")
        while True:
            buffer_data = self.recorder.get_last_seconds(15)
            transcribed_text = self.transcribe_audio(buffer_data)
            found = self.check_toxicity(transcribed_text)
            print(f"[ToxiGuard] Found toxic keywords: {found}")
            if found:
                print("[ToxiGuard] Toxic behavior detected! Saving report...")
                self.transcribe_and_score(buffer_data, transcribed_text)
            print("[ToxiGuard] Loop complete. Waiting for next check...")
            time.sleep(interval)

if __name__ == '__main__':
    app = ToxiGuardBackend()
    app.run_monitor_loop()
