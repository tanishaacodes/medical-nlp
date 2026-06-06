from rapidfuzz import process,fuzz
MEDICAL_DICT=["Amoxicillin", "Ibuprofen", "Paracetamol", "Metformin", 
    "Atorvastatin", "Lisinopril", "Amlodipine", "Albuterol"
]


def autocorrect_medical(input_word):
    result=process.extractOne(
        input_word,MEDICAL_DICT,scorer=fuzz.WRatio,score_cutoff=70
    )

    if result:
        return result[0]
    return input_word

SIG_CODES = {
    "bid": "twice a day",
    "tid": "three times a day",
    "qid": "four times a day",
    "po": "by mouth",
    "prn": "as needed",
    "pc": "after meals",
    "ac": "before meals",
    "qd": "every day",
    "stat": "immediately"
}

def translate_medical_shorthand(text):
    words = text.lower().split()
    translated = [SIG_CODES.get(w, w) for w in words]
    return " ".join(translated)
