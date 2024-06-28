import spacy
import re

nlp= spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion."

doc=nlp(text)

print(doc.ents)


# ner
for e in doc.ents:
    print(e.label_, e.text)

# relationship extraction

for token in doc:
    if token.dep_ in ("nsubj", "dobj"):
        subject = [w for w in token.head.lefts if w.dep_ == "nsubj"]
        if subject:
            print(f"Relation: ({subject[0].text}, {token.head.text}, {token.text})")



text1 = "The meeting is scheduled for 10:00 AM on 25th June 2024."

# Define patterns
date_pattern = r"\b\d{1,2}(?:th|st|nd|rd)?\s\w+\s\d{4}\b"
time_pattern = r"\b\d{1,2}:\d{2}\s?(?:AM|PM)?\b"

# Extract dates and times
dates = re.findall(date_pattern, text1)
times = re.findall(time_pattern, text1)

print("Dates:", dates)
print("Times:", times)



# Process text
text2 = "John Doe was elected president of the club on January 15th, 2024."
doc2 = nlp(text2)

# Extract events
for token in doc2:
    if token.dep_ == "nsubjpass" and token.head.lemma_ == "elect":
        subject = token.text
        object = [w.text for w in token.head.rights if w.dep_ == "attr"]
        date = [ent.text for ent in doc2.ents if ent.label_ == "DATE"]
        if object and date:
            print(f"Event: {subject} was elected as {object[0]} on {date[0]}")



# Extract dependencies
for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")