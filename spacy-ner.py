import spacy
import time
import json
import logging
logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

def main():
    conversations = json.load(open("conversations.json", "r"))
    entities = extract_entities(conversations)

    # write entities to file in JSON format
    with open("spacy-ner-output/entities_out.json", "w") as f:
        json.dump(entities, f, indent=4)

def extract_entities(conversations_list):
    entities_per_turn = {}
    # time the conversation
    start_time = time.time()
    for conversation_id, conversation in conversations_list.items():
        logging.info("Processing conversation %s", conversation_id)
        turn_entities = []
        for doc in nlp.pipe([x['text'] for x in conversation]):
            turn_entities.append([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
        entities_per_turn[conversation_id] = turn_entities
        logging.info("\tProcessing took %s seconds", time.time() - start_time)
        start_time = time.time()
    return entities_per_turn


if __name__ == "__main__":
    main()