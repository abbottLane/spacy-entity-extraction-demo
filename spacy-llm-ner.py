from spacy_llm.util import assemble
import json
import time
from collections import defaultdict
import sklearn.metrics
import logging
logging.basicConfig(level=logging.INFO)

nlp = assemble("config.cfg")

def main():
    conversations = json.load(open("conversations.json", "r"))
    entities = extract_entities(conversations)

    #evaluate(entities, conversations)

    # write entities to file in JSON format
    with open("spacy-llm-ner-output/entities_out.json", "w") as f:
        json.dump(entities, f, indent=4)

def extract_entities(conversations_list):
    entities_per_turn = {}
    # time the conversation
    start_time = time.time()
    for conversation_id, conversation in conversations_list.items():
        logging.info("Processing conversation %s", conversation_id)
        turn_entities = list()
        for doc in nlp.pipe([x['text'] for x in conversation]):
            for ent in doc.ents:
                doc_ents = list()
                doc_ents.append({ent.label_: ent.text})
            turn_entities.append(doc_ents)
        entities_per_turn[conversation_id] = turn_entities
        logging.info("\tProcessing took %s seconds", time.time() - start_time)
        start_time = time.time()
    return entities_per_turn

def evaluate(entities, gold_conversations):
    """Calculate P/R/F1 scores for the entities extracted by the model.
       args:
            entities: tuple of [text, start_char, end_char, label] for each entity
            gold_conversations: dict of conversation_id: {"annotations": {with label name keys and list of entities as values}}
    """
    raise NotImplementedError("P/R/F1 Evaluation not implemented yet")

if __name__ == "__main__":
    main()