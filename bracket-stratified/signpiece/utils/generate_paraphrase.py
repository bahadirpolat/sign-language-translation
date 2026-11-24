import torch
import pandas as pd
from itertools import product

from transformers import pipeline
from nltk.tokenize import sent_tokenize

from tqdm.auto import tqdm 
tqdm.pandas()

def parse_args():

    import argparse
    parser = argparse.ArgumentParser(description="Generate paraphrases for annotations")

    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/annotations.json",
        help="Path to the annotation file",
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='boun-tabi-LMG/turna_paraphrasing_opensubtitles',
        help='Model ID for the paraphrasing model',
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/annotations_with_paraphrases.json",
        help="Path to the output file",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams to use for beam search",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="Number of paraphrases to generate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    paraphrasing = pipeline(
        model=args.model_id,
        device=0 if torch.cuda.is_available() else -1,
    )

    def paraphrase(sentence, 
                   num_beams=args.num_beams,
                   num_return_sequences=args.num_return_sequences,
                   max_new_tokens=20 # Max number of new tokens to generate (DEFAULT: 20)
                   ):
        
        sentences = sent_tokenize(sentence)

        print('Sentences:', sentences)

        if len(sentences) > 7:
            print("Too many sentences in the annotation. Skipping...")
            return [sentence]

        if len(sentences) == 1:
            paraphrases = [sentence] + [
                generation["generated_text"]
                for generation in paraphrasing(
                    sentence,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                )
            ]
            
            paraphrases = list(set(paraphrases))
            
            for paraphrase in paraphrases:
                print('Paraphrase:', paraphrase)

            print('*'*50)

            return paraphrases

        paraphrases = []
        for sentence in sentences:
            paraphrases.append([sentence] + [
                generation["generated_text"]
                for generation in paraphrasing(
                    sentence,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                )
            ])
    
        # Combine paraphrases from different sentences
        paraphrases = list(product(*paraphrases))
        paraphrases = [" ".join(paraphrase) for paraphrase in paraphrases]

        # Get only unique paraphrases
        paraphrases = list(set(paraphrases))

        # Get maximum 10 paraphrases
        paraphrases = paraphrases[:10]

        for paraphrase in paraphrases:
            print('Paraphrase:', paraphrase)
        print('*'*50)

        return paraphrases
    
    # Generate paraphrases for each annotation
    annotations = pd.read_csv(args.annotation_file)
    if args.debug:
        annotations = annotations.head(5)

    annotations["paraphrases"] = annotations["text"].progress_apply(paraphrase)

    # Save the annotations with paraphrases
    annotations.to_csv(args.output_file, index=False)