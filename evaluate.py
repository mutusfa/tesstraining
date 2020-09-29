import argparse

import pandas as pd
import textdistance

import clean_ground_truth
import ocr
import preprocessing


def evaluate(ground_truth, recognised_text):
    char_accuracy = 1 - next(distance(ground_truth, recognised_text))
    word_accuracy = 1 - next(
        distance(ground_truth.split(), recognised_text.split())
    )
    yield char_accuracy, word_accuracy


def distance(ground_truth, recognised_text):
    """Return normalized number of edits between tokens.

    Edits are removal, insertion and replacements, weighted equally.
    Number of edits is divided by number of tokens (in the ground_truth).

    Lower number is better.
    """
    yield textdistance.levenshtein(ground_truth, recognised_text) / len(
        ground_truth
    )


def compile_text(results):
    lines = []
    words = []
    left = []
    top = []
    single_line = []
    page = []
    pixel_threshold = 40
    confidence_threshold = 50
    for page_number, text, conf, bounding_box in results:
        if round(conf) < confidence_threshold:
            continue
        words.append(text)
        left.append(bounding_box[0])
        page.append(page_number)
        # Manage slightly rotated text
        if not single_line:  # First line in a document
            single_line.append(bounding_box[1])
        elif (  # Still on the same line
            int(single_line[-1]) + pixel_threshold > int(bounding_box[1])
            and int(single_line[-1]) < int(bounding_box[1]) + pixel_threshold
            and int(page[-1]) == int(page_number)
        ):
            single_line.append(bounding_box[1])
        else:  # Line broke
            for y in single_line:
                top.append(single_line[0])
            single_line[:] = [bounding_box[1]]
    for y in single_line:  # last line didn't break
        top.append(single_line[0])
    df = pd.DataFrame({"word": words, "left": left, "top": top, "page": page})
    df = df.sort_values(by=["page", "top", "left"])
    grouped_by_lines = df.groupby(by=["page", "top"])
    for (page, top), line in grouped_by_lines:
        lines.append(" ".join(line["word"].tolist()))
    return "\n".join(lines)


def process_file(filepath):
    # tesseract read text
    tiff_image_path = next(preprocessing.process(filepath))
    tessract_results = ocr.process(tiff_image_path)
    tess_text = compile_text(tessract_results)
    yield tess_text
    # ground truth
    yield from clean_ground_truth.process(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Script to test accuracy of ocr"
    )
    parser.add_argument("--files", type=str, nargs="+", required=True)
    args = parser.parse_args()

    char_accuracy_avg = 0
    word_acuracy_avg = 0
    num_files = len(args.files)
    for filepath in args.files:
        processed = process_file(filepath)
        tess_text = next(processed)
        ground_truth = next(processed)
        print(tess_text)
        print(ground_truth)
        char_accuracy, word_accuracy = next(evaluate(ground_truth, tess_text))
        print(f"{char_accuracy:3.2}, {word_accuracy:3.2}, {filepath}")
        char_accuracy_avg += char_accuracy / num_files
        word_acuracy_avg += word_accuracy / num_files
        break
    print(f"Normalized {char_accuracy_avg:3.2}, {word_acuracy_avg:3.2}")


if __name__ == "__main__":
    main()
