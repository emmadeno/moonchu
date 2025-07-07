import os
import re
from glob import glob
import ast
import pandas as pd # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
import google.generativeai as genai # type: ignore

FULL_TEXT_PDF = "../../Data/FullText.pdf"
ALL_PAGES_PNG = "../../Data/OCR/AllPagesPNG/"
ALL_PAGES_TXT_TESSERACT = "../../Data/OCR/AllPagesTXT/TESSERACT/"
ALL_PAGES_TXT_GEMINI = "../../Data/OCR/AllPag esTXT/GEMINI/"
ALL_PAGES_TXT_VISION_LINES = "../../Data/OCR/AllPagesTXT/VISION/LINES/"
ALL_PAGES_TXT_VISION_PARAGRAPHS = "../../Data/OCR/AllPagesTXT/VISION/PARAGRAPHS/"
ALL_PAGES_TXT_DOCUMENT = "../../Data/OCR/AllPagesTXT/DOCUMENT_AI/"

ALL_SENTENCES = "../../Data/TEXT/ALL_SENTENCES/"

ALL_CHAPTERS = "../../Data/TEXT/ALL_CHAPTERS/"

BOOK_TITLE = "CHINA THROUGH THE STEREOSCOPE."

ALL_SENTENCES_DF = "../../Data/TEXT/ALL_SENTENCES.csv"

FIRST_PAGE_OCR = 22
LAST_PAGE_OCR = 361

def get_list_topics():
    topics = read_file("../../Data/FINAL_TOPICS.txt")
    return ast.literal_eval(topics)

def API_KEY():
    key = read_file("../../API_KEY.txt")
    return key

def get_chapter(file_path):
    return file_path.split("/")[-2].split("_")[-1]

def get_all_sentences_df():
    all_files = []
    all_folders = get_folders_in_folder(ALL_SENTENCES)
    for folder in all_folders:
        all_files += get_files_in_directory(folder)

    all_sentences = []
    for file in all_files:
        text_in_file = read_file(file)
        sentences = text_in_file.split("\n")
        chapter_num = get_chapter(file)
        for sentence in sentences:
            all_sentences.append((sentence, chapter_num))
    return pd.DataFrame(all_sentences, columns=["text", "chapter"])

def get_df():
    stop_words = set(stopwords.words('english'))

    def filter_sentence(sentence):
        word_tokens = word_tokenize(sentence)
        tokens = [w for w in word_tokens if not w.lower() in stop_words and len(w) > 1]
        return tokens

    all_sentences = get_all_sentences_df()
    df = pd.DataFrame(all_sentences, columns=["text", "chapter"])
    df["tokens"] = df["text"].apply(lambda x : filter_sentence(x))
    df["num_tokens"] = df["tokens"].apply(lambda x : len(x))
    df = df[df["num_tokens"] > 0]
    df = df.reset_index(drop=True)
    return df

def join_split_words(text):
    split_text = text.split("\n")
    new_text = ""
    skip_first_word = False
    for i, line in enumerate(split_text):
        if len(line) > 0 and line[-1] == "-":
            this_line_words = line.split(" ")
            next_line_words = split_text[i + 1].split(" ")
            combined_word = this_line_words[-1][:-1] + next_line_words[0]
            this_line_words[:-1].append(combined_word)
            new_text += " ".join(this_line_words)
            skip_first_word = True
        else:
            if skip_first_word:
                line = (" ").join(line.split(" ")[1:])
                skip_first_word = False
            new_text += line
    return new_text
            

def write_to_file(file_name, text, create=False):
     file_rights = "x" if create else "w"
     f = open(file_name, file_rights)
     f.write(text)
     f.close()

def read_file(file_name):
    f = open(file_name, "r")
    text = f.read()
    f.close()
    return text

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_files_in_directory(directory):
    file_names = [ f.path for f in os.scandir(directory) if f.is_file() ]
    file_names.sort(key=natural_keys)
    return file_names

def get_page_number(filename: str) :
    return filename.split("/")[-1].split(".")[0].split("_")[-1]

def get_page_to_text_dictionary(directory):
    dict = {}
    file_names = get_files_in_directory(directory)
    for file in file_names:
        text = read_file(file)
        page = get_page_number(file)
        dict[page] = text
    return dict

def get_folders_in_folder(directory):
    subfolders = [ f.path for f in os.scandir(directory) if f.is_dir() ]
    subfolders.sort(key=natural_keys)
    return subfolders

def prompt_model_array(prompts : list[str]):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    all_responses = []
    for prompt in prompts:
        response = model.generate_content(contents=[prompt])
        all_responses.append(response.text)
    return all_responses

def create_directory(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        print(f"Directory '{name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")