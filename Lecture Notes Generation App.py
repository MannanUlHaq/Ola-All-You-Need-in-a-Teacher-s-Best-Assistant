import os  # Provides functions for interacting with the operating system
import numpy as np  # Used for numerical operations and handling arrays
import sounddevice as sd  # Library for recording and playing sound
import torch  # PyTorch, used for deep learning and tensor computations
import whisper  # OpenAI's Whisper model for speech-to-text transcription
import wave  # Module for reading and writing WAV files
import keyboard  # Library for detecting keyboard input events
import threading  # Used for running multiple operations concurrently
import noisereduce as nr  # Library for reducing noise in audio signals
import scipy.io.wavfile as wav  # Module for handling WAV files using SciPy
import cv2  # OpenCV for image processing tasks
from pdf2image import convert_from_path  # Converts PDF pages to images
import re  # Regular expressions for text processing
import tkinter as tk  # GUI library for creating graphical interfaces
from tkinter import filedialog  # Module for opening file dialog boxes
import tempfile  # Library for creating temporary files and directories
import pytesseract  # Optical Character Recognition (OCR) using Tesseract
import gc  # Garbage collector for managing memory
from concurrent.futures import ThreadPoolExecutor  # For parallel execution of tasks
from nltk.tokenize import sent_tokenize  # Tokenizes text into sentences
from nltk.corpus import wordnet  # WordNet for finding synonyms and meanings
from sentence_transformers import SentenceTransformer, util  # Pre-trained transformer model for sentence embeddings and similarity
import warnings  # Handles warnings in the program
warnings.filterwarnings("ignore")  # Suppresses warnings to keep the output clean
import time  # Provides time-related functions (e.g., measuring execution time)
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Function to load the Whisper model
def load_whisper_model():
    """
    Loads the Whisper model from a local .pt file.
    """

    # Check if a GPU is available; otherwise, default to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if the Whisper model file exists in the current directory
    if os.path.exists("large-v3-001.pt"):        
        # Load the Whisper model and assign it to the specified device (GPU or CPU)
        model = whisper.load_model("large-v3-001.pt", device=device)
        print("Whisper model Loaded Successfully...")
        
        return model  # Return the loaded model
    else:
        # Raise an error if the model file is not found
        raise FileNotFoundError("Whisper model not found.")

# Load fine-tuned T5 explanation generation model and tokenizer
model_path = "t5_explanation_model"
t5model = T5ForConditionalGeneration.from_pretrained(model_path)
t5tokenizer = T5Tokenizer.from_pretrained(model_path)
# Load fine-tuned T5 summarization model and tokenizer
model_path = "t5_summary_model"
t5_summarization_model = T5ForConditionalGeneration.from_pretrained(model_path)
t5_summarization_tokenizer = T5Tokenizer.from_pretrained(model_path)
print("T5 models Loaded Successfully...")

# Load the SentenceTransformer model for generating sentence embeddings
mpnetMODEL = SentenceTransformer("paraphrase_mpnet_model")
print("MPNet model Loaded Successfully...")

# Global variables

# Flag to track whether recording is in progress
recording = False  

# List to store recorded audio data for each slide
audio_data = []  

# List to store extracted text from slides
slide_texts = []  

# Variable to keep track of the current slide index
current_slide = 0

# Function to open a file dialog and allow the user to select a PDF file
def select_pdf():
    # Create a hidden root window (Tkinter main window)
    root = tk.Tk()
    root.withdraw()  # Hide the main GUI window
    root.attributes('-topmost', True)  # Ensure the dialog appears on top
    root.update()  # Force the window to refresh and appear correctly

    # Open a file dialog to select a PDF file
    file_path = filedialog.askopenfilename(
        title="Select PDF File",  # Title of the file dialog window
        filetypes=[("PDF Files", "*.pdf")]  # Only allow selection of PDF files
    )
    
    # Return the selected file path if a file is chosen, otherwise return None
    return file_path if file_path else None

# Function to extract slides (images) from the selected PDF
def extract_slides():
    # Prompt user to select a PDF file
    pdf_path = select_pdf()
    
    # Check if no file was selected
    if not pdf_path:
        print("No file selected. Exiting...")
        exit()  # Terminate the program if no file is selected
    
    print(f"Processing PDF: {pdf_path}")  # Print selected PDF file path
    
    # Convert the selected PDF file into images (one per page/slide)
    return convert_from_path(pdf_path)  # Ensure convert_from_path is imported

def clean_ocr_text(text):
    """Fixes common OCR mistakes while preserving numbering."""

    # Ensure proper spacing after multi-level numbers (e.g., "1.1.Subheading" → "1.1. Subheading")
    text = re.sub(r"(\d+\.\d+)\.(\w)", r"\1. \2", text)  

    # Ensure spacing after single-level numbers (e.g., "1.Heading" → "1. Heading")
    text = re.sub(r"(\d+)\.([A-Za-z])", r"\1. \2", text)

    # Remove incorrect spaces before punctuation (e.g., "Hello , world ." → "Hello, world.")
    text = re.sub(r"\s+\.", ".", text)  # Remove extra spaces before a period
    text = re.sub(r"\s+,", ",", text)  # Remove extra spaces before a comma

    # Fix incorrect newlines breaking words or sentences
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # Fix hyphenated word splits across lines
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)  # Merge words split by a newline into a single sentence

    return text

def extract_text_from_slide(slide_image, current_slide):
    """
    Extracts and formats text from a slide image using OCR and text processing techniques.
    """

    print(f"Extracting text from Slide {current_slide}...")

    # Save slide as a temporary image file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        slide_image.save(temp_file.name, format="PNG")  # Save the image in PNG format

    # Load the saved image and convert it to grayscale for better OCR accuracy
    image = cv2.imread(temp_file.name)  # Read the image using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Perform OCR (Optical Character Recognition) to extract text from the grayscale image
    ocr_result = pytesseract.image_to_string(gray)  # Extract text using Tesseract OCR
    ocr_result = clean_ocr_text(ocr_result)  # Clean the extracted text using OCR correction function

    # Initialize a dictionary to store structured extracted data
    extracted_data = {"headings": []}  # List to hold structured heading and text data
    current_heading = None  # Variable to store the current heading
    current_subheading = None  # Variable to store the current subheading
    current_text = []  # List to store extracted text blocks

    # Helper function to store collected text under the appropriate heading/subheading
    def store_text():
        """
        Stores collected text under the relevant heading or subheading.
        """
        nonlocal current_heading, current_subheading, current_text

        if not current_text:
            return  # No text to store, exit function

        text_block = " ".join(current_text).strip()  # Join text lines into a single block

        if current_subheading:  
            # If a subheading exists, append text under the last subheading
            if extracted_data["headings"] and extracted_data["headings"][-1]["subheadings"]:
                extracted_data["headings"][-1]["subheadings"][-1]["text"] += " " + text_block
        elif current_heading:  
            # If a heading exists, append text under the last heading
            if extracted_data["headings"]:
                extracted_data["headings"][-1]["text"] += " " + text_block

        current_text = []  # Reset text collection for the next block

    # Process extracted text line by line
    for line in ocr_result.split("\n"):
        line = line.strip()  # Remove leading and trailing whitespace
        if not line:
            continue  # Skip empty lines

        # Identify headings (e.g., "1. Heading:")
        heading_match = re.match(r"^(\d+\.)\s(.+):?$", line)  # Matches "1. Heading:"
        subheading_match = re.match(r"^(\d+\.\d+)\.\s(.+):?$", line)  # Matches "1.1. Subheading:"

        if heading_match:
            store_text()  # Store previous text before starting a new heading
            current_heading = f"{heading_match.group(1)} {heading_match.group(2)}"  # Preserve number + heading
            current_subheading = None  # Reset subheading
            extracted_data["headings"].append({"heading": current_heading, "text": "", "subheadings": []})  # Add heading entry

        elif subheading_match:
            store_text()  # Store previous text before starting a new subheading
            current_subheading = f"{subheading_match.group(1)}. {subheading_match.group(2)}"  # Preserve correct format
            if extracted_data["headings"]:
                extracted_data["headings"][-1]["subheadings"].append({"subheading": current_subheading, "text": ""})  # Add subheading entry

        else:
            current_text.append(line)  # Collect text content
            if line.endswith("."):  # If the line ends with a period, store the text
                store_text()

    # Store any remaining text at the end of the loop
    store_text()

    return extracted_data  # Return structured extracted data

def record_audio(sample_rate=16000, stop_event=None):
    """
    Records audio in a separate thread until 'SPACE' is pressed.

    Parameters:
    sample_rate (int): The sampling rate for audio recording (default: 16000 Hz).
    stop_event (threading.Event): A threading event used to stop the recording.
    """    
    global audio_data
    audio_data = []  # Initialize an empty list to store recorded audio chunks.

    def callback(indata, frames, time, status):
        """
        Callback function that gets called for each audio block captured.

        Parameters:
        indata (numpy.ndarray): The recorded audio data.
        frames (int): Number of frames in the audio buffer.
        time (CData): Time information for the audio.
        status (CallbackFlags): Status flags indicating errors or warnings.
        """
        if status:
            print(status)  # Print any errors or warnings from the audio stream.
        audio_data.append(indata.copy())  # Append the recorded audio data to the list.

    # Start an audio input stream with the specified sample rate and callback function.
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, callback=callback):
        while not stop_event.is_set():  # Keep recording until the stop event is triggered.
            pass  # Do nothing, just keep the loop running.

    print("Recording stopped.")  # Print confirmation when recording stops.

def preprocess_audio(audio_data, sample_rate=16000):
    """
    Normalizes the volume and reduces background noise in the audio signal.

    Parameters:
    audio_data (numpy.ndarray): The raw audio signal as a NumPy array.
    sample_rate (int): The sample rate of the audio (default: 16000 Hz).

    Returns:
    numpy.ndarray: The processed audio signal with reduced noise.
    """

    # Normalize the audio to ensure values are between -1 and 1
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Reduce background noise using the "noisereduce" library
    return nr.reduce_noise(y=audio_data.flatten(), sr=sample_rate)

def save_audio(audio_filename, audio_data, sample_rate=16000):
    """
    Saves the processed audio as a WAV file.

    Parameters:
    audio_filename (str): The name of the output audio file (e.g., "lecture1.wav").
    audio_data (numpy.ndarray): The processed audio data as a NumPy array.
    sample_rate (int): The sample rate of the audio (default: 16000 Hz).

    Returns:
    None
    """

    # Define the full file path where the audio will be saved
    file_path = rf"D:\Studies Material\OLA Project\audio_files\{audio_filename}"

    # Convert floating-point audio data (-1 to 1) to 16-bit PCM format (range: -32768 to 32767)
    int_audio_data = (audio_data * 32767).astype(np.int16)

    # Save the audio as a WAV file using scipy's wavfile.write
    wav.write(file_path, sample_rate, int_audio_data)

def transcribe_audio(whisper_model, file_path):
    """
    Transcribes speech from an audio file using the Whisper model.

    Parameters:
    whisper_model (whisper.Whisper): The preloaded Whisper model used for transcription.
    file_path (str): The path to the audio file to be transcribed.

    Returns:
    str: The transcribed text from the audio file.
    """

    # Use the Whisper model to transcribe the given audio file
    result = whisper_model.transcribe(file_path)

    # Extract and return only the transcribed text from the result
    return result['text']

# Function to extract keywords from a heading by removing numbering
def extract_keyword(title):
    """
    Extracts the main keyword from a heading by removing numbering before a colon.
    
    Example:
        Input: "1.2. Introduction:"
        Output: "Introduction"
    """
    match = re.search(r"\d+\.\d*\.?\s*(.*?):", title)  # Extract text before ":"
    return match.group(1).strip() if match else title.strip()

# Function to find synonyms for a given word using WordNet
def get_synonyms(word):
    """
    Retrieves a list of synonyms for a given word using WordNet.
    
    Example:
        Input: "happy"
        Output: ["glad", "joyful", "pleased", ...]
    """
    synonyms = set()
    for syn in wordnet.synsets(word):  # Get all synsets of the word
        for lemma in syn.lemmas():  # Iterate over lemmas (different word forms)
            synonyms.add(lemma.name().lower().replace("_", " "))  # Convert to readable format
    return list(synonyms)

# Function to generate abbreviations for a given term
def generate_abbreviation(term):
    """
    Generates an abbreviation by taking the first letter of each word in the term.
    Common stopwords are ignored.

    Example:
        Input: "Natural Language Processing"
        Output: ["NLP"]
    """
    stopwords = {"of", "and", "the", "in", "for", "on", "at", "to", "with", "by", "an", "a"}
    words = re.split(r'[\s\-]', term)  # Split words by spaces and hyphens
    words = [w for w in words if w.isalpha()]  # Keep only alphabetic words

    if not words:
        return []

    # Extract uppercase first letters for abbreviation (if present)
    abbreviation = "".join([w[0] for w in words if w.isupper()])
    
    # If no uppercase abbreviation found, use all first letters except stopwords
    if not abbreviation:
        abbreviation = "".join([w[0].upper() for w in words if w.lower() not in stopwords])

    return [abbreviation] if len(abbreviation) >= 2 else []

# Class to manage section headings and associated text
class SectionManager:
    def __init__(self, heading_data, model, tokenizer):
        """
        Initializes a section manager that processes headings and subheadings from a structured data format.

        Parameters:
        heading_data (dict): Contains "headings" with their respective texts and subheadings.
        model (T5ForConditionalGeneration): Fine-tuned T5 model.
        tokenizer (T5Tokenizer): Tokenizer for T5 model.
        """
        self.section_labels = []  # List of section names
        self.labels_keywords = []  # List of synonyms and abbreviations for each section
        self.section_texts = {}  # Dictionary to store section texts
        self.section_embeddings = {}  # Dictionary to store section embeddings
        self.model = model  # Store model
        self.tokenizer = tokenizer  # Store tokenizer

        # Process each heading in the document
        for heading_info in heading_data["headings"]:
            heading = extract_keyword(heading_info["heading"])  # Extract cleaned heading
            explanation = self.generate_explanation(heading)  # Generate explanation
            combined_text = heading_info["text"] + " " + explanation  # Combine explanation with slide text
            self.add_section(heading, combined_text)  # Add to section manager

            # Process each subheading under the current heading
            for subheading_info in heading_info["subheadings"]:
                subheading = extract_keyword(subheading_info["subheading"])
                explanation = self.generate_explanation(subheading)  # Generate explanation
                combined_text = subheading_info["text"] + " " + explanation  # Combine explanation with slide text
                self.add_section(subheading, combined_text)  # Add to section manager

    def generate_explanation(self, text, max_length=120):
        """
        Uses the fine-tuned T5 model to generate an explanation for a given text.
        Cleans the explanation by removing prefixes and ensuring complete sentences.
        """
        input_ids = self.tokenizer("explain: " + text, return_tensors="pt").input_ids
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=8, no_repeat_ngram_size=3, repetition_penalty=2.0)
        
        explanation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove unnecessary phrases like "explanation of this is:"
        explanation = self.clean_explanation(explanation)
        
        # Ensure no incomplete sentences at the end
        explanation = self.remove_incomplete_sentence(explanation)
        
        return explanation

    def clean_explanation(self, explanation):
        """
        Removes unnecessary prefixes like 'Generated Explanation:' and handles specific 
        cases like headings followed by colons (e.g., "Operating Systems:").
        """
        # Remove unwanted prefix
        unwanted_prefixes = ["generated explanation:", "explanation:", "explanation of:"]
        for prefix in unwanted_prefixes:
            if explanation.lower().startswith(prefix):
                explanation = explanation[len(prefix):].strip()

        # Remove heading if it is followed by a colon (e.g., "Operating Systems:")
        # Remove the part before and including the first colon
        explanation = re.sub(r'^[^:]+:', '', explanation).strip()

        return explanation.strip()

    def remove_incomplete_sentence(self, explanation):
        """
        Ensures that the explanation does not end with an incomplete sentence.
        It checks if the last sentence is incomplete (i.e., does not end with a punctuation mark).
        """
        sentences = re.split(r'(?<=[.!?]) +', explanation)  # Split based on punctuation marks
    
        if sentences and sentences[-1] != "":
            last_sentence = sentences[-1]
        
            # Check if the last sentence ends with a full stop or similar punctuation
            if last_sentence[-1] not in ".!?":
                sentences = sentences[:-1]  # Remove the last sentence if it is incomplete

        return ' '.join(sentences)

    def add_section(self, label, text):
        """
        Adds a new section along with its synonyms, abbreviations, and initial text.
        Also computes an embedding for the section based on its keywords and text.

        Parameters:
        label (str): The title of the section.
        text (str): The initial text content for the section.
        """
        self.section_labels.append(label)  # Store the section label

        # Generate synonyms and abbreviations for better keyword matching
        synonyms = get_synonyms(label)
        keyword_variants = [label] + synonyms + generate_abbreviation(label)
        self.labels_keywords.append(keyword_variants)

        # Store initial text for the section
        self.section_texts[label] = text

        # Compute embedding using SentenceTransformer
        expanded_text = " ".join(keyword_variants) + " " + text  # Combine keywords and text
        self.section_embeddings[label] = mpnetMODEL.encode(expanded_text, convert_to_tensor=True)

    def update_section(self, label, new_sentence):
        """
        Updates a section by appending a new sentence and recomputing its embedding.

        Parameters:
        label (str): The section label to update.
        new_sentence (str): The new sentence to add to the section.
        """
        if label in self.section_texts:
            self.section_texts[label] += " " + new_sentence  # Append new sentence
            updated_text = " ".join(self.labels_keywords[self.section_labels.index(label)]) + " " + self.section_texts[label]
            self.section_embeddings[label] = mpnetMODEL.encode(updated_text, convert_to_tensor=True)  # Update embedding

# Function to find the most relevant section for each transcribed sentence
def find_relevant_sentences(transcribed_sentences, section_manager):
    """
    Matches each transcribed sentence to the most relevant section using cosine similarity.

    Parameters:
    transcribed_sentences (list): List of sentences from transcribed audio.
    section_manager (SectionManager): The section manager containing precomputed section embeddings.

    Returns:
    dict: A mapping of section labels to their assigned transcribed sentences.
    """
    sentence_embeddings = mpnetMODEL.encode(transcribed_sentences, convert_to_tensor=True)

    # Convert section embeddings into a tensor list
    section_labels = section_manager.section_labels
    section_embeddings = list(section_manager.section_embeddings.values())

    # If no sections exist, return empty mappings
    if len(section_embeddings) == 0:
        return {label: [] for label in section_labels}

    section_embeddings = util.torch.stack(section_embeddings)  # Stack embeddings into tensor format

    # Compute cosine similarity between each sentence and section
    similarities = util.cos_sim(sentence_embeddings, section_embeddings)

    sentence_assignments = {label: [] for label in section_labels}

    # Assign each sentence to the section with the highest similarity score
    for i, sentence in enumerate(transcribed_sentences):
        max_index = similarities[i].argmax().item()  # Get the index of the best-matching section
        best_match = section_labels[max_index]  # Retrieve the best-matching section label

        # Assign sentence to the identified section and update section text
        sentence_assignments[best_match].append(sentence)
        section_manager.update_section(best_match, sentence)

    return sentence_assignments

# Function to merge slide text with transcribed speech
def combine_slide_text_with_audio(slide_content, transcribed_text, model, tokenizer):
    """
    Integrates extracted slide text with transcribed speech by mapping sentences to relevant sections.
    And add explanation to the content.
    
    Parameters:
    slide_content (dict): A structured dictionary containing headings and their respective texts.
    transcribed_text (str): The transcribed audio content.
    model (T5ForConditionalGeneration): Fine-tuned T5 model.
    tokenizer (T5Tokenizer): Tokenizer for T5 model.

    Returns:
    dict: A structured dictionary containing headings with updated text from audio.
    """
    # Tokenize transcribed text into sentences
    transcribed_sentences = [s.strip() for s in sent_tokenize(transcribed_text) if s.strip()]

    # Initialize SectionManager with the model and tokenizer
    section_manager = SectionManager(slide_content, model, tokenizer)

    # Assign transcribed sentences to the most relevant sections
    sentence_assignments = find_relevant_sentences(transcribed_sentences, section_manager)

    # Construct the final combined output structure
    combined_output = {"headings": []}

    for heading_data in slide_content["headings"]:
        heading = heading_data["heading"]
        heading_keyword = extract_keyword(heading)

        heading_entry = {
            "heading": heading,
            "text": section_manager.section_texts.get(heading_keyword, ""),
            "subheadings": []
        }

        for subheading_data in heading_data["subheadings"]:
            subheading = subheading_data["subheading"]
            subheading_keyword = extract_keyword(subheading)

            subheading_entry = {
                "subheading": subheading,
                "text": section_manager.section_texts.get(subheading_keyword, "")
            }

            heading_entry["subheadings"].append(subheading_entry)

        combined_output["headings"].append(heading_entry)

    return combined_output

# Function to generate summary
def generate_summary(explanation_text, max_input_length=512, max_output_length=512):
    # Prepare the input (you can prepend "summarize: " if your model was trained that way)
    input_text = "summarize: " + explanation_text
    input_ids = t5_summarization_tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

    # Generate summary
    summary_ids = t5_summarization_model.generate(
        input_ids,
        max_length=max_output_length,
        num_beams=8,
        no_repeat_ngram_size=3,
        repetition_penalty=2.5,
        length_penalty=1.0
    )

    # Decode summary
    summary = t5_summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = clean_summary_text(summary)
    return summary

# Cleaning function
def clean_summary_text(summary):
    """
    Cleans the generated summary:
    - Removes unwanted prefixes like "In summary,"
    - Ensures the first letter is capitalized
    - Removes any incomplete sentence at the end
    """
    # Remove unwanted prefix
    unwanted_prefixes = ["in summary,", "in conclusion,", "in short,", "summary:", "conclusion:"]
    for prefix in unwanted_prefixes:
        if summary.lower().startswith(prefix):
            summary = summary[len(prefix):].strip()
            if summary:
                summary = summary[0].upper() + summary[1:]

    # Remove incomplete sentence at the end
    sentences = re.split(r'(?<=[.!?]) +', summary)
    if sentences and sentences[-1] != "":
        last_sentence = sentences[-1]
        if last_sentence[-1] not in ".!?":
            sentences = sentences[:-1]
    cleaned_summary = ' '.join(sentences)

    return cleaned_summary

# Generating final notes
def generate_notes(content):
    final_notes = {"headings": []}

    for heading_data in content["headings"]:
        heading = heading_data["heading"]

        # Get and summarize heading text
        heading_text = heading_data["text"]
        summarized_heading_text = generate_summary(heading_text)

        heading_entry = {
            "heading": heading,
            "text": summarized_heading_text,
            "subheadings": []
        }

        for subheading_data in heading_data["subheadings"]:
            subheading = subheading_data["subheading"]

            sub_text = subheading_data["text"]
            summarized_sub_text = generate_summary(sub_text)

            subheading_entry = {
                "subheading": subheading,
                "text": summarized_sub_text
            }

            heading_entry["subheadings"].append(subheading_entry)

        final_notes["headings"].append(heading_entry)

    return final_notes
    
def print_notes(notes):
    for heading_data in notes["headings"]:
        heading_text = heading_data["text"].strip()
        print(f"\033[1m{heading_data['heading']}\033[0m")  # Bold heading
        if heading_text:
            print(f"  {heading_text}\n")
        
        for subheading_data in heading_data["subheadings"]:
            subheading_text = subheading_data["text"].strip()
            print(f"    \033[94m{subheading_data['subheading']}\033[0m")  # Blue subheading
            if subheading_text:
                print(f"      {subheading_text}\n")

class LectureNotesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lecture Notes Generator")
        self.attributes('-fullscreen', True)  # Start in fullscreen
        
        # Set custom theme
        self.style = ttk.Style()
        self.style.theme_create("lecture_theme", parent="clam", settings={
            "TFrame": {"configure": {"background": "#f5f7fa"}},
            "TLabelFrame": {"configure": {
                "background": "#f5f7fa",
                "foreground": "#2c3e50",
                "bordercolor": "#dfe6e9",
                "relief": "groove",
                "labelmargins": (10, 5)
            }},
            "TLabel": {"configure": {
                "background": "#f5f7fa",
                "foreground": "#2c3e50",
                "font": ("Segoe UI", 10)
            }},
            "TButton": {"configure": {
                "background": "#3498db",
                "foreground": "white",
                "font": ("Segoe UI", 10, "bold"),
                "borderwidth": 1,
                "relief": "raised",
                "padding": (10, 5)
            }, "map": {
                "background": [("active", "#2980b9"), ("disabled", "#bdc3c7")],
                "foreground": [("disabled", "#7f8c8d")]
            }},
            "TEntry": {"configure": {
                "fieldbackground": "white",
                "foreground": "#2c3e50",
                "insertcolor": "#2c3e50",
                "font": ("Segoe UI", 10)
            }},
            "Vertical.TScrollbar": {"configure": {
                "arrowsize": 14,
                "troughcolor": "#ecf0f1",
                "background": "#bdc3c7"
            }}
        })
        self.style.theme_use("lecture_theme")
        
        # Initialize core variables
        self.recording = False
        self.audio_data = []
        self.slide_texts = []
        self.current_slide = 0
        self.slides = []
        self.audio_files = []
        self.whisper_model = load_whisper_model()
        
        # Create UI elements
        self.create_widgets()
        self.create_menu()
        self.create_bindings()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        self.lbl_title = ttk.Label(header_frame, 
                                 text="Lecture Notes Generator", 
                                 font=("Segoe UI", 16, "bold"),
                                 foreground="#2c3e50")
        self.lbl_title.pack(side=tk.LEFT)

        # PDF Upload Section
        upload_frame = ttk.LabelFrame(main_frame, text="PDF Processing")
        upload_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        upload_btn_frame = ttk.Frame(upload_frame)
        upload_btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_upload = ttk.Button(upload_btn_frame, text="Upload PDF", command=self.load_pdf)
        self.btn_upload.pack(side=tk.LEFT, padx=5)
        
        self.lbl_pdf_status = ttk.Label(upload_btn_frame, text="No PDF loaded", foreground="#7f8c8d")
        self.lbl_pdf_status.pack(side=tk.LEFT, padx=10)

        # Slide Display
        self.slide_frame = ttk.LabelFrame(main_frame, text="Current Slide")
        self.slide_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        self.slide_container = ttk.Frame(self.slide_frame)
        self.slide_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.slide_label = ttk.Label(self.slide_container)
        self.slide_label.pack()

        # Slide Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        self.lbl_slide_count = ttk.Label(nav_frame, text="Slide: 0/0", foreground="#2c3e50")
        self.lbl_slide_count.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = ttk.Button(nav_frame, text="Start Recording", 
                                  command=self.start_recording, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_next = ttk.Button(nav_frame, text="Next Slide", 
                                 command=self.handle_next_slide, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = ttk.Label(nav_frame, text="Status: Ready", foreground="#27ae60")
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        # Results Display
        result_frame = ttk.LabelFrame(main_frame, text="Generated Notes")
        result_frame.grid(row=1, column=1, rowspan=3, sticky="nsew", padx=5, pady=5)
        
        self.txt_notes = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            font=("Segoe UI", 10),
            padx=10, 
            pady=10,
            bg="white",
            fg="#2c3e50",
            insertbackground="#3498db",
            selectbackground="#3498db",
            selectforeground="white"
        )
        self.txt_notes.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=0)

    def create_menu(self):
        menubar = tk.Menu(self, bg="#ecf0f1", fg="#2c3e50", activebackground="#3498db", activeforeground="white")
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg="#ecf0f1", fg="#2c3e50", 
                          activebackground="#3498db", activeforeground="white")
        file_menu.add_command(label="Open PDF", command=self.load_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg="#ecf0f1", fg="#2c3e50",
                          activebackground="#3498db", activeforeground="white")
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)

    def show_about(self):
        about_window = tk.Toplevel(self)
        about_window.title("About Lecture Notes Generator")
        about_window.geometry("400x200")
        about_window.resizable(False, False)
        
        about_frame = ttk.Frame(about_window)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(about_frame, text="Lecture Notes Generator", font=("Segoe UI", 14, "bold")).pack(pady=5)
        ttk.Label(about_frame, text="Version 1.0").pack(pady=5)
        ttk.Label(about_frame, text="Automatically generate lecture notes from slides and audio").pack(pady=5)
        ttk.Label(about_frame, text="© 2023 Your Name").pack(pady=5)
        
        btn_close = ttk.Button(about_frame, text="Close", command=about_window.destroy)
        btn_close.pack(pady=10)

    def create_bindings(self):
        self.bind("<Escape>", lambda e: self.destroy())

    def load_pdf(self):
        pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if pdf_path:
            try:
                self.lbl_status.config(text="Processing PDF...", foreground="#f39c12")
                self.update()
                
                self.slides = convert_from_path(pdf_path)
                self.current_slide = 0
                self.show_slide()
                self.btn_start['state'] = tk.NORMAL
                self.btn_next['state'] = tk.NORMAL
                self.update_slide_count()
                self.txt_notes.delete(1.0, tk.END)
                self.lbl_pdf_status.config(text=f"Loaded: {pdf_path.split('/')[-1]}", foreground="#27ae60")
                self.lbl_status.config(text="Status: Ready", foreground="#27ae60")
            except Exception as e:
                self.lbl_status.config(text=f"Error: {str(e)}", foreground="#e74c3c")
                self.lbl_pdf_status.config(text="Failed to load PDF", foreground="#e74c3c")

    def show_slide(self):
        if self.slides:
            slide = self.slides[self.current_slide]
            # Maintain aspect ratio while fitting within the frame
            width, height = slide.size
            max_width = self.winfo_screenwidth() // 2
            max_height = self.winfo_screenheight() - 200
            ratio = min(max_width/width, max_height/height)
            new_size = (int(width * ratio), int(height * ratio))
            
            image = ImageTk.PhotoImage(slide.resize(new_size))
            self.slide_label.config(image=image)
            self.slide_label.image = image

    def update_slide_count(self):
        if self.slides:
            total = len(self.slides)
            current = self.current_slide + 1
            self.lbl_slide_count.config(text=f"Slide: {current}/{total}")

    def handle_next_slide(self):
        if self.recording:
            self.stop_recording()
            # Continue to next slide after processing is complete
            if self.current_slide < len(self.slides) - 1:
                self.current_slide += 1
                self.show_slide()
                self.update_slide_count()
                self.btn_start['state'] = tk.NORMAL
        else:
            if self.current_slide < len(self.slides) - 1:
                self.current_slide += 1
                self.show_slide()
                self.update_slide_count()
                self.btn_start['state'] = tk.NORMAL
            else:
                self.lbl_status.config(text="No more slides", foreground="#e74c3c")

    def start_recording(self):
        self.recording = True
        self.btn_start['state'] = tk.DISABLED
        self.lbl_status['text'] = "Recording... Press Next Slide to stop"
        self.lbl_status['foreground'] = "#e74c3c"
        self.stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def stop_recording(self):
        if self.recording:
            self.stop_event.set()
            self.audio_thread.join()
            self.lbl_status['text'] = "Processing audio..."
            self.lbl_status['foreground'] = "#f39c12"
            self.update()
            
            self.process_audio()
            self.recording = False
            
            if self.current_slide < len(self.slides) - 1:
                self.lbl_status['text'] = "Ready for next slide"
                self.lbl_status['foreground'] = "#27ae60"
            else:
                self.process_transcription()
                self.btn_start['state'] = tk.DISABLED
                self.lbl_status['text'] = "Processing complete!"
                self.lbl_status['foreground'] = "#27ae60"

    def record_audio(self):
        self.audio_data = []
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.append(indata.copy())
        
        with sd.InputStream(samplerate=16000, channels=1, dtype=np.float32, callback=callback):
            while not self.stop_event.is_set():
                pass

    def process_audio(self):
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            processed_audio = preprocess_audio(audio_array)
            audio_filename = f"slide_{self.current_slide}.wav"
            save_audio(audio_filename, processed_audio)
            self.audio_files.append(audio_filename)
            
            # Process slide text
            slide_text = extract_text_from_slide(self.slides[self.current_slide], self.current_slide + 1)
            self.slide_texts.append(slide_text)

    def process_transcription(self):
        self.lbl_status['text'] = "Transcribing audio..."
        self.lbl_status['foreground'] = "#f39c12"
        self.update()
        
        with ThreadPoolExecutor(max_workers=min(torch.cuda.device_count(), 4)) as executor:
            transcribed_texts = list(executor.map(self.transcribe_audio_wrapper, self.audio_files))
        
        self.txt_notes.delete(1.0, tk.END)
        for i, transcribed_text in enumerate(transcribed_texts):
            combined_text = combine_slide_text_with_audio(self.slide_texts[i], transcribed_text, t5model, t5tokenizer)
            notes = generate_notes(combined_text)
            self.display_notes(notes)
        
        self.lbl_status['text'] = "Transcription complete!"
        self.lbl_status['foreground'] = "#27ae60"

    def transcribe_audio_wrapper(self, audio_filename):
        file_path = rf"D:\Studies Material\OLA Project\audio_files\{audio_filename}"
        return transcribe_audio(self.whisper_model, file_path)

    def display_notes(self, notes):
        # Configure text tags for styling
        self.txt_notes.tag_configure("heading", 
                                   font=('Segoe UI', 12, 'bold'), 
                                   foreground="#2c3e50",
                                   spacing1=10, spacing3=5)
        self.txt_notes.tag_configure("subheading", 
                                   font=('Segoe UI', 10, 'bold'), 
                                   foreground="#3498db",
                                   spacing1=5, spacing3=3)
        self.txt_notes.tag_configure("normal", 
                                   font=('Segoe UI', 10),
                                   foreground="#2c3e50")
        
        for heading_data in notes["headings"]:
            heading_text = heading_data["text"].strip()
            self.txt_notes.insert(tk.END, f"{heading_data['heading']}\n", "heading")
            if heading_text:
                self.txt_notes.insert(tk.END, f"{heading_text}\n\n", "normal")
            
            for subheading_data in heading_data["subheadings"]:
                subheading_text = subheading_data["text"].strip()
                self.txt_notes.insert(tk.END, f"  {subheading_data['subheading']}\n", "subheading")
                if subheading_text:
                    self.txt_notes.insert(tk.END, f"    {subheading_text}\n\n", "normal")

if __name__ == "__main__":
    app = LectureNotesApp()
    app.mainloop()