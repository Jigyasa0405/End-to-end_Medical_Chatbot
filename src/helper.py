from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import whisper

# Extract data from pdf file
def load_pdf_file(data):
    loader= DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Split yhe data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings



def record_audio(filename="mic_input.wav", duration=5, fs=44100):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write_wav(filename, fs, audio)
    print("✅ Recording complete.")
    return filename

def transcribe_audio(filename):
    print("🧠 Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    text = result["text"]
    print("🗣️ You said:", text)
    return text
