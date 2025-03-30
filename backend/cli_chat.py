import os
import sys
import dotenv
from test import whisper_test
import uuid

dotenv.load_dotenv()
#extra

# Constants
VIDEO_PATH = 'test/video/test_video.mp4'
OUTPUT_DIR = 'test/chunks/'
MODEL_SIZE = 'base'
RAG_FILES = ['FAKED'] #file pathway to pdfs that contribute to RAG

def create_http_client():
    from llama_stack_client import LlamaStackClient
    from llama_stack_client import RAGDocument #import RAG utilities

    return LlamaStackClient(
        base_url=f"http://llama-stack:{os.getenv('LLAMA_STACK_PORT')}"
    )

client = create_http_client()

# List available models
models = client.models.list()
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

model_id = os.getenv("INFERENCE_MODEL")
if not model_id:
    print("INFERENCE_MODEL not set in .env")
    sys.exit(1)

#setting up a RAG system
#creates RAG documents
documents = [
    RAGDocument(
        document_id=f"num-{i}",
        content=file,
        mime_type="application/pdf",
        metadata={},
    )
    for i, file in enumerate(RAG_FILES)
]

#create vector database
vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
)

#makes the http client insert the documents, having been
#vectorized in the database. chunking occurs automatically.
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)

#create an agent that can respond to prompts
rag_agent = Agent(
    client,
    model=model_id,
    instructions="""You are a AI that analyzes incidents in bodycam transcriptions.
        You will be given audio transcript files representing segments of
        a single event. Using the documents provided to you,
        recognize events that constitute conflict or altercations,
        summarize them, and flag them for manual review for each transcript and audio file. There may be multiple
        incidents in one transcript. Each transcript is marked by a new line and the name of the video file.
        if the given file and its transcript contains nothing of note, say
        that there is nothing of note.""",
    tools = [
        {
          "name": "builtin::rag/knowledge_search",
          "args" : {
            "vector_db_ids": [vector_db_id],
          }
        }
    ],
)

#create agent session
session_id = rag_agent.create_session("test-session")

#CURRENTLY HARDCODES VIDEO PATH, OUTPUT DIRECTORY, AND MODEL_SIZE
#transcribe all chunks of videos, and pushes transcriptions + file to chunk file in array of tuples
total_transcripts = whisper_test.get_transcriptions(VIDEO_PATH,
                                                    OUTPUT_DIR,
                                                    MODEL_SIZE)

#stick in the path to the audio file and the transcription as prompts to the agent
for audio_path, text in total_transcripts:
    prompt = audio_path + "\n" + text
    cprint(f'User> {prompt}', 'green')
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
        log.print()