import os 
import dspy
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dspy.retrieve.faiss_rm import FaissRM
import time


class ragSignature(dspy.Signature):
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1000 words")

class RAG(dspy.Module):

    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(ragSignature)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

def prepare_data():

    load_dotenv()
    # Load the pdf documents
    loader = PyPDFDirectoryLoader("data/")
    docs = [doc.page_content for doc in loader.load()]

    # Langchain text splitter 
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=20000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    # Extract content to pass in text splitter 
    docs = text_splitter.create_documents(docs)

    # Extract the document part from list of document objects returned by text splitter
    docs_str = [doc.page_content for doc in docs]

    # Define FAISS retriver 
    from dspy.retrieve.faiss_rm import FaissRM
    frm = FaissRM(docs_str)

    # Configure generation model and retriver model for dspy
    # colbertv2_wiki17_abstracts = dspy.ColBERTv2(docs)
    turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=1000)
    dspy.settings.configure(lm=turbo, rm=frm)

def get_response(question):
    # my_question = "what are the differences between Pru health critical illness extended care III and first protector II"
    pred = RAG(10)
    prediction = pred.forward(question)
    for word in prediction.answer.split():
        yield word + ' '
        time.sleep(0.01)


