from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub
import os
from dataloader import load_high
from agentic_chunker import AgenticChunker
from langchain_ollama import ChatOllama

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]


def get_propositions(text, runnable, extraction_chain):
    runnable_output = runnable.invoke({
    	"input": text
    }).content

    print("Runnable output:     ", runnable_output)
    
    # propositions = extraction_chain.invoke(runnable_output)[0].sentences
    propositions = extraction_chain.invoke(runnable_output)
    print("Propositions:     ", propositions)
    return propositions

def run_chunk(essay):

    obj = hub.pull("wfh/proposal-indexing")
    # llm = ChatOpenAI(model='gpt-4-1106-preview', openai_api_key = os.getenv("OPENAI_API_KEY"))
    llm = ChatOllama(model='llama3.2')

    runnable = obj | llm

    # Extraction
    # extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
    extraction_chain = llm.with_structured_output(Sentences)

    paragraphs = essay.split("\n\n")

    essay_propositions = []

    for i, para in enumerate(paragraphs):
        # print("I:", i, "'\n Para: ", para, "\n runnable: ", runnable, "\n extraction_chain: ", extraction_chain)
        propositions = get_propositions(para, runnable, extraction_chain)
        
        essay_propositions.extend(propositions)
        print (f"Done with {i}")

    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    print("Chunk:      ", chunks)
    return chunks
    print(chunks)