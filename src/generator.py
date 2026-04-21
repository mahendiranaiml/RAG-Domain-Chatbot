from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod
import yaml

import logging 
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s - %(module)s:%(lineno)d"
)
logger = logging.getLogger(__name__)


class AnswerGenerator(ABC):

    @abstractmethod
    def generator(self, query, final_pages_for_llm):
        pass



class LLMGenerator(AnswerGenerator):

    def __init__(self, llm):
        self.llm = llm
        # 1. Setup the strict prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer ONLY using the provided context below.
                        - Be concise and factual.
                        - You may give structred answer with respect to the context.
                        - If the answer is not in the context, say:
                        'This information is not available in the document.'
                        - Use table rows exactly when the question is about tables.
                        - Use image text only if the context explicitly contains an image block.
                        - Prefer the exact title or heading text when asked.
                        - Mention page numbers when they are available in the context.
                        Context:
                        {context}"""),
            ("human", "{input}"),
        ])

        # 2. Initialize the chain
        self.chain = create_stuff_documents_chain(self.llm, prompt)
        

    def generator(self, query, final_pages_for_llm):

        # 4. Generate the answer
        response = self.chain.invoke({
            "input": query,
            "context": final_pages_for_llm
        })
        logger.info("Answer Generated ")
        return response

def generate_response(query, llm, final_pages_for_llm):
    obj = LLMGenerator(llm)
    response = obj.generator(query, final_pages_for_llm)
    
    return response
