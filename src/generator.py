from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod


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
                        - If the answer is not in the context, say: 
                        'This information is not available in the document.'
                        - Never reference chapter numbers, page numbers, or document structure.: \n\n{context}"""),
            ("human", "{input}"),
        ])

        # 2. Initialize the chain
        self.chain = create_stuff_documents_chain(self.llm, prompt)
        

    def generator(self, query, final_pages_for_llm):

        # 4. Generate the answer
        response = self.chain.stream({
            "input": query,
            "context": final_pages_for_llm
        })

        return response

def generate(query, llm, final_pages_for_llm):
    obj = LLMGenerator(llm)
    response = obj.generator(query, final_pages_for_llm)
    
    return response
