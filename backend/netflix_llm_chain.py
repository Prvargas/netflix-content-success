#The purpose of this file is to contain the class object that facilitates running LLM chain calls to enhance Netflix engagement data.


from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
#PHIL #1: Added runnables from langchain for LCEL use DID NOT WORK
from langchain_core.runnables import RunnablePassthrough
#PHIL #2: Added LLMChain to overcome LCEL | issue. Just explicitly call in this langchain function
from langchain.chains import LLMChain

class Netflix_Few_Shot_Class:
    """
    This class is designed to facilitate the creation and execution of few-shot learning prompts
    using the LangChain library, specifically tailored for determining details about Netflix titles.

    It encapsulates the process of defining a prompt template, setting up few-shot examples,
    initializing a language learning model (LLM), and parsing the output.

    Methods:
    - create_example_prompt(template, input_variables): Defines the prompt template with placeholders for input variables.
    - create_few_shot_examples(examples_list): Sets up few-shot examples based on the provided list.
    - load_llm_model(llm_model): Loads the specified LLM model for generating responses.
    - load_parser_object(): Initializes an output parser for parsing the response from the LLM.
    - chain_components(): Chains the few-shot prompt, LLM model, and output parser together for execution.
    - run_chain(input_question): Executes the chained components with the given input question, returning the LLM's response.

    Usage:
    1. Initialize the class with an API key for OpenAI.
    2. Call `create_example_prompt` with a template string and list of input variables.
    3. Call `create_few_shot_examples` with a list of examples matching the template structure.
    4. Create a `ChatOpenAI` object with your API key and preferred model, then pass it to `load_llm_model`.
    5. Call `load_parser_object` to initialize the output parser.
    6. Call `chain_components` to prepare the execution pipeline.
    7. Use `run_chain` with a specific question to get the processed response from the LLM.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.netflix_example_prompt = None
        self.few_shot_prompt = None
        self.llm_model = None
        self.output_parser = StrOutputParser() #PHIL #3: Already have parser read to avoid chain error
        self.parser_chain = None

    def create_example_prompt(self, template, input_variables):
        self.netflix_example_prompt = PromptTemplate(template=template, input_variables=input_variables)

    def create_few_shot_examples(self, examples_list):
        print(f'TOTAL FEW-SHOT EXAMPLES: {len(examples_list)}\n\n')
        self.few_shot_prompt = FewShotPromptTemplate(
            examples=examples_list,
            example_prompt=self.netflix_example_prompt,
            suffix="Question: {input}",
            input_variables=["input"]
        )

    def load_llm_model(self, llm_model):
        self.llm_model = llm_model

    def load_parser_object(self):
        self.output_parser = StrOutputParser()

    def chain_components(self):
        #NOTE: Can NOT use LCEL | notation in class object
        #PHIL #2: Input parameters to create the object are “llm” (LLM Model to call) and “prompt” (Prompt object to use)
        chain = LLMChain(llm=self.llm_model, prompt=self.few_shot_prompt, output_parser=self.output_parser )

        #self.parser_chain = self.few_shot_prompt | self.llm_model | self.output_parser
        self.parser_chain = chain


    def run_chain(self, input_question):
        return self.parser_chain.invoke({"input": input_question})

    #PHIL #3: THIS THE FUNCTION TO USE ON DATAFRAME TO RETURN ONLY TEXT
    def run_chain_text_only(self, input_question):
        return self.parser_chain.invoke({"input": input_question})['text']


