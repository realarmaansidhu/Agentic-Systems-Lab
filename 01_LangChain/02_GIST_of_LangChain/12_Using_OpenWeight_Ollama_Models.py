# LangChain code to implement a text summarization chain using the Ollama API to access a language model. The code uses the PromptTemplate class from LangChain to create prompts for summarization and the ChatOllama class to interact with the Ollama API.

import os
import langchain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama as Ollama
from dotenv import load_dotenv

load_dotenv()

information = """
    Jeffrey Preston Bezos (/ˈbeɪzoʊs/ BAY-zohss;[2] né Jorgensen; born January 12, 1964) is an American businessman best known as the founder, executive chairman, and former president and CEO of Amazon, the world's largest e-commerce and cloud computing company. According to Forbes, as of December 2025, Bezos's estimated net worth is US$239.4 billion, making him the fourth richest person in the world.[3] He was the wealthiest person from 2017 to 2021, according to Forbes and the Bloomberg Billionaires Index.[4]

    Bezos was born in Albuquerque, and raised in Houston and Miami. He graduated from Princeton University in 1986 with a degree in engineering. He worked on Wall Street in a variety of related fields from 1986 to early 1994. Bezos founded Amazon in mid-1994 on a road trip from New York City to Seattle. The company began as an online bookstore and has since expanded to a variety of other e-commerce products and services, including video and audio streaming, cloud computing, and artificial intelligence. It is the world's largest online sales company, the largest Internet company by revenue, and the largest provider of virtual assistants and cloud infrastructure services through its Amazon Web Services branch.

    Bezos founded the aerospace manufacturer and sub-orbital spaceflight services company Blue Origin in 2000. Blue Origin's New Shepard vehicle reached space in 2015 and afterwards successfully landed back on Earth; he flew into space on Blue Origin NS-16 in 2021. He purchased the major American newspaper The Washington Post in 2013 for $250 million (equivalent to $337,464,286 in 2024) and manages many other investments through his venture capital firm, Bezos Expeditions. In September 2021, Bezos co-founded Altos Labs with Mail.ru founder Yuri Milner.[5]

    The first centibillionaire on the Forbes Real Time Billionaires Index and the second ever to have achieved the feat since Bill Gates in 1999, Bezos was named the "richest man in modern history" after his net worth increased to $150 billion in July 2018 (equivalent to $187,827,988,338 in 2024).[6] In August 2020, according to Forbes, he had a net worth exceeding $200 billion (equivalent to $242,998,585,573 in 2024). On July 5, 2021, Bezos stepped down as the CEO and president of Amazon and took over the role of executive chairman. Amazon Web Services CEO Andy Jassy succeeded Bezos as the CEO and president of Amazon
    """

summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

llm = Ollama(model="llama3.2:latest", temperature=0, verbose=True)

summary_template = PromptTemplate(input_variables=["information"], template=summary_template)

chain = summary_template | llm
results = chain.invoke(input={"information": information})
print(results.content)