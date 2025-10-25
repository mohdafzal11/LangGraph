from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser ,  PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=GEMINI_API_KEY)

class OutputSchema(BaseModel):
    feedback: str = Field(..., description="Detailed feedback on the essay.")
    score: float = Field(..., ge=0, le=10, description="Score between 0 and 10.")

object_parser = PydanticOutputParser(pydantic_object=OutputSchema)
str_parser = StrOutputParser()

clt_prompt = PromptTemplate(
    input_variables=["content"],
    partial_variables= {"format_instructions":object_parser.get_format_instructions()},
    template="Provide detailed feedback on the following essay content focusing on clarity of thoughts. {format_instructions}\n\nEssay Content:\n{content}"
    )

doa_prompt = PromptTemplate(
    input_variables=["content"],
    partial_variables= {"format_instructions":object_parser.get_format_instructions()},
    template="Provide detailed feedback on the following essay content focusing on depth of the topic anaysis. {format_instructions}\n\nEssay Content:\n{content}"
    )

lang_prompt = PromptTemplate(
    input_variables=["content"],
    partial_variables= {"format_instructions":object_parser.get_format_instructions()},
    template="Provide detailed feedback on the following essay content focusing on language. {format_instructions}\n\nEssay Content:\n{content}"
    )

summary_prompt = PromptTemplate(
    input_variables=['clt_feedback' , 'doa_feedback' , 'lang_feedback'],
    template="Provide detailed feedback on the following feedbacks clearity of thoghts feedback , depth of topic feedback , language feedback. Clearity of thougts {clt_feedback}\n\nDepth of analysis:\n{doa_feedback}\n'n Language Feedback\n {lang_feedback}"
    )


class EssayState(TypedDict):
    content:str
    clt_feedback:str
    clt_score:float
    doa_feedback:str
    doa_score:float
    lang_feedback:str
    lang_score:float
    summary:str
    final_score:float
    
    
graph = StateGraph(EssayState)


def clearity_of_thoughts(EssayState):
    content = EssayState['content']
    chain = clt_prompt | llm | object_parser
    result = chain.invoke({'content':content})
    return {
        'clt_feedback':result.feedback,
         'clt_score':result.score
    }
    
def depth_of_analysis_topic(EssayState):
    content = EssayState['content']
    chain = doa_prompt | llm | object_parser
    result = chain.invoke({'content':content})
    return {
        'doa_feedback':result.feedback,
         'doa_score':result.score
    }
    
def language(EssayState):
    content = EssayState['content']
    chain = lang_prompt | llm | object_parser
    result = chain.invoke({'content':content})
    return {
        'lang_feedback':result.feedback,
         'lang_score':result.score
    }
    
def summary(EssayState):        
    chain = summary_prompt | llm | str_parser
    result = chain.invoke({
         'clt_feedback':EssayState['clt_feedback'],
         'doa_feedback':EssayState['doa_feedback'],
         'lang_feedback':EssayState['lang_feedback'],
    })
    print("Final Result"  , result)
    return{
        'summary' : result,
        'final_score':EssayState['clt_score'] + EssayState['doa_score'] + EssayState['lang_score']
    }
    

graph.add_node("clearity_of_thoughts", clearity_of_thoughts)
graph.add_node("depth_of_analysis_topic", depth_of_analysis_topic)
graph.add_node("language", language)
graph.add_node("summary" , summary)
graph.add_edge(START , 'clearity_of_thoughts')
graph.add_edge(START , 'depth_of_analysis_topic')
graph.add_edge(START , 'language')
graph.add_edge('clearity_of_thoughts' , 'summary')
graph.add_edge('depth_of_analysis_topic' , 'summary')
graph.add_edge('language' , 'summary')
graph.add_edge('summary' , END)

content = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

workflow = graph.compile()
final_result = workflow.invoke({"content":content})
print("Summary" , final_result['summary'])
print("Score" , final_result['final_score'])




