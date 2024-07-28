from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase
import os
from crewai_tools import SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
# Uncomment the following line to use an example of a custom tool
# from assesment.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool
search_tool = SerperDevTool()
web_search_tool = WebsiteSearchTool()

os.environ["OPENAI_API_KEY"]="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"
os.environ["OPENAI_MODEL_NAME"]="llama3-8b-8192"
os.environ["OPENAI_API_BASE"]="https://api.groq.com/openai/v1"
os.environ["SERPER_API_KEY"]="5a3be95d516df6071c1fd20396a1ca52ddd319c5"
os.environ["MISTRAL_API_KEY"]="uQ7WKLxACytIiddduZ2JWwNcLH3qyuIP"
print("this is from crew")

@CrewBase
class AssesmentCrew():
    """Assessment crew"""
    agents_config = 'config/agents.yaml'
    
    def __init__(self, raw_text , email):
        self.raw_text = raw_text
        self.email = email
        self.setup_agents()
        self.setup_tasks()
        self.setup_crew()

    def setup_agents(self):
        self.blood_test_analyst = Agent(
            role='Blood Test Analyst',
            goal='Analyze the blood test report and summarize the findings.',
            backstory='A medical expert specializing in blood test analysis.',
            verbose=True,
            allow_delegation=False
        )

        self.article_researcher = Agent(
            role='Article Researcher',
            goal='Search for health articles based on blood test results.',
            backstory='An expert researcher proficient in finding health-related articles.',
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True,
            allow_delegation=False
        )

        self.health_advisor = Agent(
            role='Health Advisor',
            goal='Provide health recommendations based on the articles found.',
            backstory='A health advisor with extensive knowledge in providing health advice.',
            verbose=True,
            allow_delegation=False
        )

    def setup_tasks(self):
        self.analyze_blood_test_task = Task(
            description=f'You have to analyze the blood test report from "{self.raw_text}" blood report',
            expected_output='A summary of the blood test results.',
            agent=self.blood_test_analyst,
        )

        self.find_articles_task = Task(
            description='Search for health articles based on the blood test analysis.',
            expected_output='A list of relevant health articles with links.',
            agent=self.article_researcher,
            context=[self.analyze_blood_test_task]
        )

        self.provide_recommendations_task = Task(
            description='Provide health recommendations based on the articles found.',
            expected_output='Health recommendations with links to the articles.',
            agent=self.health_advisor,
            context=[self.find_articles_task]
        )

    def setup_crew(self):
        self.crew = Crew(
            agents=[self.blood_test_analyst, self.article_researcher, self.health_advisor],
            tasks=[self.analyze_blood_test_task, self.find_articles_task, self.provide_recommendations_task],
            verbose=2,
            process=Process.sequential,
            embedder={
                 "provider": "mistralai",
                    "api_key": os.environ.get("MISTRAL_API_KEY"),
                    "model_name": "mistral-embed",
             }
        )
        self.crew.kickoff()

#     @agent
#     def data_extractor(self) -> Agent:
#         return Agent(
#             config=self.agents_config['data_extractor'],
#             verbose=True,
#         )

#     @agent
#     def range_finder(self) -> Agent:
#         return Agent(
#             config=self.agents_config['range_finder'],
#             verbose=True
#         )

#     @agent
#     def abnormality_detector(self) -> Agent:
#         return Agent(
#             config=self.agents_config['abnormality_detector'],
#             verbose=True
#         )

#     @agent
#     def email_sender(self) -> Agent:
#         return Agent(
#             config=self.agents_config['email_sender'],
#             verbose=True
#         )

#     @task
#     def extract_metrics_task(self ) -> Task:
#         return Task(
#             config=self.tasks_config['extract_metrics_task'],
#             description=f"""Your a medical data analysis expert with extensive experience in interpreting and extracting information from blood reports. You have a keen eye for detail and can identify key measures and their corresponding values quickly and accurately.

# Your task is to extract all key measures from a provided blood report text. Please return a list of each key measure along with its value presented in the text.

# Here is the blood report text for analysis: {self.pdf_text}

# Be sure to capture any common blood measures such as hemoglobin levels, white blood cell counts, platelet counts, cholesterol levels, and any other relevant parameters that may appear. Structure your response in a clear and organized manner, listing each measure followed by its respective value.

# For example, if the text mentions "Hemoglobin: 13.5 g/dL," your output should explicitly state "Hemoglobin: 13.5 g/dL".
#             """,
#             expected_output='array of metrics',
#             agent=self.data_extractor(),
#         )

#     @task
#     def search_ranges_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['search_ranges_task'],
#             agent=self.range_finder()
#         )

#     @task
#     def find_abnormalities_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['find_abnormalities_task'],
#             agent=self.abnormality_detector()
#         )

#     @task
#     def send_recommendations_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['send_recommendations_task'],
#             agent=self.email_sender()
#         )

#     def __init__(self , pdf_text , email):
#         self.pdf_text = pdf_text
#         self.email = email
#         print(pdf_text ,"this is from crew")
    
    # blood_test_analyst = Agent(
    #     role='Blood Test Analyst',
    #     goal='Analyze the blood test report and summarize the findings.',
    #     backstory='A medical expert specializing in blood test analysis.',
    #     verbose=True,
    #     allow_delegation=False
    # )

    # article_researcher = Agent(
    #     role='Article Researcher',
    #     goal='Search for health articles based on blood test results.',
    #     backstory='An expert researcher proficient in finding health-related articles.',
    #     tools=[search_tool, web_search_tool],
    #     verbose=True,
    #     allow_delegation = False,
    # )

    # health_advisor = Agent(
    #     role='Health Advisor',
    #     goal='Provide health recommendations based on the articles found.',
    #     backstory='A health advisor with extensive knowledge in providing health advice.',
    #     verbose=True,
    #     allow_delegation = False,
    # )

    # # Define Tasks
    # analyze_blood_test_task = Task(
    #     description= f'You have to analyze the blood test report from "{raw_text}" blood report',
    #     expected_output='A summary of the blood test results.',
    #     agent=blood_test_analyst,
    # )

    # find_articles_task = Task(
    #     description='Search for health articles based on the blood test analysis.',
    #     expected_output='A list of relevant health articles with links.',
    #     agent=article_researcher,
    #     context=[analyze_blood_test_task]
    # )

    # provide_recommendations_task = Task(
    #     description='Provide health recommendations based on the articles found.',
    #     expected_output='Health recommendations with links to the articles.',
    #     agent=health_advisor,
    #     context=[find_articles_task]
    # )
    # @crew
    # def crew(self ) -> Crew:
    #     """Creates the Assessment crew"""
    #     return Crew(
    #         agents=[blood_test_analyst, article_researcher, health_advisor],
    #         tasks=[analyze_blood_test_task, find_articles_task, provide_recommendations_task],
    #         verbose=2
    #     )
    
#     agents_config = 'config/agents.yaml'


# #     @agent
# #     def data_extractor(self) -> Agent:
# #         return Agent(
# #             config=self.agents_config['data_extractor'],
# #             verbose=True,
# #         )

# #     @agent
# #     def range_finder(self) -> Agent:
# #         return Agent(
# #             config=self.agents_config['range_finder'],
# #             verbose=True
# #         )

# #     @agent
# #     def abnormality_detector(self) -> Agent:
# #         return Agent(
# #             config=self.agents_config['abnormality_detector'],
# #             verbose=True
# #         )

# #     @agent
# #     def email_sender(self) -> Agent:
# #         return Agent(
# #             config=self.agents_config['email_sender'],
# #             verbose=True
# #         )

# #     @task
# #     def extract_metrics_task(self ) -> Task:
# #         return Task(
# #             config=self.tasks_config['extract_metrics_task'],
# #             description=f"""Your a medical data analysis expert with extensive experience in interpreting and extracting information from blood reports. You have a keen eye for detail and can identify key measures and their corresponding values quickly and accurately.

# # Your task is to extract all key measures from a provided blood report text. Please return a list of each key measure along with its value presented in the text.

# # Here is the blood report text for analysis: {self.pdf_text}

# # Be sure to capture any common blood measures such as hemoglobin levels, white blood cell counts, platelet counts, cholesterol levels, and any other relevant parameters that may appear. Structure your response in a clear and organized manner, listing each measure followed by its respective value.

# # For example, if the text mentions "Hemoglobin: 13.5 g/dL," your output should explicitly state "Hemoglobin: 13.5 g/dL".
# #             """,
# #             expected_output='array of metrics',
# #             agent=self.data_extractor(),
# #         )

# #     @task
# #     def search_ranges_task(self) -> Task:
# #         return Task(
# #             config=self.tasks_config['search_ranges_task'],
# #             agent=self.range_finder()
# #         )

# #     @task
# #     def find_abnormalities_task(self) -> Task:
# #         return Task(
# #             config=self.tasks_config['find_abnormalities_task'],
# #             agent=self.abnormality_detector()
# #         )

# #     @task
# #     def send_recommendations_task(self) -> Task:
# #         return Task(
# #             config=self.tasks_config['send_recommendations_task'],
# #             agent=self.email_sender()
# #         )

# #     def __init__(self , pdf_text , email):
# #         self.pdf_text = pdf_text
# #         self.email = email
# #         print(pdf_text ,"this is from crew")
    
#     blood_test_analyst = Agent(
#         role='Blood Test Analyst',
#         goal='Analyze the blood test report and summarize the findings.',
#         backstory='A medical expert specializing in blood test analysis.',
#         verbose=True,
#         allow_delegation=False
#     )

#     article_researcher = Agent(
#         role='Article Researcher',
#         goal='Search for health articles based on blood test results.',
#         backstory='An expert researcher proficient in finding health-related articles.',
#         tools=[search_tool, web_search_tool],
#         verbose=True,
#         allow_delegation = False,
#     )

#     health_advisor = Agent(
#         role='Health Advisor',
#         goal='Provide health recommendations based on the articles found.',
#         backstory='A health advisor with extensive knowledge in providing health advice.',
#         verbose=True,
#         allow_delegation = False,
#     )

#     # Define Tasks
#     analyze_blood_test_task = Task(
#         description= f'You have to analyze the blood test report from "{raw_text}" blood report',
#         expected_output='A summary of the blood test results.',
#         agent=blood_test_analyst,
#     )

#     find_articles_task = Task(
#         description='Search for health articles based on the blood test analysis.',
#         expected_output='A list of relevant health articles with links.',
#         agent=article_researcher,
#         context=[analyze_blood_test_task]
#     )

#     provide_recommendations_task = Task(
#         description='Provide health recommendations based on the articles found.',
#         expected_output='Health recommendations with links to the articles.',
#         agent=health_advisor,
#         context=[find_articles_task]
#     )
#     @crew
#     def crew(self ) -> Crew:
#         """Creates the Assessment crew"""
#         return Crew(
#             agents=[blood_test_analyst, article_researcher, health_advisor],
#             tasks=[analyze_blood_test_task, find_articles_task, provide_recommendations_task],
#             verbose=2
#         )
    