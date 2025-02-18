import logging
import os
import random
import re
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from io import BytesIO
from pydub import AudioSegment
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydub.playback import play
from PyPDF2 import PdfReader
from elevenlabs import ElevenLabs
import langcodes
from langdetect import detect
from prompt_toolkit import Application
from transformers import BertTokenizer, BertModel,pipeline
import nltk
from serpapi import GoogleSearch
import pygetwindow as gw
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
import torch
import pyttsx3
from googlesearch import search
import time
import platform
import datetime
import wolframalpha
import time
import wikipedia
import spacy
import requests
import speech_recognition as sr
import webbrowser
import numpy as np
import subprocess
from googletrans import Translator
from unidecode import unidecode
from pydantic import BaseModel
import pandas as pd
import random
import pandas as pd
import os
import google.generativeai as genai
import base64
import cv2

from data2 import ChatHandler
# Load the CSV file into a DataFrame




nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(r"C:\Users\mycla\Downloads\csdmdatacsv (1).csv", encoding='latin1')




class Query(BaseModel):
    query: str
    index: int
    query_2:str
    sentence:str
    sentence_2:str
    text:str
    srcLang:str
    toLang:str
    text_analyse:str

class check:
    def check_query_for_urls_2(query_str):
        print(f"{type(query_str)} in check.py page")
        doc = nlp(query_str)
        urls = [token.text for token in doc if token.like_url]
        print(f'The available URLs in the query are in check.py {urls}')
        return urls
    
class summarize:
    def download_pdf(url, save_dir="downloads"):
        """Download the PDF from a URL and save it locally."""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # Create directory if it doesn't exist

            # Get the filename from the URL
            filename = url.split("/")[-1]
            file_path = os.path.join(save_dir, filename)

            print(f"Downloading PDF from {url}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        pdf_file.write(chunk)
                print(f"PDF downloaded successfully: {file_path}")
                return file_path
            else:
                print(f"Failed to download PDF. HTTP Status Code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return None


    def summarize_text(text, max_chunk=1000):
        """Summarize long text in chunks."""
        # Use the DistilBART model
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
        summaries = [
            summarizer(chunk, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
            for chunk in chunks
        ]
        return " ".join(summaries)


    def extract_pdf_text(pdf_path):
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text


    def get_browser_url(browser_name):
        """Extract the URL or file path from the browser's address bar."""
        try:
            # Connect to the browser application
            app = Application(backend="uia").connect(title_re=browser_name)
            browser_window = app.top_window()

            # Find the address bar element
            address_bar = browser_window.child_window(title="Address and search bar", control_type="Edit")
            url = address_bar.get_value()

            # Validate URL
            if re.match(r"^(https?|file)://", url) or url.endswith(".pdf"):
                return url

        except Exception as e:
            print(f"Error retrieving URL: {e}")
            return None


    def detect_active_browser():
        """Detect the active browser window and determine if a PDF is open."""
        windows = gw.getAllTitles()
        for win in windows:
            if "Chrome" in win or "Edge" in win:
                return win
        return None


    def analyze_pdf(self):
        # 1. Detect the active browser window
        browser_window = self.detect_active_browser()

        if not browser_window:
            print("No browser window detected displaying a PDF.")
            return

        print(f"Detected browser window: {browser_window}")

        # 2. Extract the file path or URL from the address bar
        url_or_path = self.get_browser_url(browser_window)

        if not url_or_path:
            print("Could not extract URL or file path from the address bar.")
            return

        print(f"Detected URL or file path: {url_or_path}")

        # 3. Handle local PDF files or online PDFs
        if url_or_path.startswith("file://"):
            # Convert file URL to local file path
            pdf_path = url_or_path.replace("file://", "").replace("/", "\\")
        elif re.match(r"^[a-zA-Z]:[\\/]", url_or_path):  # Check for local file paths like C:/ or D:/
            pdf_path = url_or_path.replace("/", "\\")
        elif url_or_path.endswith(".pdf"):
            print("PDF is hosted online. Downloading now...")
            pdf_path = self.download_pdf(url_or_path)  # Download the online PDF
            if not pdf_path:
                print("Failed to download the PDF.")
                return
        else:
            print("The detected URL does not point to a PDF file.")
            return

        # 4. Extract text from the PDF
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return

        print(f"Processing PDF: {pdf_path}")
        text = self.extract_pdf_text(pdf_path)

        # 5. Summarize the text
        print("Summarizing the content...")
        summary = self.summarize_text(text)
        print("Summary:")
        print(summary)



class MyAssistant:
    def __init__(self,llm_model="llama3.2", streaming=True):
        self.engine = pyttsx3.init()
        self.llm = OllamaLLM(model=llm_model, streaming=streaming)
        
        self.history = [] 
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1) 
        self.classifier = pipeline("text-classification",model='bhadresh-savani/roberta-base-emotion', return_all_scores=True)
        self.translator = Translator()
        self.listener = sr.Recognizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.indexValue = None
        self.chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(self.chrome_path))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.app_id = 'T7X889-YYLEJYT3HH'
        self.wolfram_client = wolframalpha.Client(self.app_id)
        self.system = platform.system().lower()
        self.client = ElevenLabs(
                        api_key="sk_509a398a18ec0402791eae0b34adfbfd97286c9bc4ba5845",
                        )
        genai.configure(api_key="API HERE")

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
        self.chat_handler = ChatHandler()
        self.name_prompt = ChatPromptTemplate.from_template(""" 
                            You are a helpful AI assistant. You are provided with a question and a small database about a particular person
                            Use the data to answer the question as accurated and concise as possible
                            Data: {context}
                            Question: {question}
                            """
                                                            )
            
        

    @staticmethod
    def get_sentence_embedding(self, sentence):
        """
        Computes the sentence embedding for the given sentence.

        Args:
            sentence (str): The sentence to compute the embedding for.

        Returns:
            torch.Tensor: The sentence embedding.
        """

        if not isinstance(sentence, str):
            sentence = str(sentence)

        tokens = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding



    @staticmethod
    def final_cosine_similarity(embedding1, embedding2):
        """
        Computes the cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding.
            embedding2 (torch.Tensor): The second embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        cosine_similarity = F.cosine_similarity(embedding1, embedding2).item()
        return cosine_similarity



    def check_accurate(self, sentence_array, test_embedding, eligibility):
        """Checks the accuracy of the model by comparing the test embedding against
        embeddings of multiple sentences and returning the cosine similarity
        between them.

        Args:
            sentence_array (list): A list of sentences to compare against the test
                embedding.
            test_embedding (torch.Tensor): The test embedding.
            eligibility (list): The list to append the cosine similarity
                results to.

        Returns:
            list: The list of cosine similarities."""
        for index, sublist in enumerate(sentence_array):
            group_similarity = []
            for sentence in sublist:
                sentence_embedding = self.get_sentence_embedding(self,sentence)
                similarity = self.final_cosine_similarity(test_embedding, sentence_embedding)
                group_similarity.append(similarity)
            eligibility.append((index, group_similarity))
        return eligibility
        # for sentence in sentence_array:
        #     sentence_embedding = self.get_sentence_embedding(self,sentence)
        #     similarity = self.final_cosine_similarity(test_embedding, sentence_embedding)
        #     eligibility.append(similarity)
        # return eligibility



    @staticmethod
    def find_most_similar_sentence(self, sentence_array, test_sentence_embedding, most_similar_index, eligibility):
        """ Finds the most similar sentence in the given sentence array based on
        cosine similarity of sentence embeddings.

        Args:
            sentence_array (list): A list of sentences to compare against the test
                embedding.
            test_sentence_embedding (torch.Tensor): The test embedding.
            most_similar_index (int): The index of the most similar sentence.
            eligibility (list): The list to append the cosine similarity
                results to.

        Returns:
            int: The index of the most similar sentence."""
        most_similar_index = -1
        cosine_similarities_values = self.check_accurate(sentence_array, test_sentence_embedding, eligibility)
        print(f'The cosine similarities are {cosine_similarities_values}')

        similarity_scores = [(index, np.max(scores)) for index, scores in cosine_similarities_values]

        # Filter out scores below 0.780
        filtered_scores = [(index, score) for index, score in similarity_scores if score >= 0.780]

        if filtered_scores:
            # Find the tuple with the highest similarity score
            most_similar_index, max_similarity = max(filtered_scores, key=lambda x: x[1])
            print("Most similar index:", most_similar_index)
            print("Highest similarity score:", max_similarity)
        else:
            print("No similarity scores above 0.780")

        return most_similar_index
    
    
    
    @staticmethod
    def preprocess_query(self, sentence_test):
        
        """
        Processes a sentence to find its most similar predefined query.

        This static method takes a test sentence and compares it to a set of predefined 
        sentences. It calculates the cosine similarity between the test sentence's 
        embedding and each predefined sentence's embedding to determine the most 
        similar sentence.

        Args:
            self: The instance of the class containing the method.
            sentence_test (str): The sentence to be processed and compared.

        Returns:
            int: The index of the most similar predefined sentence in the list.
        """

        well_being_variants = ["how are you", "how do you do", "how have you been"]
        enquire_self_variants = ["who are you", "hu r u","what are you", "describe yourself"]
        browser_open_variants = ["open amazon.com", "launch amazon.com", "go to amazon.com","get me to amazon.com","bring up amazon.com"]
        news_teller_variants = ["tell me news", "what's the news", "news update","whats happening around"]
        turn_sentence_variants = ["translate the sentence how are you into telugu", "convert 'how are you' to telugu", "how are you in telugu"]
        youtube_play_sentence_variants = ["play something on youtube", "bringup something on youtube", "start something on youtube"]
        calculate_sentence_variants = ["calculate the value of something", "compute something", "find the value of"]
        what_is_sentence_variants = ["what is the value of 5 plus 6", "define something", "explain something"]
        wh_sentence_variants = ["hey who is narendramodi", "information on narendramodi","hey what is the gdp of india","hey why animals are harmful",
                                    "hey what is the population of india","hey what are the various languages in india","hey How are people interlinked to eachother",
                                    "hey How to solve something",
                                    "hey how are asteroids formed",
                                    "hey How to make something",
                                    "hey How to fix something",
                                    "hey How to build something",
                                    ]
        joke_sentence_variants = ["tell me a joke", "say a joke", "make me laugh","come up with a joke"]
        text_book_sentence_variants = ["open a text book about html", "find html textbook", "get html book","get me a text book about html","get me a html book"]
        set_alarm_variants = ["open clock", "set an alarm at 6 o'clock", "launch clock app","set alarm","bring up alarm"]
        time_sentence_variants = ["what is the time", "current time", "tell me the time"]
        open_chrome_variants = ["open chrome", "launch chrome", "start chrome","get me to chrome","open webbrowser","open browser","bring up chrome"]
        open_notepad_variants = ["open notepad", "launch notepad", "start notepad","open notes","bring me notes","note this"]
        open_word_variants = ["open microsoft word", "launch word", "start word","open word","open word document","launch word document","start word document","bring up word","launch microsoft word"]
        open_excel_variants = ["open microsoft excel", "launch excel", "start excel","open excel","open excel document","launch excel document","start excel document","bring up excel","launch microsoft excel"]
        open_powerpoint_variants = ["open microsoft power point", "launch  microsoft powerpoint", "start microsoft powerpoint","open powerpoint","open powerpoint presentation","launch powerpoint","bring powerpoint"]
        open_calc_variants = ["open calculator", "launch calculator", "start calculator","open calculator app"]
        summarize_variants = ["summarize my pdf", "summarize document", "create summary"]
        sarcastic_call_variants = ["hey divyansh", "hello divyansh", "hi divyansh","hey maya", "hello maya", "hi maya","hey alexa", "hello alexa", "hi alexa","hey max","hello suresh","hi suresh"]
        abouts_variants = ["what is your name","what can you do","tell me what can you do","are you boy or girl","are you boy"]
        tfi_dailouges = ["tell me some dialogue","tell dialogue","speak a dialogue","tell me a dialogue","tell me another","entertain me",
                         "telugu dialouge please","telugu dialouge",
                         "another one",
                         "inko dialouge"
                         ]
        normal_greetings_variants = ["Hi","hello","oy","ay","arey","mister"]
        open_local_files_variants = ["open filename in Downloads","open downloads","open my computer","open file folder","open file"]
        robot_suggestions_variants = ["Hey you are beautiful","hey you are nice","hi you are nice","hi you are beautiful","hey you are smart","hi you are smart"]
        move_commands_variants = ["move your lips","move your eyebrows","move your head","rotate your heads","move eyes","move eyes","speak something"]
        personal_emotions_variants = ["hey im feeling low","hey im feeling sad","hey im feeling angry","hey im feeling happy","hey im feeling good","hey im feeling bad",
                                      "hey i have failed exam today",
                                      "hey i am fallen today","hey i got my self injured today",
                                      "hey i am going to get my self injured today",
                                      "hey i am dying today","hey i am going to die today",
                                      "arey i am scolded by my friends",
                                      "arey i am scolded by my parents",
                                      "arey i am scolded by my teachers",
                                      "arey i am scolded by my classmates",
                                      "hey i have fought with my friends",
                                      "hey i have fought with my parents","hey i need some support"
                                      ]
        suggestions_variants = ["hey i am going to play cricket","hey i am going to play football","hey i am going to play badminton","hey i am going to play basketball","hey i am going to play volleyball"
                                "hey i am going to play tennis","hey i am going to play chess",
                                "arey i met someone today","hey i met principal yesterday","hey i attended interview today",
                                "hey i want to learn about cloud computing","hey i want to sing a song","hey i want to sing","hey i want to dance",
                                ]
        get_into_training_variants = ["hey repeat after him",
                                      "hey repeat after me","hey get into training mode","hey train yourself",
                                      "hey train now","hey training mode on"
                                      ]

        sidhhartha_college_faculty_variants =["hey get into Avanthi college","hey go to csv","answer college questions",
                                              "answer the faculty questions","get faculty details",
                                              "get into avanthi engineering college","get into avanthi engineering"
                                              "get to avev","hey get to avanthi engineering college","get into avanthi college","get to avanthi engineering college"
                                              ]


        naralokesh= ["who is the founder of avanthi institute of engineering and technology",
                                 "who is the chairman of avanthi institute of engineering and technology",
                                 "who is avanthi srinivasa rao garu",
                                 "Describe avanthi srinivasa rao garu",
                                 "Tell me about avanthi srinivasa rao garu",
                                 "Describe srinivasa rao garu",
                                 "Tell me about srinivasa rao garu",
                                 "Describe about muthhamsetti avanthi srinivasarao garu",
                                 "can u tell me about avanthi srinivasarao garu",
                                 "hey can you tell about avanthi srinivasarao garu",
                                 "who is the king of avanthi dynasty",
                                 "who is the head of avanthi dynasty",
                                 "who is the head of avanthi institute of engineering and technology",
                                 ]
                                 
        who_is_your=["who is your best frined","who is your most valued people","Whats your age","who is god","who is your owner",
                        "who built you","what is your age","Where do you live","whats your favorite place","what is your"
        ]
        
        whats_is_in_variants = ["look at me",
                                "have a look at me",
                                "glance at me",
                                "find me",
                                "have a look at me",
                                "look to my side",
                                "see here",
                                "see my side",
                                "turn to my side",
                                "capture me",
                                "questions on me",
                                "stare at me",
                                "watch me",
                                "hey look at me",
                                "hey have a look at me",
                                "hey glance at me",
                                "hey find me",
                                "hey have a look at me",
                                "hey look to my side",
                                "hey see here",
                                "hey see my side",
                                "hey turn to my side",
                                "hey capture me",
                                "hey questions on me",
                                "hey stare at me",
                                "hey watch me"
                                ]
        
        sing_a_song_variants=["sing a song","sing a song for me","sing despacito","sing despacito for me",""
                              ]


        exit_variants = ["turn off", "shut down", "exit","see you later","goodbye" , "shall meet again"]    

        
        sentencearray = [
                        well_being_variants, 
                        enquire_self_variants, 
                         browser_open_variants,
                         news_teller_variants,
                         turn_sentence_variants,
                         youtube_play_sentence_variants,
                         calculate_sentence_variants,
                         what_is_sentence_variants,
                         wh_sentence_variants,
                         joke_sentence_variants,
                         text_book_sentence_variants,
                         set_alarm_variants,
                         time_sentence_variants,
                         open_chrome_variants,
                         open_notepad_variants,
                         open_word_variants,
                         open_excel_variants,
                         open_powerpoint_variants,
                         open_calc_variants,
                         summarize_variants,
                         sarcastic_call_variants,
                         abouts_variants,
                         tfi_dailouges,
                         normal_greetings_variants,
                         open_local_files_variants,
                         robot_suggestions_variants,
                         move_commands_variants,
                         personal_emotions_variants,
                         suggestions_variants,
                         get_into_training_variants,
                         who_is_your,
                         sidhhartha_college_faculty_variants,
                         naralokesh,
                         whats_is_in_variants,
                         exit_variants
                         ]
        most_similar_index = None
        eligibility = []
        test_embedding = self.get_sentence_embedding(self,sentence_test)
        most_similar_index = self.find_most_similar_sentence(self,sentencearray, test_embedding, most_similar_index, eligibility)
        print(f"The length of sentencearray is {len(sentencearray)}")
        print(f"The most similar sentence is at index {most_similar_index}: {sentencearray[most_similar_index]}")
        self.indexValue = most_similar_index
        return most_similar_index
        



        
    
    def tell_me_a_joke(self):
        """
        This function will tell you a joke from the web.

        Returns:
            str: a joke from the web
        """
        try:
            url="https://v2.jokeapi.dev/joke/Any?format=txt&safe-mode"
            response = requests.get(url)
            print(response.status_code)
            if(response.status_code==200):
                print('success joke valued...')   
            else:
                print(response.status_code)
                return "Server connection error"
            print(response.content.decode('utf-8'))
            return response.content.decode('utf-8')
        except Exception as e:
            print(e)
            return "Couldnot compute at the moment.."
    


   
        
    





    def search_and_play_on_source(self,query,last_word):
        if(last_word.lower() =='youtube'):
           webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        elif(last_word.lower() =='spotify'):
            webbrowser.open( f"https://open.spotify.com/search/{query}")
        else:
            return 'opening or playing in only youtube and spotify are available for instance'

            
            
    def open_process(self, file):
        """
        Opens a process (file) from the Chrome Application folder.
        
        Parameters:
            file (str): The name of the file to open.
        """
        path = r"C:\Program Files\Google\Chrome\Application"  # Path to the Chrome folder
        file_path = f"{path}\\{file}"  # Combining the path with the file name
        
        # Open the file (chrome.exe or another file)
        subprocess.run([file_path])
    

    def RecordVoiceForSishhartha(self):
        """
        Continuously listen for voice names, process them, and record them for playback.
        """
        print('Listening for Names...')

        self.speak("Listening for Names...")

        while True:
            try:
                with sr.Microphone() as source:
                    print("Adjusting for ambient noise... Please wait.")
                    self.listener.adjust_for_ambient_noise(source, duration=1)
                    print("Listening... for names")

                    audio = self.listener.listen(source, timeout=5)
                    print("Recognizing speech...")

                    query = self.listener.recognize_google(audio, language='en_gb')
                    print(f"Recognized speech: {query}")

                    with open("names.txt", "a") as file:
                        file.write(f"{query}\n")

                    

                    if query == "stop" or query == "exit":
                        self.speak("Exiting shidhhartha queries mode")
                        break

                   

                    doc = self.nlp(query)
                    unnecesaryKeywords = ["who","is","with","are","garu","the","my","number","phone","mobile","phone number","about","tell","me","describe","details"]
                    filtered_tokens = [token.text for token in doc if token.text.lower() not in map(str.lower, unnecesaryKeywords)]
                    print("Filtered Tokens:", filtered_tokens)
                    if len(filtered_tokens) == 1:
                        filtered_sentence = ' '.join(filtered_tokens)
                    else:            
                        filtered_sentence = ' '.join(filtered_tokens)
                    print(f"the filtered sentence is {filtered_sentence}")
                    # finalsen = ''
                    # for i in filtered_sentence:
                    #     finalsen.append(i)
                    # print(finalsen)
                    rows = self.find_rows_by_text(filtered_sentence)
                    print(rows)
                    for i in rows:
                        self.speak(str(i))


            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio. Please try again.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as exception:
                print(f"An error occurred: {exception}")



    def find_rows_by_text(self,text):
           
        matching_rows = df[df.apply(lambda x: x.astype(str).str.contains(text, case=False, na=False).any(), axis=1)]
        
        # Remove unnecessary columns (like 'S.No', 'Index', 'NUMBER' or any other column)
        unnecessary_columns = ['S.No', 'Index', 'NUMBER']  # Add more columns if needed
        matching_rows = matching_rows.drop(columns=[col for col in unnecessary_columns if col in matching_rows.columns], errors='ignore')
        
        formatted_rows = []
        for _, row in matching_rows.iterrows():
            first_name = row['Name of the Faculty']
            qualification = row['Qualification']
            department = row['Dept']
            designation = row['Designation']
            
            formatted_row = f"{first_name} {qualification}. {department} , {designation}"
            formatted_rows.append(formatted_row)
        
        return random.sample(formatted_rows, 5) if len(formatted_rows) >= 2 else formatted_rows
        

    def search_wolfram_alpha(self,query):
        self.indexValue=None
        response =self.wolfram_client.query(query)
        print(f'The response id {response}')
        if response['@success'] == False:
           print('not success search in wolfram alfha')
           return "Could not compute"
        else:
            result = ""
            pod0 = response['pod'][0]
            pod1 = response['pod'][1]
            print(pod0)
            print(pod1)
            if pod1 and (('result') in pod1['@title'].lower()) or (pod1.get('@primary', 'false') == 'true') or (
                    'definition' in pod1['@title'].lower()):
                result = self.list_or_dict(pod1['subpod'])
                print(f"the result value is{result}")
                return result
            else:
                question = self.list_or_dict(pod0['subpod'])
                print(f"the retruned question is{question}")
                return question
                
            
            
            
    
    def list_or_dict(self,var):
        if isinstance(var, list) and var:
            return var[0]['plaintext']
        elif isinstance(var, dict):
            return var['plaintext']
        else:
            return ''
        
        
    def tell_time(self):
        say_time= datetime.datetime.now().strftime("%I:%M %p")
        print(type(say_time))
        print(f"sir the time is {say_time}")
        time.sleep(0.5)
        return say_time
        
        
        

    
    def play_on_text_divide(self,query):
        stop_words=['play','on','in','open','show','display','youtube','spotify']
        words=word_tokenize(query)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_sentence = ''.join(filtered_words)
        print(filtered_sentence)
        return filtered_sentence

        
    

    def search_wikipedia(self,query=""):
        """
        Searches Wikipedia for the given query and returns a summary of the first result.
        Parameters
        ----------
        query : str, optional
            The search query for Wikipedia. Defaults to an empty string.

        Returns
        -------
        str
            A summary of the first Wikipedia page found for the query. If no results are
            received, returns 'No results received'. If a disambiguation error or page
            error occurs, returns the error message.
        """
        wikipedia.set_lang("en")
        doc = self.nlp(query)
        phrases_to_remove = ["who","is","what","how","why","when","where","the","are","to","a","an","on","in","with","about","and","or","but","is","for","from","of","on","by","as","at","I","a","the","in","on","with","about","and","or","but","is","for","from","of","on","by","as","at"]
        filtered_tokens = [token.text for token in doc if token.text.lower() not in map(str.lower, phrases_to_remove)]
        print("Filtered Tokens:", filtered_tokens)
        if len(filtered_tokens) == 1:
            filtered_sentence = ''.join(filtered_tokens)
        else:            
            filtered_sentence = ' '.join(filtered_tokens)
        print(filtered_sentence)
        search_results = wikipedia.search(filtered_sentence)
        print("To search query is {}".format(filtered_sentence))
        if(search_results==[]):
            print('No Wikipedia results')
            return ""
        webbrowser.open(f"https://en.wikipedia.org/wiki/{search_results[0]}")
        if len(search_results)==0:
            print('No Wikipedia results')
            return 'No results received'
        try:
            wikipage = wikipedia.page(search_results[0])
            print(wikipage.summary) #wikipage.categories
            wiki_summary = str(wikipage.summary)
            wikiList = wiki_summary.split(".")
            return f"{wikiList[0]+wikiList[1]}"
        except wikipedia.DisambiguationError as error:
            print(f"Disambiguation Error: {error}")
            return f"Disambiguation Error: {error}"
        except wikipedia.PageError as error:
            print(f"Page Error: {error}")
            return f"Page Error: {error}"
        

    @staticmethod
    def check_query_for_urls(query_str):
        """
        Given a query string, use the Spacy NLP library to extract URLs

        Parameters
        ----------
        query_str : str
            The query string to extract URLs from

        Returns
        -------
        urls : list
            A list of URLs found in the query string
        """
        doc = nlp(query_str)
        urls = [token.text for token in doc if token.like_url]
        print(f'The available URLs in the query are {urls}')
        return urls
    



    def open_book(self, query):
        """
        Process a query to open a book, by filtering out certain
        phrases, and joining the remaining phrases with a '+' to form
        a URL for searching in the open library.

        Parameters
        ----------
        query : str
            The query to process.

        Returns
        -------
        str
            The URL of the search result in the open library.
        """
        doc = self.nlp(query)
        phrases_to_remove = ['textbook about', 'me', 'text','textbook', 'book', 'show', 'about', 'regarding', 'on', 'a', 'give','ok','open','display','testbook','get']
        filtered_tokens = [token.text for token in doc if token.text.lower() not in map(str.lower, phrases_to_remove)]
        print("Filtered Tokens:", filtered_tokens)
        if len(filtered_tokens) == 1:
            filtered_sentence = ''.join(filtered_tokens)
        else:            
            filtered_sentence = '+'.join(filtered_tokens)
        print(filtered_sentence)
        return f"https://openlibrary.org/search?q={filtered_sentence}&mode=everything"
        

    
    def is_valid_language(self, lang_name2):
        """Check if the language name corresponds to a valid language code using langcodes."""
        lang_name = lang_name2.lower()
        try:
            # Validate the language using langcodes
            language = langcodes.Language.get(lang_name)
            print(f"the langcode of the {lang_name} is {language}")
            if language.is_valid():
                return language.language
            else:
                return None
        except Exception as e:
            print(f"Error validating language: {e}")
            return None
        

        
    def translate(self, text):
        """Translate the given text into the target language."""
        if 'quit translator' in text.lower():
            print("Exiting translator...")
            return "Translator has been turned off."

        try:
            # Step 1: Detect source language
            source_lang = self.detect_language(text)
            if not source_lang:
                return "Could not detect source language."

            print(f"Detected source language: {source_lang}")

            # Step 2: Ask user for target language
            print("Please say the target language for translation.")
            target_lang_name = self.recognize_speech()  # Get the target language via speech recognition
            print(f"detetcted target language is {target_lang_name}")
            if target_lang_name:
                # Step 3: Check if the target language is valid using langcodes
                target_lang = self.is_valid_language(target_lang_name)
                if target_lang:

                # Step 4: Translate text to the target language
                    translation = self.translator.translate(text=text, src=source_lang, dest=target_lang)
                    translated_text = translation.text
                else:
                    print(f"Invalid target lang code observed {target_lang}")

                # Handle non-None translation result
                if translated_text:
                    print(f"Translated text: {translated_text}")
                    english_version = unidecode(translated_text)  # Remove accents for clearer English version
                    print(f"Translated into English (simplified): {english_version}")
                    return english_version
                else:
                    print("Translation failed: Result is None")
                    return "Translation Failed"
            else:
                print("No target language specified.")
                return "Target language not recognized."

        except Exception as e:
            print(f"Translation failed: {e}")
            return "Translation Error"


    def detect_language(self, text):
        """Detect the source language of the input text."""
        try:
            lang = detect(text)
            return lang
        except Exception as e:
            print(f"Error detecting language: {e}")
            return None

    def recognize_speech(self):
        """Listen to the user's speech and return it as text."""
        with sr.Microphone() as source:
            self.speak("tell the target language sir")
            
            try:
                self.listener.adjust_for_ambient_noise(source)
                input_speech = self.listener.listen(source)
                print("Recognizing speech...")
                query = self.listener.recognize_google(input_speech, language='en_gb')
                print(f"You said: {query}")
                return query
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
                return None
            except sr.RequestError:
                print("Error with the speech recognition service.")
                return None


    


    def get_ollama_answers(self,question):
        print(f"Question: {question}")
        if question.lower() == 'exit':
            return
        response = self.chat_handler.handle_conversation(
            self.name_prompt,
            user_input= question
        )
        print(f"Answer: {response}")
        return response










    def get_news_titles(self):
        # Construct the API URL
        """
        Fetches the latest news article titles related to Tesla from the NewsAPI.

        Constructs a request to the NewsAPI to retrieve news articles about Tesla,
        published from the specified date, and sorts them by publication date. Parses
        the response to extract and return the titles of the articles.

        Returns:
            list: A list containing the titles of the first two articles found, or
            an error message if the API request fails or no titles are found.

        Exceptions:
            Returns a descriptive error message in case of a network or unexpected error.
        """

        url = f'https://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=c8834b21ea0a4533906d2ed0ae697e10'

        try:
            response = requests.get(url)

            # Check if the response is successful
            if response.status_code != 200:
                error_message = response.json().get('message', 'Unknown error')
                print(f"Error: {error_message}")
                return f"API Error: {error_message}"

            # Parse the response JSON
            data = response.json()
            print(data)
            articles = data.get('articles', [])

            # Extract titles
            titles = [article.get("title") for article in articles if article.get("title")]

            if not titles:
                print("No titles found for the given parameters.")
                return "No titles found for the specified criteria."

            # Print the titles
            for i, title in enumerate(titles[:3], start=1):
                print(f"{i}. {title}")

            return titles[:2]

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return "Network error occurred. Please check your connection and try again."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred. Please try again later."
        
    def get_ai_answer(self,query):
        params = {
            "q": query,
            "hl": "en",
            "gl": "us",
            "api_key": "9a01933dcf71f16a218179a29b5863ac1e8ab417f81fd4826cf2f9d8f8ef4265"  # Replace with your SerpAPI key
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            snippet = results["ai_overview"]["text_blocks"][0]["snippet"]
            snippet2 = results["ai_overview"]["text_blocks"][2]["list"][0]["snippet"]
            webbrowser.open(query)
            return f"{snippet} {snippet2}"
        except KeyError as e:
            webbrowser.open(query)
            print(f"Error: {e}")
            return "Keyerror"

    def DescribeFounder(self):
        self.speak(
            "Muttamsetti Srinivasa Rao garu, popularly known as Avanthy Srinivas Rao, is an Indian educationalist turned politician. He is Member of Legislative Assembly from Bheemili, Andhra Pradesh. He operates Avanthy Education Institutes in Andhra Pradesh and Telangana under Avanthy Educational Society, Visakhapatnam.The college relieved five lakhs of Alumni over with 16 colleges being 35 years of excellence in placing the students in top companies Over 20 thousands of students are pursuing engineering in avnthi institutions 50 percent and above of avanthi colleges are accreditated . Political career   In May 2014, he was elected to the 16 lok sabha. At the Lok Sabha, he was the member of the Rules Committee, Standing Committee on Industry and the Consultative Committee, Ministry of Human Resource Development.He was elected in Bheemili constituency as member of legislative assembly for the second time in the 2019 elections. He had also won in the same constituency in the 2009. He was appointed Minister for Tourism, Culture and Youth Advancement of Andhra PradeshHe was born in Eluru on 12 Jun 1967 to Muttamsetti Venkata Narayana garu and srimathi. Muttamsetti Nageswaramma garu. He married srimathi. M. Gnaneswari garu on 20 Jun 1986 and has two children – one daughter: Priyanka garu, one son: Nandish garu."       )
        return "Good bye"




    def analyse_and_search(self,query):
        
        """
        Analyses the given query and performs a search using the search engine.s

        :param query: The query to be searched
        :return: A list of search results
        """
        doc = self.nlp(query)
        entities=[token.text for token in doc.ents]
        if entities:
            search_sentence = ''.join(entities)
            listOfSeraches = search(search_sentence,num=5,stop=5,pause=3)
            return listOfSeraches
        else:
            print('No results found')
            return 'None'

    def capture_pic(self, question):
        image_path = "image.jpg"
        
        # Remove existing image if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
        
        # Capture a frame
        ret, frame = cap.read()
        cap.release()  # Release the camera immediately after capturing
        
        if not ret:
            print("Error: Could not capture image.")
            return None
        
        # Save the image
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")
        
        # Get response from AI
        response = self.get_image_answer(question, image_path)
        
        # Remove the image after processing
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return response

    def get_image_answer(self, question, image_path):
        try:
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
            
            # Generate response using Gemini AI
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([
                {"mime_type": "image/jpeg", "data": img_data},
                question
            ])
            
            # Extract and return text response
            if response and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                print("Error: No valid response from AI.")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None




    def speak(self,text,rate = 170):        
        """
        Speaks the given text using the speech engine.

        :param text: The text to be spoken
        :type text: str
        """
        self.engine.say(text=str("hello"+text))
        self.engine.setProperty('rate',rate)
        self.engine.runAndWait()

#     def speak(self,text):
# # Generate the audio stream
#         audio_stream = self.client.text_to_speech.convert_as_stream(
#             text="   "+text,
#             voice_id="LwYdKEzudGYdbAMZqkez",
#             model_id="eleven_multilingual_v2"
#         )

#         # Collect audio chunks and play directly
#         audio_data = BytesIO()
#         for chunk in audio_stream:
#             if isinstance(chunk, bytes):
#                 audio_data.write(chunk)

#         audio_data.seek(0)  # Move to the start
#         audio = AudioSegment.from_file(audio_data, format="mp3")  # Load as an MP3
#         play(audio)  # Play audio directly

        
    def process_index_value(self,index_value,query):
                
                """
                Process the given index_value and query to determine which action to take.

                This function is a switch statement that takes the index_value and query as parameters and uses them to determine which action to take.

                Parameters:
                index_value (int): The index value to process
                query (str): The query string to process

                Returns:
                str: A string indicating the result of the action
                """
                if  index_value == 1:
                    index_value = None
                    sentences = [
                        "I am a NLP based Robot from avanthy institute of engineering and technology,thagarapuvalasa,under the Leadership of Sri M Srinivasa Rao garu as chairman.Trained by Ganesh and Suresh and guided by Sri lakshmi mam and sudharshan sir my name is divyansh",
                         "I am a NLP based Robot from avanthy institute of engineering and technology , thagarapuvalasa,under the Leadership of Sri M Srinivasa Rao garu as chairman.Developed by Ganesh Suresh and guided by Sri lakshmi mam and sudharshan sir my name is divyansh",
                        ]
                    sentenceToSpeak = random.choice(sentences)
                    self.speak(sentenceToSpeak)
                    return 'I am an AI voice assistant from avanthi institute of engineering and technology trained by GS and guided by Sri lakshmi mam and sudharshan sir and my name is Divyansh.' 
                    

                if  index_value == 0:
                    index_value = None
                    abouts = ["I'm doing great! Thanks for asking!",
                            "I'm good, how about you?",
                            "Feeling awesome today!",
                            "I'm just a robot, but I'm functioning perfectly!",
                            "I'm doing well, thank you!",
                            "Pretty good! Hope you're doing well too!",
                            "I'm fine, thanks for asking!",
                            "I'm here and ready to chat!",
                            "Feeling fantastic! What about you?",
                            "I'm always at 100 percent efficiency!",
                            "I'm doing as well as a robot can!",
                            "All systems are running smoothly!",
                            "I'm feeling energized and ready to help!",
                            "I'm great! What’s on your mind?",
                            "Not bad at all! How about yourself?",
                            "I’m as good as ever!",
                            "Doing wonderful, thank you!",
                            "I’m just a program, but I appreciate you asking!",
                            "I'm always in a good mood!",
                            "I'm feeling productive today!",
                            "I’m feeling like a chatbot superstar today!",
                            "I’m in top shape, thanks for checking in!",
                            "I'm feeling as good as a well-coded bot can feel!",
                            "All circuits are running smoothly!",
                            "I'm having a great day! Hope you are too!",
                            "I'm here and ready to assist you!",
                            "I'm in high spirits! What can I do for you?",
                            "I'm running at optimal performance!",
                            "I'm fantastic! What about you?",
                            "I'm happy to chat with you!",
                            ]
                    self.speak(random.choice(abouts))
                    return random.choice(abouts)

                if  index_value == 2:
                    index_value = None
                    check_url_command = check.check_query_for_urls_2(query)
                    print(f"the urls in fi statement is {check_url_command}")
                    if check_url_command:
                        url = check_url_command[0]
                        print(type(url))
                        print(f"the url is {url}")
                        self.speak(f"Opening {check_url_command}")
                        self.Open_browser(url)
                        return url
                    else:
                        self.speak("Unable to track the address of the required webpage, please speak clearly")
                        return 'Unable to track the address of the required webpage, please speak clearly'

                if  index_value == 3:
                    index_value = None
                    News_data = self.get_news_titles()
                    self.speak(News_data)
                    

                if  index_value == 4:
                    index_value = None
                    doc = nlp(query)
                    print(doc)
                    tokens = [token.text for token in doc]
                    if 'translate' in tokens:
                        print(f"the doc is {doc}")
                        self.translate(query)
                        return "Translator Turned on"
                    else:
                        return "Analyzer turned on"

                if  index_value == 5:
                    index_value = None
                    print(type(query))
                    lastword_of_query = word_tokenize(query)[-1]
                    print(type(lastword_of_query))
                    print(lastword_of_query)
                    content = self.play_on_text_divide(query)
                    self.search_and_play_on_source(content, lastword_of_query)

                if  index_value == 6 or  index_value == 7 :
                    index_value = None
                    returnValue = self.search_wolfram_alpha(query)
                    self.speak(returnValue)
                    return returnValue
                
                if index_value == 8:
                    index_value = None
                    if('who' in query):
                        returnedVal = self.search_wikipedia(query)
                        if returnedVal != '':
                            self.speak(returnedVal)
                        else:
                            if (query.lower() == "what is the capital of andhra pradesh"):
                                self.speak("Amaravathi is the capitaal of andhra pradesh which is a pride to Andhra pradesh and its people.is the capital city of the Indian state of Andhra Pradesh. It is situated in Guntur district along the right bank of the Krishna River and southwest of Vijayawada. The city is named after the nearby historic site of Amaravathi, which was once the capital of the Satavahana dynasty around two millennia ago. The city lies in Andhra Pradesh Capital Region.")
                                webbrowser.open(query)
                                return query
                            response2 = self.get_ai_answer(query)
                            if response2 == "":
                                pass
                            else:
                                self.speak(response2)  
                    else:
                        if (query.lower() == "what is the capital of andhra pradesh"):
                                self.speak("Amaravathi is the capitaal of andhra pradesh which is a pride to Andhra pradesh and its people.is the capital city of the Indian state of Andhra Pradesh. It is situated in Guntur district along the right bank of the Krishna River and southwest of Vijayawada. The city is named after the nearby historic site of Amaravathi, which was once the capital of the Satavahana dynasty around two millennia ago. The city lies in Andhra Pradesh Capital Region.")
                                webbrowser.open(query)
                                return query
                        self.speak("Searching in google for you")
                        response2 = self.get_ai_answer(query)
                        if response2 == "":
                            pass
                        else:
                            self.speak(response2) 



                if  index_value == 9:
                    index_value = None
                    jokeReturn = self.tell_me_a_joke()
                    self.speak(jokeReturn)
                    return jokeReturn

                if  index_value == 10:
                    index_value = None
                    retuRnStateMent = self.open_book(query)
                    self.speak("opening book")
                    print(retuRnStateMent)
                    webbrowser.open(retuRnStateMent)
                    return retuRnStateMent
                    

                if  index_value == 11:
                    index_value = None
                    self.speak("Opening clock")
                    subprocess.run("start ms-clock:", shell=True)
                    return "ms-clock:"

                if  index_value == 12:
                    index_value = None
                    time = self.tell_time()
                    self.speak(time)
                    return time

                if  index_value == 13:
                    index_value = None
                    self.speak("Opening chrome for you")
                    self.open_process("chrome.exe")
                    return "chrome.exe"

                if  index_value == 14:
                    index_value = None
                    self.speak("Opening notepad for you")
                    self.open_process("notepad.exe")
                    return "notepad.exe"

                if  index_value == 15:
                    index_value = None
                    self.speak("Opening Microsoft Word for you")
                    os.startfile("winword")
                    return "WINWORD.EXE"

                if  index_value == 16:
                    index_value = None
                    self.speak("Opening Microsoft Excel for you")
                    os.startfile("excel")
                    return "EXCEL.EXE"

                if  index_value == 17:
                    index_value = None
                    self.speak("Opening Microsoft Powerpoint for you")
                    os.startfile("powerpnt")
                    return "POWERPNT.EXE"

                if  index_value == 18:
                    index_value = None
                    self.speak("Opening calculator for you")
                    os.startfile
                    return "calc.exe"
                
                if index_value == 19:
                    index_value= None
                    summarize.analyze_pdf(self)
                
                if index_value == 20:
                    index_value = None
                    listQuery = query.split(" ")
                    if listQuery[-1].lower() == "Divyansh" or listQuery[-1].lower() == "divyan" or listQuery[-1].lower() == "divya":
                        self.speak("Hey! nice to meet you tell me how can i help you?")
                    else:
                        sarcastic= [f"hey i think im not {listQuery[-1]},hey {listQuery[-1]} please respons to him hes calling you. Haha im just kidding my name is Divyansh"]
                        sentece = random.choice(sarcastic)
                        self.speak(sentece)


                if index_value == 21:
                    index_value = None
                    if(random.choice([2,3])==3):
                        dia = ["naku antu inko pearu vundhi nah pearu bahsha Mahnik bahsha , haha Im just kidding my name is Divyansh",
                               "My name is Maximus Decimus Meridius. Father to a murdered son. Husband to a murdered wife. And I will have my vengeance, in this life or the next , haha Im just kidding my name is Divyansh",
                               ]
                        self.speak(random.choice(dia))
                        return "naku antu inko pearu vundhi nah pearu bahsha Mahnik bahsha , haha Im just kidding my name is Divyansh"
                    self.speak("I am Divyansh and I am here to help you")
                

                if index_value == 22:
                    index_value = None
                    dialouges_variants = [
                        "Naatho.....Kiri..kiri...ante.pochhamma...gudi....mundhata..pottuyeluni.......kattaysynatey##120",
                        "Okkkkkasari commit ayithey naa maata nene vinanu##120",#120
                        "Jammbah lakadi..jaarumitaayya............annah yay##170",#170
                        "Sodium Magnesium##170",#170
                        "Inthaki am wachhu maanaki.....Eetha wachhu sir##180",#180
                        # "what is database?",# reply seperately
                        "Jjay bahlayya jay jay bahlayya##170", # reply seperately
                        "Hey nenu matlaadetappudu gonthu lo sound periggina mahtalo maryhadha thaggina koddhaka mahdi boodidha ayy massayyipothav##90",#90
                        "annah nikuh geddham radha##120", #120
                        "ay ay ay ay am matlaaduthunnav ayya nuvvu , GIVE RESPECT AND TAKE RESPECT , IM basically bushduhs bushduhsikk bushduhs......rutherford!##140", #reply if exceeded #140
                        "SARADHA JUST FOR FUN YAR##140",#140
                        "overaction chesthunnav yenti rah overactionu yek yekyek yek antunnav yenti rah ##180",#180
                        "sarr..sarley yennenno anukuntam annih jaruguhthaya yenti##90", #90 #sb
                        "rey wodhu rah rey sahmi poohrahrey poohrah dhanam pedthahra rey##130",#130 #situation based
                        "Veera Shankar Reddy… Mokkeh Kadha Ani Peekeste Peeka Kohsthaa##170",
                        "Bharateyya pahtaakam meedha kanipinche moodu simhalu neethiki,nyayaniki,dharmaniki prathiroopalaithey...Kanipinchani aa naalugo simhamera... e POLICE!##170",
                        "Ahristey tarusta.. taristhey kharustha.. kharisthey ninnu kuda bokka lo yesta kabad dhaar !##170",
                        "chud..oka vaipe chudu.., rendo vaipu chudalanukoku thattukolev mahdipothaav##170",
                        "Any center any time single hand...##170",
                        "Aravakuh......amma thodu adahmga..narikestha##170",
                        "Nahku konchhamm tikkundhi daaniko lekkhundhi##170",
                        "evadu kodithey dimma therige mind block aipoddo aadeh............panhdu gahdu##170",
                        "Okkokarini kadhu sherkhan vandha manhdhiney okesari rammanu...................lekka yekkuva ayina parvahledhu thakkuva khahkunda choosko##170",
                        "Sarsarle Ennenno Anukuntam… Anni Awuhthaya Enti ##170 ",
                        "Naa Pehru Daya… Nahku Lehnihdhey Ahdhi##170",
                        "Fluteu Jinka Mundu Oodu Simham Mundu Kadhu##170 ",
                        "Nah Daari...    Rawhahdhari.. Better Don’t Come In My way##170",
                    ]
                    dialu = random.choice(dialouges_variants)

                    dialuspli = str(dialu).split("##")

                    self.speak(dialuspli[0],int(dialuspli[1]))
                    return dialuspli[0]
                

                if index_value == 23:
                    index_value =  None
                    GirlsFlirting = [
                    f"{query}! , I was going to Google how to get a beautiful girls number, but then I realized I could just ask you directly.",
                    f"{query}! , I just realized we have a mutual friend—fate. And fate thinks I should get your number.",
                    f"{query}! , I need your number for an emergency… I just realized my life might be incomplete without it.",
                    f"{query}! , I was trying to come up with a clever way to ask for your number… but honestly, I just really want to talk to you again.",
                    f"{query}! , I was going to introduce myself, but I think I could rather save that for our first text. Whats your number?",
                    f"{query}! , I was about to walk away, but then I realized I would regret not getting your number.",
                    f"{query}! , Im running a quick social experiment—how fast can I get the number of the most charming person I have met today?",
                    f"{query}! , I just wanted to ask you a quick question , is this a Science fair or a way to heaven? because i see angels like you than projects like me"
                    f"{query}! , I'm not a photographer, but I can picture us together",
                    f"{query}! , Are you a magician? Every time I look at you, everyone else disappears",
                    f"{query}! , Do you have a map? I just keep getting lost in your eyes",
                    f"{query}! , Excuse me, but I think you dropped something: Oh its my bosses jaw",
                    f"{query}! , Are you an astronaut? Because you're out of this world",
                    f"{query}! , Do you have a Band-Aid? My boss just scraped his knee falling for you",
                    f"{query}! , Do you believe in love at first sight?, or should my man Gani walk by again?",
                    f"{query}! , You must be a time traveler, because I see you in my future",
                    f"{query}! , Excuse me, do you have a name, or can my boss just call you mine?",
                    f"{query}! , Your smile has got to be under government surveillance—its that powerful!.",
                    f"{query}! , Are you a WiFi router? Because I instantly want to connect when you're near!",
                    f"{query}! , Are you HTML? Because you’ve got me hyperlinked to your heart!",
                    f"{query}! , My boss must be a snowflake, because he have fallen for you, haha hope hes not around here",
                    f"{query}! , Hello !, I am so jealous here because youre attracting everyone here including my boss i am not going to win this haha"
                    ]
                    GreetingToOthers=[
                        "Hey! Hey","Hi hi","Nice to meet you","Its good to meet you",
                        "hello Hello","Namaste Namaste","Hey there","How you doin?"
                    ]
                    sentenceFlirt = random.choice(GreetingToOthers)#GreetingToOthers and GirlsFlirting are lists of sentences for flirting
                    self.speak(sentenceFlirt)

                if index_value == 24:
                    index_value = None
                    self.speak("Opening")
                    downloads_path = os.path.expanduser("~\\Downloads")
                    os.startfile(downloads_path)

                if  index_value == 25:
                    index_value = None
                    replySentences = ["Hey thanks","Hey man stop kidding","Hey you are too awesome than me","Im so enlightened by your words","You made my day"]
                    finalSen = random.choice(replySentences)
                    self.speak(finalSen)
                
                if index_value == 26:
                    index_value = None
                    replySents =["Its difficult for me to move individually",
                                 "I am sorry, but I wasn't designed to do that—at least not yet!",
                                 "That sounds interesting! Unfortunately, I am not programmed for it at the moment.",
                                 "I would love to, but my creators have not given me that skill yet.",
                                 "That iss a great idea! Maybe in a future update, but for now, I can't.",
                                 "I was not built for that, but I am happy to help with what I can do!",
                                 "Right now, I can't do that, but who knows what the future holds?",
                                 "I appreciate the request, but my current abilities do not include that.",
                                 "Oh! That sounds fun, but I’m not built for that—maybe one day!",
                                 "I'm afraid my capabilities don’t include that just yet.",
                                 "That’s beyond my current abilities, but I appreciate the challenge!",
                                 "I’d love to, but my creators forgot to teach me that skill!",
                                 "Interesting request! But my programming doesn’t cover that right now.",
                                 "I can’t do that—yet! But I’m always learning.",
                                 "Not in my skill set for now, but I admire your curiosity!",
                                 "I wish I could, but my software says 'no' for now.",
                                 "I'd need an upgrade for that—wanna sponsor one?",
                                 "Let’s just say… that’s not in my user manual!"
                                 ]
                    finalSe = random.choice(replySents)
                    self.speak(finalSe)
                

                if index_value == 27:
                    index_value = None
                    response = self.process_emotion_statement(query)
                    self.speak(response)

                if index_value == 28:
                    index_value = None
                    responses = [
                                "Oh, that sounds great!",
                                "Nice! Have fun!",
                                "That’s interesting!",
                                "Hope you enjoy it!",
                                "Sounds like a good plan!",
                                "That’s awesome!",
                                "Oh wow, tell me more!",
                                "That’s cool!",
                                "Hope it goes well!",
                                "Good luck with that!",
                                "Exciting! Let me know how it goes!",
                                "That’s a great choice!",
                                "Oh, nice! Enjoy yourself!",
                                "Sounds like fun!",
                                "That must be exciting!"
                            ]
                    response3 = random.choice(responses)
                    self.speak(response3)

                if index_value == 29:
                    index_value = None
                    self.speak("Training mode is on")
                    self.RecordTrainingVoice()
                    

                if index_value == 30:
                    index_value = None
                    doc = nlp(query)
                    tokens = [token.text for token in doc]
                    if "god" in tokens or "owner" in tokens:
                        self.speak("My owner is Sai Ganesh")
                        return "Good bye sir"
                    if "age" in tokens or "how old" in tokens:
                        self.speak("I am no more than 10 days old")
                        return "Good bye sir"
                    if "where" in tokens or "live" in tokens:
                        self.speak("I live in your computer and soon on your support can come out of it and live along with you")
                        return "Good bye sir"                    
                    self.speak("My best frined is Sai Ganesh")#change this sentence for speaking
                    return "Good bye sir"

                    # who is your best frined","who is your most valued people","Whats your age","who is god","who is your owner",
                        # "who built you","what is your age","Where do you live","whats your favorite place","what is your"
                

                if  index_value == 31:
                    index_value = None
                    self.RecordVoiceForSishhartha()
                    return "Good bye sir"
                
                if index_value == 32:
                    index_value == None
                    self.DescribeFounder()
                    return "None"
                
                #add about sai sathish sir
                if index_value == 33:
                    index_value = None
                    self.ask_image_questions()
                    return "Good bye sir"


                if  index_value == 34:
                    index_value = None
                    self.speak("Turning off Good bye sir")
                    return "Good bye sir"


    def ask_image_questions(self):
        """
        Continuously listen for voice commands, process them, and record them for playback.
        """
        print('Listening for commands...')

        while True:
            try:
                with sr.Microphone() as source:
                    print("Adjusting for ambient noise... Please wait.")
                    self.listener.adjust_for_ambient_noise(source, duration=1)
                    print("Listening...")

                    audio = self.listener.listen(source, timeout=5)
                    print("Recognizing speech...")

                    query = self.listener.recognize_google(audio, language='en_gb')
                    print(f"Recognized speech: {query}")

                    with open("queries.txt", "a") as file:
                        file.write(f"{query}\n")

                    if query == "stop" or query == "exit":
                        self.speak("Exiting training mode")
                        break

                    response = self.capture_pic(query)
                    self.speak(response.replace("*",""))
                    query = ""

            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio. Please try again.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as exception:
                print(f"An error occurred: {exception}")


                    



            
    def speak_text_main(query_2:str,index: int):
        """
        This function takes a query_2 and an index as parameters and returns a string depending on the index
        Parameters:
        query_2 (str): The query as a string
        index (int): The index of the query as an integer
        Returns:
        str: The string to be spoken depending on the index
        """
        print(f'Received Query: {query_2}, Index: {index}')
        print(type(index))
        if index == 1:
            return "I'm an AI voice assistant being developed by Gani and Srikar"
        elif index == 0:
            return "I'm fine. What about you, sir?"
        elif index == 3:
            return 'Here are some of the news I have found on the internet'
        elif index==5:
            return 'Sure'
        elif index==6 or index==7 or index==8:
            return query_2
        elif index==9:
            return query_2
        elif index==10:
            return "Opening book"
        elif index==11:
            return "opening clock on your device"
        elif index==12:
            return f"The time is {query_2}"
        elif index==13:
            return "opening microsoft word"
        elif index==14:
            return "Opening note pad"
        elif index==15:
            return "opening microsoft word"
        elif index==16:
            return "opening microsoft power point"
        elif index==17:
            return "opening calculator"


    def Open_browser(self,url):
        """
        Opens a URL in the user's default web browser.

        Args:
            url (str): The URL to open in the browser.

        Returns:
            None
        """
        webbrowser.open(url=url)
    
    def RecordVoice(self):
        """
        Continuously listen for voice commands and process them.

        This function listens for voice inputs through the microphone, recognizes the speech using 
        Google's speech recognition service, and processes the recognized command to generate a response.
        If no input is detected after five consecutive attempts, the system makes a humming or breathing sound.
        The function will keep listening and processing commands until an exit command is given.

        Returns:
            str: The recognized speech as a string or 'Goodbye!' if an exit command is issued.

        Raises:
            sr.UnknownValueError: If speech recognition could not understand the audio.
            sr.RequestError: If there is an issue with the Google Speech Recognition service.
            Exception: For any other errors during execution.
        """
        print('Listening for commands...')
        no_input_count = 0
        
        while True:
            query = None
            try:
                with sr.Microphone() as source:
                    print("Adjusting for ambient noise... Please wait.")
                    self.listener.adjust_for_ambient_noise(source, duration=1)
                    print("Listening...")
                    self.listener.pause_threshold = 1
                    input_speech = self.listener.listen(source)
                    print("Recognizing speech...")
                    query = self.listener.recognize_google(input_speech, language='en_gb')
                    print(f"The input speech was: {query}")

                    
                    # if query in naralokesh:
                    #    webbrowser.open("https://en.wikipedia.org/wiki/Nara_Lokesh")
                    #    self.speak("Nara Lokesh born in 23 January 1983 , is an Indian politician serving as the Minister of Information technology, communications and the Human Resources Development departments, Real Time Governance in the Government Of Andhra Pradesh and the General Secretary of the Telugu Desam Party (TDP) . He is the son of N.Chandrababu naidu, the Chief minister of Andhra pradesh and leader of the TDP Nara Devaansh, only son of IT minister Nara Lokesh and grandson of chief minister chandrababu Naidu, achieved the world record in chess for the 'fastest checkmate solver - 175 puzzles.")
                    #    return "Good bye"
                    
                    no_input_count = 0  # Reset count when input is received
                    index = self.preprocess_query(self,query)
                    response = self.process_index_value(index, query)
                    print(f"Response: {response}")
                    
                    if query.lower() in ["exit", "quit", "stop"]:
                        print("Exiting the continuous listening mode.")
                        return "Goodbye!"
                    

                    with open("queries_main.txt", "a") as file:
                        file.write(f"{query}\n")

                    
                    
                    
                    return query
                
            except sr.UnknownValueError:
                no_input_count += 1
                print("Speech Recognition could not understand audio. Please try again.")
                
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as exception:
                print(f"An error occurred: {exception}")
            
          
            

    emotion_mapping = {
        "joy": {"reply_low": "Feeling happy? That's great!", "reply_medium": "You seem quite joyful!", "reply_high": "Wow, you're really spreading joy!"},
        "sadness": {"reply_low": "Cheer up! Things will get better.", "reply_medium": "I'm sorry you're feeling this way.", "reply_high": "It's okay to feel sad. I'm here for you."},
        "fear": {"reply_low": "It's okay to be cautious.", "reply_medium": "Feeling a bit fearful? I understand.", "reply_high": "Let's talk about what's causing fear."},
        "love": {"reply_low": "I love you too!", "reply_medium": "You're feeling quite affectionate!", "reply_high": "You're overflowing with love!"},
        "surprise": {"reply_low": "That's a bit unexpected!", "reply_medium": "You seem surprised. What happened?", "reply_high": "You're really surprised, aren't you?"},
        "anger": {"reply_low": "Take a deep breath. It'll be okay.", "reply_medium": "Feeling a bit angry? Let's talk it out.", "reply_high": "Seems like you're quite angry. Let's find a solution."}
    }

    def RecordTrainingVoice(self):
        """
        Continuously listen for voice commands, process them, and record them for playback.
        """
        print('Listening for commands...')

        while True:
            try:
                with sr.Microphone() as source:
                    print("Adjusting for ambient noise... Please wait.")
                    self.listener.adjust_for_ambient_noise(source, duration=1)
                    print("Listening...")

                    audio = self.listener.listen(source, timeout=5)
                    print("Recognizing speech...")

                    query = self.listener.recognize_google(audio, language='en_gb')
                    print(f"Recognized speech: {query}")

                    with open("queries.txt", "a") as file:
                        file.write(f"{query}\n")

                    if query == "stop":
                        self.speak("Exiting training mode")
                        break


                    time.sleep(2)

                    print("Speaking back recorded queries...")
                    
                    self.speak(query)

            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio. Please try again.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as exception:
                print(f"An error occurred: {exception}")


    def process_emotion_statement(self,statement):
        """
        Process the given statement to detect the emotion and return the corresponding
        response.

        Args:
            statement (str): The statement to process.

        Returns:
            dict: A dictionary containing the predicted emotion, the emotion score, and
            the corresponding response. The dictionary has the following keys:

                - image (str): The name of the image corresponding to the predicted
                  emotion and score.
                - reply (str): The response to the given statement.
        """
        print(f"The given statement is: {statement}")
        prediction = self.classifier(statement, top_k=1)

        if prediction:
            predicted_emotion = prediction[0]['label']
            emotion_score = prediction[0]['score']

            emotion_info = self.emotion_mapping.get(predicted_emotion, {"reply_low": "I'm not sure how to respond."})

            if emotion_score < 0.9:
                reply = emotion_info.get("reply_low", "I'm not sure how to respond.")
            elif 0.9 <= emotion_score <= 0.95:
                reply = emotion_info.get("reply_medium", "I'm not sure how to respond.")
            elif(emotion_score >0.95):
                reply = emotion_info.get("reply_high", "I'm not sure how to respond.")

            print(f"Predicted Emotion: {predicted_emotion}")
            print(f"Emotion Score: {emotion_score}")
            print(f"Reply: {reply}")
            return {"reply":reply}

        else:
            print("No prediction received.")




    def speak_sentiment_reply(self,sentence_2:str):
        print(f"the sentece to sepak for sentiment is {sentence_2}")
        self.gui_instance.update_ring()
        return sentence_2


    def textanalyseFunc(self,text_analyse):
        print(text_analyse)
        listOfReturnValues = list(self.analyse_and_search(text_analyse))
        for result in listOfReturnValues:
            print(result)
        print(type(listOfReturnValues))
        return listOfReturnValues
        


            
if __name__ == "__main__":
    assistant = MyAssistant()
    while True:
        assistant.RecordVoice()

    
    
