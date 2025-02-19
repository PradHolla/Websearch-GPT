### properly format the code
### solve bug of diff dimensions in the retrieved snippets vs sentences similarities
### solve bug when retrieving thumbnail metadata
# solve bug for second turn chatonwards
### consolidate model selection in UI and save variables
### split response in two, one to apply the rag while referencing, big models can do this, and then another to apply css styling, can be a fast model.

import os
# os.environ['HF_HOME'] = '/data1/demobot/hf'
# Type nvidia-smi in the terminal and choose two GPUs that is not being used, remember that we can only use 0,1,2,3,4
# cuda_llama = 0
# cuda_llava = 4
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_llama},{cuda_llava}"
# os.environ["TRANSFORMERS_CACHE"] = '/data1/demobot/hf'
import uuid
import re
import json
import numpy as np
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import http.client
http.client._MAXHEADERS = 1000

import scripts.utils as utils
import streamlit as st

# import torch
# from huggingface_hub import login
from sentence_transformers import SentenceTransformer
# from sentence_transformers.quantization import quantize_embeddings,semantic_search_usearch
# from llama_cpp import Llama
# from llama_cpp.llama_chat_format import Llama3VisionAlpha
from groq import Groq

import ast
import requests
from PIL import Image
from io import BytesIO
import base64

# from dotenv import load_dotenv
# load_dotenv()


#login hf key to use llama models
# HF_TOKEN = os.getenv("HF_TOKEN")
# YOU_API_KEY = os.getenv("YOU_API_KEY")
# login(HF_TOKEN)

# Page configs
st.set_page_config(page_title="🌐 Internet Bot",
                page_icon="💬",
                layout='wide')

# DEMOBOT_HOME = os.getenv("DEMOBOT_HOME")
# Loading CSS Style
with open(os.path.join(st.secrets["DEMO_PATH"], "scripts/style.css"), "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.header('🌐 Web Grounded Chatbot')
st.header('\n')

class LLMNET:
    def __init__(self, generation_model, embedding_model):
        self.model_id = "/data1/demobot/hf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        self.similarity_model_id = "all-mpnet-base-v2"
        self.similarity_model_id2 = "nvidia/NV-Embed-v2"
        self.embed_model_id = "mixedbread-ai/mxbai-embed-large-v1"
        # self.cache_dir = '/data1/demobot/hf'
        self.cache_dir = st.secrets["CACHE_DIR"]
        # self.cache_folder = '/data1/demobot/hf'
        self.cache_folder = st.secrets["CACHE_DIR"]
        self.generation_model = generation_model
        self.you_api_key = st.secrets["YOU_API_KEY"]
        # self.log_writer('START', 'create_log')
        
        
    # def log_writer(self, messages, log_type , generation_type = None):
    #     """
    #     Args:
    #         messages (list): messages to be logged
    #         type (str): type of the generation request
    #     """
        
    #     if log_type == 'create_log':
    #         if not os.path.exists(f"{DEMOBOT_HOME}/logs/flow_conversation/"):
    #             os.makedirs(f"{DEMOBOT_HOME}/logs/flow_conversation/")
            
    #         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'w') as file:
    #             file.write(f"{messages}\n")
    #             dashes = '-' * 20
    #             file.write(f"{dashes}\n")
                
    #     if log_type == 'messages_sent':
    #         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
    #             file.write(f"\nMessages sent to '{generation_type}' generation:\n\n")
    #             for msg in messages:
    #                 file.write(f"+++{msg['role']}: {msg['content']}\n")
    #             file.write(f"\n\n")
    #             dashes = '-' * 20
    #             file.write(f"{dashes}\n") 
            
    #     if log_type == 'general':
    #         with open(f"{DEMOBOT_HOME}/logs/flow_conversation/{st.session_state.session_id}.txt", 'a') as file:
    #             file.write(f"{messages}\n\n")
    #             dashes = '-' * 20
    #             file.write(f"{dashes}\n") 

    def load_sys_prompts(self, prompt_name):
        if prompt_name == 'router':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_router_LLMNET.json"), 'r') as file:
                prompt = json.load(file)
                
        if prompt_name == 'chat':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_chat_LLMNET.json"), 'r') as file:
                prompt = json.load(file)
                
        if prompt_name == 'create_query':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_create_query_LLMNET.json"), 'r') as file:
                prompt = json.load(file)
                
        if prompt_name == 'respond_query':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_respond_query_LLMNET2.json"), 'r') as file:
                prompt = json.load(file)
                
        if prompt_name == 'html_format':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_html_format_LLMNET.json"), 'r') as file:
                prompt = json.load(file)
                
        if prompt_name == 'color_code':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_color_code_LLMNET.json"), 'r') as file:
                prompt = json.load(file)
                
        if prompt_name == 'score_websites':
            with open(os.path.join(st.secrets["DEMO_PATH"], "prompts/internet/sys_prompt_source_scorer_LLMNET.json"), 'r') as file:
                prompt = json.load(file)

        return prompt

    def setup_model(self, model_type):
        # if model_type == 'llama':
        #     @st.cache_resource
        #     def load_llama():
        #         llm = Llama(
        #             model_path=self.model_id,
        #             chat_format = 'llama-3',
        #             n_gpu_layers=-1,
        #             device = f"cuda:{cuda_llama}",
        #             verbose = False,
        #             main_gpu = 0,
        #             n_ctx = 0
        #             # seed=1337, # Uncomment to set a specific seed
        #             )
        #         return llm
        #     return load_llama()
        
        if model_type == 'groq':
            @st.cache_resource
            def load_groq_client():
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                return client
            return load_groq_client()
        
        if model_type == 'similarity_base':
            @st.cache_resource
            def load_similarity_model():
                model = SentenceTransformer(self.similarity_model_id, cache_folder = self.cache_folder)
                return model
            return load_similarity_model()
        
        if model_type == 'similarity2':
            @st.cache_resource
            def load_similarity_model():
                model = AutoModel.from_pretrained(self.similarity_model_id2, trust_remote_code=True, cache_dir = self.cache_dir)
                return model
            return load_similarity_model()
        
    
    def format_search_query(self, bot_response):
        """
        Args:
            bot_response (str): Search Query generated by the bot
        Returns:
            formatted_bot_response: str : Formatted Search Query, stripped from the prefix: Search Query: 
        """
        search_query_formatted = None
        pattern = r"Search Query: (.*)"

        match = re.search(pattern, bot_response)
        if match:
            search_query_formatted = match.group(1).strip()

        return search_query_formatted
    
    def get_api_snippets_for_query(self, query):
        """
        Search the web for the query
        Args:
            query (str): Query to search for
        Returns:
            dict: Query result
        """
        
        query = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', query)
        
        headers = {"X-API-Key": self.you_api_key}
        print(headers)
        params = {"query": query, "num_web_results": 5}
        
        # with open(os.path.join(st.secrets["DEMO_PATH"], "logs/flow_conversation/{st.session_state.session_id}.txt"), 'a') as file:
        #         file.write(f"API_key used: {st.secrets['YOU_API_KEY']}\n\n")
        #         file.write(f"Query Sent to the search API: {query}\n\n")
        #         dashes = '-' * 20
        #         file.write(f"{dashes}\n")
                
        search_results = requests.get(
                            f"https://api.ydc-index.io/search?query={query}",
                            params=params,
                            headers=headers,
                            ).json()
        
            
        return search_results['hits']
    

    def search_retrieve_web(self, user):
        """
        Handles the search, summary and image retrieval from the web
        Args:
            user (str): User input

        Returns:
            str: query result
            str:  summary of the query result
            list: images retrieved from the web
            dict: sources of the information retrieved
        """

        # Create Query
        search_query = self.prepare_and_generate(user, generation_type='create_query')
        # user_prompt_create_query, sys_prompt_create_query = self.format_user_sys_prompts(user, type = 'create_query')
        # messages_create_query = self.messages_builder(sys_prompt_create_query, user_prompt_create_query, type = 'create_query')
        # search_query_raw = self.generate(messages_create_query,)
        # self.log_writer(f"RAW Query created: {search_query_raw}", 'general')
        # search_query = self.format_response(search_query_raw, format_type='drop_assistant')
        # self.log_writer(f"Formatted Query created: {search_query}", 'general')
        
        # Search the web
        query_result = self.get_api_snippets_for_query(search_query)
        # self.log_writer(f"Retrieved information from the web: {query_result}", 'general')
        
        # create dict with the sources information, keys are the website prefixes
        all_snippets = ''
        dict_sources = dict()
        all_websites_prefix = []
        # max_snippets = 0
        for result in query_result:
            website = result['url']
            snippets = result['snippets']
            # max_snippets = max(max_snippets, len(snippets))
            
            # match = re.search(r'^(https?://[^/]+\.com)', website)
            match = re.search(r'^(https?://[^/]+\.[a-z]{2,})', website)
            website_prefix = match.group(1) if match else None
            
            all_snippets += "Source website for below: " + website_prefix + '\n'
            for snippet in snippets:
                all_snippets += snippet + '\n'
            all_snippets += '\n\n'   
            
                
            if website_prefix not in dict_sources:
                all_websites_prefix.append(website_prefix)
                # get the metadata from the website
                metadata = self.get_page_metadata(website)
                if metadata:
                    thumbnail_url = metadata['thumbnail_url']
                    title = metadata['title']
                    description = metadata['description']
                dict_sources[website_prefix] = {'snippets': snippets,
                                                'url': result['url'],
                                                'thumbnail_url': thumbnail_url,
                                                'title': title
                                                }
        # no source can contain a diff number of snippets, due to limitation o
        # for website in dict_sources.keys():
        #     if len(dict_sources[website]['snippets']) < max_snippets:
        #         del dict_sources[website]
        
        scoring_websites_prompt = f"Search Query: {search_query}\nList of Websites: {all_websites_prefix}"
        
        scored_websites_str = self.prepare_and_generate(scoring_websites_prompt, generation_type='score_websites')
        scored_websites_dict = json.loads(scored_websites_str)
        for key, value in scored_websites_dict.items():
            dict_sources[key]['credibility'] = value
            
        # self.log_writer(f"Dict Sources: {dict_sources}", 'general')
        
        
                
        # create a popover with the sources
        with st.popover(f"Sources"):
            for source in dict_sources:
                st.markdown(f"Source: {source}")
                # st.markdown(f"Description: {dict_sources[source]['description']}")
                for snippet in dict_sources[source]['snippets']:
                    st.markdown(f"- {snippet}")
                st.markdown(f"url: {dict_sources[source]['url']}\n\n")
                st.markdown(f"-----------------------------------\n\n")
                    

        return query_result, dict_sources, all_snippets
    
    
    def format_user_sys_prompts(self, user, generation_type, query_result = None):
        """
        Args:
            user (str): Depending on the type of task: either last user, conversation or search results
            type (str): type of the generation request

        Returns:
            dict: user_prompt
            dict: sys_prompt
            Returns the formatted user and sys dict used for the LLM generation according to the task type
        """
        
        if generation_type == 'router':
            ### The Router decides if in the current turn of conversation, the chatbot is going to perform a normal reply or search the web and make a RAG
            #TODO Currently, the router only receives the last user message, it would be interesting to check if including a window of conversation History would be beneficial
            item_summary = None
            user_prompt = {"role": "user", "content": f"{user}"}

            # loading router prompt
            sys_prompt = self.load_sys_prompts('router')

        
        # The chat responder is used whenever the router has decided that a simple reply, without search is the next best action
        # TODO: The prompt here includes instruction on how to deal with question such as "who created you", artificially pointing to stevens
        elif generation_type == 'chat':
            # fig_description = None
            user_prompt = {"role": "user", "content": f"{user}"}
            sys_prompt = self.load_sys_prompts('chat')
            
        
        # If the Router has decided to Search before replying, a new generation has to be made: the Search Query, from the conversation histpry
        elif generation_type == 'create_query':
            user_prompt = {"role": "user", "content": f"{user}"}
            #TODO understand if I need artificially include rows in the history such as  user_prompt = {"role": "user", "content": f"Please create a search query."}
            #TODO how does the search query creator deal with conversation history? should it be a window?
            sys_prompt = self.load_sys_prompts('create_query')
        
        # After the Search Query generation and information retrieval through the webAPI, we need to provide the LLM with the conversation + retrieval result so that it can generate a grounded response
        elif generation_type == 'respond_query':
            #TODO Should I include this artificial turn?
            # user_prompt = {"role": "user", "content": "Nice!"}
            # WE are including the retrieved documents in the system message, in this case, the user is what has been retrieved from the web
            sys_prompt = self.load_sys_prompts('respond_query')
            sys_prompt['content'] = sys_prompt['content'].format(query_result)
            
            user_prompt = {"role": "user", "content": f"{user}"}
        
        elif generation_type == 'html_format':
            sys_prompt = self.load_sys_prompts('html_format')
            user_prompt = {"role": "user", "content": f"{user}"}
            
        elif generation_type == 'color_code':
            sys_prompt = self.load_sys_prompts('color_code')
            user_prompt = {"role": "user", "content": f"{user}"}

        elif generation_type == 'score_websites':
            sys_prompt = self.load_sys_prompts('score_websites')
            user_prompt = {"role": "user", "content": f"{user}"}
            
            
        return user_prompt, sys_prompt
        
    
    def messages_builder(self, sys_prompt, user_prompt, generation_type = None):
        """
        Args:
            sys_prompt (dict): System Prompt
            user_prompt (dict): User Prompt
            type (str): Type of the generation request

        Returns:
            list: messages to be sent to the LLM generation
        """
        messages = [sys_prompt]
        for msg in st.session_state["messages"][-6:-1]:
                messages.append({
                                'role': msg['role'],
                                'content': msg['content']
                                })
        messages.append(user_prompt)
        # self.log_writer(messages, log_type='messages_sent', generation_type=generation_type)
        
        return messages


    def generate(self, messages):
        
        if self.generation_model == 'llama3':
            response = llm.create_chat_completion(
                max_tokens = 512,
                temperature = 0.2,
                top_p=0.9,
                messages = messages
                )
            return response['choices'][0]['message']['content']
        
        elif self.generation_model == 'groq':
            chat_completion = llm.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,)
            
            return chat_completion.choices[0].message.content
            
        
    def replace_with_tooltip(self, match, dict_sources):
        full_url = match.group(1)
        # domain_name = match.group(2)
        # print('full_url',full_url)
        # print('domain_name',domain_name)
        # Look up the source using the domain name (e.g., eatingwell.com)
        source_details = dict_sources.get(full_url, {})
        print(full_url)
        
        if source_details:
            return f"""
            <a href='{source_details['url']}' style='text-decoration: underline; cursor: pointer;'>
                <div class="tooltip">
                    [Reference]
                    <span class="tooltiptext" style="padding: 10px;">
                        <strong style="margin-bottom: 5px; display: block;">{source_details['title']}</strong>
                        <img src='{source_details['thumbnail_url']}' alt='Thumbnail' width='100'><br>
                        <br>
                        <span style='color: #fff;'>{full_url}</span>
                    </span>
                </div>
            </a>
            """
        return match.group(0)
    
    def format_response(self, sentence, format_type, formatted_response = None, dict_sources = None, max_source = None):
        if format_type == 'drop_assistant':
            sentence =re.sub(r'\$', '\$', sentence)
            if sentence.startswith('Assistant') or sentence.startswith('assistant'):
                return sentence[len('Assistant: '): ]
            else:
                return sentence
            
        elif format_type == 'tool_tip':
            dict_sources = dict_sources
            # pattern = r'\(<a href="(https://(?:www\.)?[a-zA-Z0-9\-]+\.[a-z]{2,3})">([^<]+)</a>\)'
            # pattern = r'\(<a href="(https://(?:www\.)?[a-zA-Z0-9\-]+\.[a-z]{2,3})"(?: style="[^"]*")?>[^<]+</a>\)'
            # pattern = r'\(<a href="(https?://(?:www\.)?[a-zA-Z0-9\-]+\.[a-z]{2,3})"(?: style="[^"]*")?>[^<]+</a>\)'
            pattern = r'\(<a href="(https?://(?:www\.)?[a-zA-Z0-9\-]+\.[a-z]{2,3}/?)"(?: style="[^"]*")?>[^<]+</a>\)'

            sentence = re.sub(pattern, lambda match: self.replace_with_tooltip(match, dict_sources), sentence)
            return sentence

            
        elif format_type == 'format_html':
        
            # if sentence.strip().startswith(tuple(str(i) for i in range(1, 10))):
                # Use an ordered list for formatting
                sentence = re.sub(r'^\d+\.\s*', '', sentence)
                
                # formatted_response += f"<li>{sentence} <span title='Source: {max_source} Snippet: {snippet}' style='text-decoration: underline; cursor: pointer;'>[reference]</span></li>\n"
                # formatted_response += f"<li>{sentence} <a href='{dict_sources[max_source]['url']}' title='Snippet: {snippet}' style='text-decoration: underline; cursor: pointer;'>[reference]</a></li>\n"
                formatted_response += f"""
                                        <li>{sentence} 
                                            <a href='{dict_sources[max_source]['url']}' style='text-decoration: underline; cursor: pointer;'>
                                            <div class="tooltip">
                                                [reference]
                                                <span class="tooltiptext" style="padding: 10px;">
                                                    <strong style="margin-bottom: 5px; display: block;">{dict_sources[max_source]['title']}</strong>
                                                    <img src='{dict_sources[max_source]['thumbnail_url']}' alt='Thumbnail' width='100'><br>
                                                    <br>
                                                    <span style='color: #fff;'>{max_source}</span>
                                                </span>
                                            </div>
                                        </a>
                                    </li>"""
            # else:
            #     if formatted_response == "":
            #         formatted_response += f"<p>{sentence}</p>"
            #     else:
            #         formatted_response += f"<br><p>{sentence}</p>"
        
        return formatted_response
        
    def prepare_and_generate(self, user, generation_type, query_result = None):
        user_prompt, sys_prompt = self.format_user_sys_prompts(user, generation_type = generation_type, query_result = query_result)
        messages = self.messages_builder(sys_prompt, user_prompt, generation_type = generation_type)
        response = self.generate(messages)
        response = self.format_response(response, format_type='drop_assistant')
        return response


    def chatLlama3(self, user):
        """
        Handles the router and the response from the LLM
        Args:
            user (str): User input

        Returns:
            str: Response from the LLM
        """
        response, query_result, dict_sources = '', '', ''
        ### Generate router response
        ### build prompts to route between response or query for lookup operation
        router_response = self.prepare_and_generate(user, generation_type = 'router')
        # self.log_writer(f"Router decision: {router_response}\n\n", 'general')

        ### Analyze router decision and propagate next step
        # in the case of SEARCH action
        if router_response.startswith('SEARCH') or router_response.startswith(' SEARCH') or "SEARCH" in router_response:
            
            # First search for snippets relevant to the user query
            query_result, dict_sources, all_snippets = self.search_retrieve_web(user)
            # Produce the response based on the query result and conversation
            # In this case, we give the retrieved items as context to the bot for it to generate a grounded response
            response = self.prepare_and_generate(user, generation_type = 'respond_query', query_result = all_snippets)
            # self.log_writer(f"Response generated through RAG: {response}\n\n", 'general') 

        # In the case of simple chat
        else:
            response = self.prepare_and_generate(user, generation_type = 'chat')
            # self.log_writer(f"Response generated through normal chat: {response}\n\n", 'general')
            
        # trying now to apply html styling through a model
        html_response = self.prepare_and_generate(response, generation_type='html_format')
        # self.log_writer(f"HTML Response: {html_response}\n\n", 'general')
        
        # after html editing, we need to color code the sentences according to the source websites credibility
        if query_result:
            all_scores = 'Website Scores\n'
            for key in dict_sources.keys():
                all_scores += key + ':' + str(dict_sources[key]['credibility']) + '\n'
                
            
            color_coded_response = self.prepare_and_generate(html_response + '\n' + f"{all_scores}", generation_type='color_code')
            response = color_coded_response
            # self.log_writer(f"Color coded response: {color_coded_response}", 'general')
            
        return response, query_result, dict_sources
    
    def similarity_search3(self, response, dict_sources):
        response_sentences = response.split('\n')  # Split response into sentences
        embeddings_response_sentences = similarity_model.encode(response_sentences)
        # embeddings_response_sentences = F.normalize(embeddings_response_sentences, p=2, dim=1)
        
        similarities = {}
        max_max_dict = {}
        max_across_all = [-1] * len(response_sentences)
        # Iterate through each source in the query result
        for source in dict_sources:
            snippets = dict_sources[source]['snippets']
            if not snippets:
                continue
            
            embeddings_passages = similarity_model.encode(snippets)  # Encode snippets
            # embeddings_passages = F.normalize(embeddings_passages, p=2, dim=1)
            # Calculate similarities for each response sentence with the snippets
            similarities_per_source = similarity_model.similarity(embeddings_passages, embeddings_response_sentences)
            similarities[source] = [[round(similarity.item(), 2) for similarity in snippet_similarities] for snippet_similarities in similarities_per_source]
            
            array = np.array(similarities[source])
            # Find the maximum values across the vertical axis (columns)
            max_value_across_columns = np.max(array, axis=0)
            max_index_across_columns = np.argmax(array, axis=0)
            for i, max_per_sentence in enumerate(max_value_across_columns):
                if max_per_sentence > max_across_all[i]:
                    max_across_all[i] = max_per_sentence
                    max_max_dict[i] = {'source': source, 'value': max_across_all[i], 'snippet':snippets[max_index_across_columns[i]]}
                
                
            
            # similarities_per_source = (embeddings_response_sentences @ embeddings_passages.T).tolist()
            # similarities[source] = [[round(similarity, 2) for similarity in snippet_similarities] for snippet_similarities in similarities_per_source]
        # self.log_writer(f"rounded similarities: {similarities}", 'general')
        # max_indices_dict = {key: [(max(sublist), sublist.index(max(sublist))) for sublist in value] for key, value in similarities.items()}
        # self.log_writer(f"maximized similarities: {max_max_dict}", 'general')
        # Store the index of the new max value
        return max_max_dict

        
    
    def similarity_search2(self, response, dict_sources):
        # Encode the response into sentences
        task_name_to_instruct = {"example": "Given a sentence, measure the semantic and syntactic similarity to the passages",}
        query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
        passage_prefix = ""
        max_length = 32768
        
        response_sentences = response.split('\n')  # Split response into sentences
        
        embeddings_response_sentences = similarity_model.encode(response_sentences, instruction=query_prefix, max_length=max_length)
        embeddings_response_sentences = F.normalize(embeddings_response_sentences, p=2, dim=1)
        
        similarities = {}
        # Iterate through each source in the query result
        for source in dict_sources:
            snippets = dict_sources[source]['snippets']
            embeddings_passages = similarity_model.encode(snippets, instruction=passage_prefix, max_length=max_length)  # Encode snippets
            embeddings_passages = F.normalize(embeddings_passages, p=2, dim=1)
            # Calculate similarities for each response sentence with the snippets
            similarities_per_source = (embeddings_response_sentences @ embeddings_passages.T).tolist()
            similarities[source] = [[round(similarity, 2) for similarity in snippet_similarities] for snippet_similarities in similarities_per_source]
            
        max_indices_dict = {key: [(max(sublist), sublist.index(max(sublist))) for sublist in value] for key, value in similarities.items()}
        max_source_dict = {}

        # Iterate over the indices
        for i in range(len(next(iter(max_indices_dict.values())))):  # Use len of first list for range
            max_value = float('-inf')  # Initialize the maximum value
            max_source = None  # Initialize the source for max value
            max_index = None  # Initialize the index for max value
            
            # Find the maximum value at index i and corresponding source
            for source, values in max_indices_dict.items():
                if i < len(values):  # Ensure index is within the range of the list
                    if values[i][0] > max_value:  # Found a new maximum
                        max_value = values[i][0]
                        max_source = source  # Store the source of the new max value
                        max_index = values[i][1]  # Store the index of the new max value

            # Assign the max source, value, and index to the new dictionary
            if max_source is not None:
                max_source_dict[i] = {'source': max_source, 'value': max_value, 'index': max_index}

        return max_source_dict
    
    
    def similarity_search(self, response, dict_sources):
        # Encode the response into sentences
        response_sentences = response.split('\n')  # Split response into sentences
        embeddings_response_sentences = similarity_model.encode(response_sentences)

        similarities = {}
        
        # Iterate through each source in the query result
        for source in dict_sources:
            snippets = dict_sources[source]['snippets']
            embeddings_passages = similarity_model.encode(snippets)  # Encode snippets

            # Calculate similarities for each response sentence with the snippets
            similarities_per_source = similarity_model.similarity(embeddings_passages, embeddings_response_sentences)
            similarities[source] = [[round(similarity.item(), 2) for similarity in snippet_similarities] for snippet_similarities in similarities_per_source]
        return similarities
    
    def get_page_metadata(self,url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get the Open Graph or Twitter Card title, or fall back to the <title> tag
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
        page_title = soup.title.string if soup.title else None

        title = og_title['content'] if og_title else (twitter_title['content'] if twitter_title else page_title)

        # Get the Open Graph or Twitter Card description, or fall back to the meta description
        og_description = soup.find('meta', attrs={'property': 'og:description'})
        twitter_description = soup.find('meta', attrs={'name': 'twitter:description'})
        meta_description = soup.find('meta', attrs={'name': 'description'})

        description = og_description['content'] if og_description else (twitter_description['content'] if twitter_description else (meta_description['content'] if meta_description else None))

        # Get the Open Graph or Twitter Card image
        og_image = soup.find('meta', attrs={'property': 'og:image'})
        twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})

        thumbnail_url = og_image['content'] if og_image else (twitter_image['content'] if twitter_image else None)

        return {
            'title': title,
            'description': description,
            'thumbnail_url': thumbnail_url
        }
        
    def find_best_similarity(self, dict_sources, similarity_search, sentence_idx):
        res_pas_similarities_for_idx_sentence = [[similarity_search[source][i][sentence_idx] for i in range(len(similarity_search[source]))] for source in similarity_search]
        res_pas_similarities_for_idx_sentence = np.array(res_pas_similarities_for_idx_sentence)
        # Get the indices of the maximum value
        max_indices = np.unravel_index(np.argmax(res_pas_similarities_for_idx_sentence), res_pas_similarities_for_idx_sentence.shape)
        max_similarity_idx_source = max_indices[0]  # Row index of the max similarity
        max_similarity_sub_idx_snippet = max_indices[1]
        
        max_source = list(similarity_search.keys())[max_similarity_idx_source]
        snippet_best_match = dict_sources[max_source]['snippets'][max_similarity_sub_idx_snippet]
        similarity_score_best_match = res_pas_similarities_for_idx_sentence[max_similarity_idx_source][max_similarity_sub_idx_snippet]
    
        return max_similarity_idx_source, max_similarity_sub_idx_snippet, snippet_best_match, similarity_score_best_match, max_source

    
    def sentence_wise_similarity_formatter(self, response, dict_sources):
        
        similarity_search = self.similarity_search3(response, dict_sources)
        # self.log_writer(f"Similarity Search: {similarity_search}\n\n", 'general')
        response_sentences = response.split('\n')
        formatted_response = ""

        for sentence_idx, sentence in enumerate(response_sentences):
            # Get the similarities for the current sentence
            # if sentence:
                # max_similarity_idx_source, max_similarity_sub_idx_snippet, snippet_best_match, similarity_score_best_match, max_source = self.find_best_similarity(dict_sources, similarity_search, sentence_idx)
                similarity_score_best_match = similarity_search[sentence_idx]['value']
                max_source = similarity_search[sentence_idx]['source']
                # snippet_best_match_index = similarity_search[sentence_idx]['index']
                snippet_best_match = similarity_search[sentence_idx]['snippet']
                # snippet_best_match = dict_sources[max_source]['snippets'][snippet_best_match_index]
                
                # self.log_writer(f"max score: {similarity_score_best_match} -> sentence: {sentence}\n\nSnippet: {snippet_best_match}",'general')
                sentence = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', sentence)
                if similarity_score_best_match > 0.4:
                    formatted_response = self.format_response(sentence = sentence, formatted_response=formatted_response, format_type = 'format_html', dict_sources = dict_sources, max_source = max_source)
                else:
                    formatted_response += f"<p>{sentence}</p>"
                    
        return formatted_response
    
    
    @utils.enable_chat_history_pure
    def main(self):
        """
        Main function to run the chatbot.
        Handles the display of the chat, the user input, the response from the LLM and the image retrieval.
        """
        
        user = st.chat_input(placeholder="Ask me anything!")
        if user:
            utils.display_msg(user, 'user')
            with st.chat_message("assistant"):

                # Log the user input
                # self.log_writer(f"User New Input: {user}\n\n", 'general')
                # Generate the response
                response, query_result, dict_sources = self.chatLlama3(user)
                
                if not query_result:
                    # st.write(response)
                    # st.markdown(f"<ol>{response}</ol>", unsafe_allow_html=True)
                    st.markdown(response, unsafe_allow_html=True)
                if query_result:
                    response = self.format_response(response, format_type='tool_tip', dict_sources=dict_sources)
                    # self.log_writer(f"Response after tooltip: {response}", 'general')
                    # response = self.sentence_wise_similarity_formatter(response, dict_sources)
                    # st.markdown(f"<ol>{response}</ol>", unsafe_allow_html=True)
                    st.markdown(response, unsafe_allow_html=True)
                    

                # Add the response to the session state dictionary
                st.session_state.messages.append({"role": "assistant",
                                                "content": response,
                                                'items_retrieved': query_result,})
generation_model = 'groq'
embedding_model = 'similarity_base'

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "LLMNET" not in st.session_state:
    LLMNET = LLMNET(generation_model, embedding_model)
    st.session_state.LLMNET = LLMNET

llm = st.session_state.LLMNET.setup_model(generation_model)
similarity_model = st.session_state.LLMNET.setup_model(model_type = embedding_model)
st.session_state.LLMNET.main()