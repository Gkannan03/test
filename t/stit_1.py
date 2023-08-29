import streamlit as st
from PIL import Image
import json
import numpy as np
import os
import csv
import datetime
import streamlit_authenticator as stauth
import json
from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel, InputOutputTextPair, ChatMessage
from vertexai.preview.language_models import TextEmbeddingModel
import pinecone
import time
import tiktoken

# Get the base directory of your project
base_dir = os.path.dirname(os.path.abspath(__file__))


# print('BASE_DIR', base_dir)
# # Construct relative paths
# pinecone_config_path = os.path.join(base_dir, 'configs', 'pinecone_config.json')
# google_credentials_path = os.path.join(base_dir, 'trial-395311-03cfd0ddfa15.json')

# print('PINECONE_CONFIG',pinecone_config_path)
# print('GOOGLE_CONFIG',google_credentials_path)
# # Load pinecone config
# with open(pinecone_config_path) as f:
#     pinecone_config = json.load(f)

# Set Google Cloud credentials
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

# ... rest of your code ...

image_1 =Image.open("C:\\Users\\kg3249915\\Downloads\\Logo-removebg.png")
image_array = np.array(image_1)

# VertexAI
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:\\Users\\kg3249915\\Downloads\\trial-395311-03cfd0ddfa15.json'

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")

st.set_page_config(page_title= 'TaskGPT powered by Google Palm', layout="wide")

# Logging details function
def main():
    def write_login_to_csv(operation, username):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        with open("C:\\Users\\kg3249915\\KG\\testfile\\login_details.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([ username, operation, timestamp])

    def logout():
        st.session_state.clear()
        authenticator.logout('Logout', 'main')
        write_login_to_csv(operation = "Log out", username=username)

    # Calculate the token
    def count_tokens(text):
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        return len(tokens)

    # Retriving the top text results
    def match_response(query_response, company_name, question):
        file_name_received = f"C:\\Users\\kg3249915\\testing\\testenv\\{company_name}.txt"
        with open(file_name_received, encoding="utf8") as f:
            file_text_dict = json.load(f)
        files_string = ""
        pinecone_query_response = []
        is_below_threshold = False
        for i, result in enumerate(query_response.matches):
            # Retrieve text associated with the current result from the dictionary "file_text_dict"
            file_text = file_text_dict.get(result.id)

            # Add the "text_chunk" key to the current "result" object with the corresponding text value
            result.text_chunk = file_text

            # Append the modified result object to pinecone_query_response list
            pinecone_query_response.append(result.to_dict())
            # Check if the result score is less than "COSINE_SIM_THRESHOLD" AND if the index "i" is greater than 0
            if result.score < 0.7 and i > 0:
                print(f"[get_answer_from_files] score {result.score} is below threshold {0.7} and i is {i}, breaking")
                # Exit the for loop early because all subsequent results will also have scores below threshold
                is_below_threshold = True
                break

            # If the result score is greater than or equal to "COSINE_SIM_THRESHOLD"
            if result.score >= 0.7:
                # Add the filename and text to the "files_string" with specific formatting
                files_string += f"###\n\"{result.metadata['filename']}\"\n{file_text}\n"
        print(files_string)
        write_login_to_csv(operation = {"file_string":files_string}, username=username)
        EXTRACTED_PORTIONS_OF_THE_ARTICLES = files_string
        # If "files_string" is empty and none of the dictionaries in result passed the threshold condition, log a message saying so using "is_below_threshold" flag
        if not files_string and is_below_threshold:
            print("[get_answer_from_files] None of the dictionaries in the query result passed the condition\n")
            prompt = f"We couldn't find a match for the user's question in our knowledge base articles. If the question is relevant considering the chat history, please provide an appropriate answer. Otherwise, respond with, \"I'm sorry, I don't have the information you're looking for at the moment. Let me learn more and get back to you, or feel free to ask me another question."
            return prompt, files_string
        else:
            prompt = f"\nYou are an AI assistant tasked with answering questions related to {company_name}.\n\nUse these guidelines as an AI assistant to offer a conversational response:\n1. Carefully study and understand the articles' excerpted portions.\n2. Use the data from the chat history and the information from the extracted article parts to properly address the user's inquiries.\n3. When providing a response, it is important to exclude any mention of the original source of information, such as mentioning \"Based on article\" or using phrases like \"according to the article.\", \"Based on the extracted article portions\" Additionally, the response should not contain phrases or words like \"help_answer:\" or \"Answer:\".\n4. Avoid making up an answer or assuming anything.\n5. Whenever the user asks for help or expresses help-related intents(Help, help, help?, need help etc..), respond with the help_answer provided below.\n6. If you don't know the answer, say \"I'm sorry, I don't have the information you're looking for at the moment. Let me learn more and get back to you, or feel free to ask me another question.\"\n---\n {EXTRACTED_PORTIONS_OF_THE_ARTICLES}\n---\nUser Question: {question}\n---\nIf the question is out of scope, not related to {company_name} services:\n   Answer: <\"I'm sorry, I don't have the information you're looking for at the moment. Let me learn more and get back to you, or feel free to ask me another question.\">\nElse:\n   Answer: <answer>\n"
            return prompt, files_string

    # Retriving the top results from pinecone
    def match(question, company_name):
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:\\Users\\kg3249915\\Downloads\\trial-395311-03cfd0ddfa15.json'
        # embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        query_embedding = embedding_model.get_embeddings([question])    
        search_query_embedding = [embed_value.values for embed_value in query_embedding]
        with open ("C:\\Users\\kg3249915\\testing\\testenv\\pinecone_config.json") as f:
        # with open (pinecone_config_path) as f:
            file = json.load(f)
        pinecone_config = file['PINECONE_CONFIG']
        pinecone.init(api_key=pinecone_config['API_KEY'], environment= pinecone_config['ENV'])
        index_name = pinecone_config['INDEX']
        pinecone_index = pinecone.Index(index_name)
        pinecone_index_key = pinecone_config[f"INDEX_KEY_{company_name.upper()}"]
        write_login_to_csv(operation = {"Pinecone_index":pinecone_index_key}, username=username)
        query_response = pinecone_index.query(namespace=pinecone_index_key,top_k= 5,include_values=False,include_metadata=True,vector= search_query_embedding)
        write_login_to_csv(operation = {"Pinecone_response":query_response}, username=username)
        prompt, files_string = match_response(query_response, company_name, question)
        return prompt, files_string

    # completion process
    def chat(query, company_name, model, total_token_count, message_history=None):
        print('called')
        write_login_to_csv(operation = "Question: "+query, username=username)
        if model == "Chat-bison-001":
            context_string, matching_chunks = match(question= query, company_name= company_name)
            chat_model = ChatModel.from_pretrained("chat-bison@001")
            print("CONTEXT", context_string)
            # message_history = [ChatMessage(content="Hi", author="user"), ChatMessage(content="Hello, How Can I help you", author="AI")]
            chat = chat_model.start_chat(context = context_string, message_history= message_history)
            print("QUERY",query)
            # chat = chat_model.start_chat(message_history= message_history)
            response = chat.send_message(query, temperature=0.1, max_output_tokens= 1024,)
            if len(st.session_state[f"{company_name}chat_history"]) <1:
                message_history.append(ChatMessage(content=query, author= "user"))
                message_history.append(ChatMessage(content=response.text, author= "bot"))
            
            for message_content in message_history:
                total_token_count += count_tokens(message_content.content)
            if total_token_count>2000:
                removed_token =0
                for removed_content in message_history[:4]:
                    removed_token+= count_tokens(removed_content.content)
                    print('Removed_token',removed_token)
                message_history=message_history[4:]
                total_token_count-=removed_token
            print('TOTAL_TOKEN', total_token_count)
                # print(len(message_history))
                # for m in message_history:
                #     if m.content == prompt:
                #         m.content = query
            print('===================================')
            print('message_history', message_history)
            print('===================================')
            print(response.__dict__)
            write_login_to_csv(operation = {"Answer: ":response.__dict__}, username=username)
            return (response.text)
        

        elif model == "Text-bison-001":
            prompt, matching_chunks = match(question= query, company_name= company_name)
            mod = TextGenerationModel.from_pretrained("text-bison@001")
            print('Prompt',prompt)
            text_prompt= " CURRENT_QUESTION: "+prompt + "(CHAT_HISTORY : "
            if message_history is not None:
                for j in message_history:
                    text_prompt+=j
                    text_prompt+=','
                text_prompt+= ")"
            print(text_prompt)
            response = mod.predict(
            text_prompt,
            temperature=0.1,
            max_output_tokens=1024, top_p = 0.95, top_k = 40)

            message_history.append('Question: '+query)
            message_history.append('Answer: '+ response.text)
            
            for message_content in message_history:
                print('MESSAGE_CONTENT',message_content)
                total_token_count += count_tokens(message_content)
            
            if total_token_count>5000:
                removed_token =0
                for removed_content in message_history[:4]:
                    removed_token+= count_tokens(removed_content)
                    print('Removed_token',removed_token)
                message_history=message_history[4:]
                total_token_count-=removed_token

            print('TOTAL_TOKEN', total_token_count)

            print("message", message_history)


            print(response.__dict__)
            write_login_to_csv(operation = {"Answer: ":response.__dict__}, username=username)
            return(response.text)
        
    # Session state initialisation and reading previous questions and answers
    def bot(username):
        st.button('Logout', on_click=logout)
        # test=authenticator.logout('Logout', 'main')
        # st.set_page_config(page_title= 'TaskGPT powered by Google Palm', layout="wide")
        image = Image.open('C:\\Users\\kg3249915\\Downloads\\image.png')
        st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        #Creating the chatbot interface
        st.title("TaskGPT powered by Google PaLM")
        st.subheader("Proof of Concept")

        # Session state for company
        if "selected_FAQ" not in st.session_state:
            st.session_state.selected_FAQ = ""        

        # Session state for model option
        if "selected_option" not in st.session_state:
            st.session_state.selected_option = ""

        col1, col2 = st.columns([1, 5])
        with col1:
            company = st.selectbox("Select an option", ["shopify", "veho", "etsyseller"], key="company_FAQ")

        with col2:
            st.write("")  # To align the selectbox to the right  



        col1, col2 = st.columns([1, 5])
        with col1:
            option = st.selectbox("Select an option", [ "Text-bison-001", "Chat-bison-001"], key="selectbox")

            if company != st.session_state.selected_FAQ or option != st.session_state.selected_option:
                st.session_state.selected_option = option
                st.session_state.selected_FAQ = company
                write_login_to_csv(operation = {"Selected ": company + " " + option}, username=username)
        with col2:
            st.write("")  # To align the selectbox to the right    


        company_name = company

        print('COMPANY',company_name)

        if company == 'shopify':
            company_name = 'shopify'
        elif company =='veho':
            company_name ='veho'
        
        ## This session state is for chat message history saving 

        if f"{company_name}chat_history" not in st.session_state:
            st.session_state[f"{company_name}chat_history"] = []

        model = ''

        # Session state for appending the previous questions and aswers for displaying using streamlit
        if f"{company}{option}textmessages" not in st.session_state:
            st.session_state[f"{company}{option}textmessages"] = []


        # Session state for counting the chat_history limit
        if f"{company}{option}count" not in st.session_state:
            st.session_state[f"{company}{option}count"] = 0


        # Display chat messages from history on app rerun
        for message in st.session_state[f"{company}{option}textmessages"]:
            if message['role'] == "user":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"],avatar=image_array):
                    st.markdown(message["content"])
         

        model= option
        # React to user input
        if prompt := st.chat_input("What is up?"):
            with st.spinner("Generating response..."):  # Show a loading spinner while generating the response
                full_response = ""
                query = prompt
                for response in chat(query, company_name, model,st.session_state[f"{company}{option}count"], st.session_state[f"{company_name}chat_history"]):
                    full_response += response

            st.session_state[f"{company}{option}textmessages"].append({"role": "user", "content": prompt})
            st.session_state[f"{company}{option}textmessages"].append({"role": "assistant", "content": full_response})
            
            
            with st.chat_message("user"):
                st.markdown(prompt)
                print("PR", prompt)
            with st.chat_message("assistant",avatar=image_array):
                print("Assi", full_response)
                st.markdown(full_response)

             # Hiding "Made with Streamlit Footer and Mainmenu option"
            hide_streamlit_style = """
                        <style>
                        MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
    names = ['kannan G','Aravind']
    usernames = ['kannan.g@taskus.com','aravind.ramesh@taskus.com']
    passwords = ['Taskus123','TaskUs@123']
    hashed_passwords = stauth.Hasher(passwords).generate()
    authenticator = stauth.Authenticate(names,usernames,hashed_passwords,'some_cookie_name','some_signature_key',cookie_expiry_days=0)
    name, authentication_status, username = authenticator.login('Login', 'main')
           
    if st.session_state["authentication_status"]:
        if "user_name" not in st.session_state:
            st.session_state.user_name= username
            write_login_to_csv(operation = "Logged in", username=username)
        bot(username)

    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    main()  