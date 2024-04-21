from BizBotz import *
from BizBotz.Models import *
import uuid
from  flask import url_for
import traceback
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.retriever import create_retriever_tool
import tiktoken
import json
import pickle

import time
from openai import OpenAI

@app.route('/')
def index():
    return render_template('create_dataset.html')
def gpt_AgentBuilder():
    from typing import List

    from langchain_core.pydantic_v1 import BaseModel, Field


    class Dataset_builder(BaseModel):
        """Final response to the user based on the input"""

        instructions: str = Field(description="Write a prompt / instruction for the bot based on it's purpose and description. Make it long and make sure it follows the instructions.")
        data_items: List[str] = Field(
            description="A list of data that has been asked to be generated based on the details provide by the company"
        )
        
    def parse(output):
        # If no function was invoked, return to user
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        # Parse out the function call
        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])
        print("Name is ",name,"\n")
        # If the Response function was invoked, return to the user with the function inputs
        if name == "Dataset_builder":
            return AgentFinish(return_values=inputs, log=str(function_call))
        # Otherwise, return an agent action
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
            )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides information on the document in the formatted output"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI()

    llm_with_tools = llm.bind_functions([Dataset_builder])
    agent = (
        {
            "input": lambda x: x["input"],
            # Format agent scratchpad from intermediate steps
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | parse
    )

    agent_executor = AgentExecutor(tools=[], agent=agent, verbose=True)
    return agent_executor

def gpt_datasetAgent(retriever):
    class Answer(BaseModel):
        """Final response to the user based on the input"""
        Answer: str = Field(description="Asnwer to the question asked  by taking data from the provide context from the dataset.")
    def parse(output):
        # If no function was invoked, return to user
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        # Parse out the function call
        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])
        print("Name is ",name,"\n")
        # Otherwise, return an agent action
        if name == "Answer":
            return AgentFinish(return_values=inputs, log=str(function_call))
        # Otherwise, return an agent action
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
            )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides information on the document in the formatted output"),
            ("user", "{data}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI()

    llm_with_tools = llm.bind_functions([Answer])
    agent = (
        {
            "input": lambda x: x["input"],
            "data": lambda x: x["data"],
            # Format agent scratchpad from intermediate steps
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | parse
    )

    agent_executor = AgentExecutor(tools=[], agent=agent, verbose=True)
    return agent_executor

def fine_tune_openai(filepath,data):
    jsonlfile=open(filepath,"w+")
    data_len=len(data)
    for item in data:
        system_message={"role":"system","content":item["instructions"]}
        context_message={"role":"system","content":'Only Answer the questions from the given context '+item["data"]}
        user_message={"role":"user","content":item["question"]}
        assistant_message={"role":"assistant","content":item["output"]}
        row_dict={"messages":[system_message,context_message,user_message,assistant_message]}
        jsonlfile.write(json.dumps(row_dict)) 
        jsonlfile.write("\n")
    jsonlfile.close()
    client = OpenAI()
    resp=client.files.create(
    file=open(filepath, "rb"),
    purpose="fine-tune"
    )
    client.fine_tuning.jobs.create(
    training_file="file-abc123", 
    model="gpt-3.5-turbo"
    )
    print(resp)

@app.route('/save_dataset/<string:item_uuid>/', methods=['POST',"GET"])
def save_dataset(item_uuid):
    item_uuid=uuid.UUID(item_uuid)
    dataset=Dataset.query.filter_by(item_uuid=item_uuid).first()
    dataset.dataset_status="Saved"
    db.session.commit()
    return redirect(url_for("view_dataset",item_uuid=item_uuid))

@app.route('/start_finetuning/<string:item_uuid>/', methods=['POST',"GET"])
def start_finetuning(item_uuid):
    if request.method == 'POST':
        option=request.form['Model_name']
        item_uuid=uuid.UUID(item_uuid)
        dataset=Dataset.query.filter_by(item_uuid=item_uuid).first()
        file=open( dataset.dataset_collection_name.replace("Chroma","pickle.pkl"), "rb" )
        data = pickle.load( file)
        file.close()
        if option=="OpenAI":
            fine_tune_openai(dataset.dataset_collection_name.replace("Chroma","OpenAI_data.jsonl"),data)
    return {"code":200}

@app.route('/update_dataset/<string:item_uuid>/' ,methods=['POST',"GET"])
def update_dataset(item_uuid):
    if request.method == 'POST':
        item_id=int(request.form['id'])
        instructions=request.form['instructions']
        question=request.form['question']
        output=request.form['output']
        item_uuid=uuid.UUID(item_uuid)
        dataset=Dataset.query.filter_by(item_uuid=item_uuid).first()
        file=open( dataset.dataset_collection_name.replace("Chroma","pickle.pkl"), "rb" )
        data = pickle.load( file)
        file.close()
        for i in data:
            print(i["ID"],"=====",item_id,i["ID"]==item_id)
            if i["ID"]==item_id:
                i["instructions"]=instructions
                i["question"]=question
                i["output"]=output
        file=open( dataset.dataset_collection_name.replace("Chroma","pickle.pkl"), "wb" )
        pickle.dump( data, file )
        file.close()
        return redirect(url_for("view_dataset",item_uuid=item_uuid))

@app.route('/view_dataset/<string:item_uuid>/', methods=['POST',"GET"])
def view_dataset(item_uuid):
    item_uuid=uuid.UUID(item_uuid)
    dataset=Dataset.query.filter_by(item_uuid=item_uuid).first()
    if os.path.isfile(dataset.dataset_collection_name.replace("Chroma","pickle.pkl")):
        file=open( dataset.dataset_collection_name.replace("Chroma","pickle.pkl"), "rb" )
        output = pickle.load( file)
        file.close()
    else:
        embedding = tiktoken.get_encoding("cl100k_base")
        vectordb= Chroma(persist_directory=dataset.dataset_collection_name,embedding_function=embedding)
        retriever = vectordb.as_retriever()
        agent=gpt_AgentBuilder()
        resp=agent.invoke(
        {"input": "The Main objective of this bot is to generate data on the documents of a company called "+dataset.dataset_name+". It is described as follows "+dataset.dataset_description+".\
            Create a list of 15 to 20 responses that simulate the following behaviour from an user point of view without referening the bot. '"+dataset.dataset_purpose+"'."},
        return_only_outputs=True,
        )
        print(resp["data_items"])
        output=[]
        count=1
        for item in resp["data_items"]:
            docs = retriever.get_relevant_documents(item)
            data="### Context: "
            for i in docs:
                string_content=i.page_content.replace("\n"," ")
                data+=str(string_content.encode('ascii', 'ignore'))+"\n\n"
            resp_agent=gpt_datasetAgent(retriever)
            answer=resp_agent.invoke(
                {"input":item,"data":data },
                return_only_outputs=True,
            )
            if answer.get("Answer",None)!=None:
                output.append({"ID":count,"instructions":resp["instructions"],"question":item,"data":data,"output":answer['Answer']})
            else:
                output.append({"ID":count,"instructions":resp["instructions"],"question":item,"data":data,"output":answer['output']})
            if count%3==0:
                time.sleep(30)
            count+=1
        file=open( dataset.dataset_collection_name.replace("Chroma","pickle.pkl"), "wb" )
        pickle.dump( output, file )
        file.close()
    return render_template("generate_dataset.html",output=output,dataset=dataset)

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    if request.method == 'POST':
        dataset_name = request.form['dataset_name']
        dataset_description = request.form['dataset_description']
        dataset_purpose = request.form['dataset_purpose']
        files = request.files.getlist('pdf_files')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name+str(uuid.uuid4()))
        os.makedirs(file_path,exist_ok=True)
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                file.save( os.path.join(file_path,filename))
        try:
            loader = DirectoryLoader(file_path, glob="./*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            # text_splitter = SemanticChunker(embeddings=HuggingFaceEmbeddings())
            text_splitter =CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=300, chunk_overlap=0)
            # text_splitter=RecursiveCharacterTextSplitter()
            texts = text_splitter.split_documents(documents)
            persist_directory = os.path.join(file_path,"Chroma")
            new_dataset = Dataset(dataset_status="Created",dataset_name=dataset_name, dataset_description=dataset_description, dataset_purpose=dataset_purpose,dataset_collection_name=persist_directory)
            # Add new_dataset to the database session
            db.session.add(new_dataset)
            # Commit changes to the database
            db.session.commit()
            ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
            embedding = tiktoken.get_encoding("cl100k_base")

            vectordb = Chroma.from_documents(documents=texts, 
                                            embedding=embedding,
                                            persist_directory=persist_directory)
            vectordb.persist()
            flash('Dataset submitted successfully.', 'success')
            return {"code":200,"redirect_url":url_for("view_dataset",item_uuid=str(new_dataset.item_uuid))}
        except Exception as e:
            # Rollback changes if an error occurs
            db.session.rollback()
            flash('Error submitting dataset. Please try again.', 'danger')
            print(traceback.format_exc())
        return {"code":201}
    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    # file = request.files['file']
    # dataset_name=req
    # dataset_description=req
    # dataset_purpose=req
    # pdf_files=req
    # if file.filename == '':
    #     flash('No selected file')
    #     return redirect(request.url)
    # if file and file.filename!=None:
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     flash('File uploaded successfully')
    #     return redirect('/')
    # else:
    #     flash('Invalid file format. Only .txt and .csv files are allowed.')
    #     return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
