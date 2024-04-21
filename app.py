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
import requests
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
    print("====================================================\n",resp)
    training_resp=client.fine_tuning.jobs.create(
    training_file=resp.id, 
    model="gpt-3.5-turbo"
    )
    print("====================================================\n",training_resp)
    while True:
        update_status=client.fine_tuning.jobs.retrieve(training_resp.id)
        time.sleep(10)
        print("====================================================\n",update_status)
        if update_status.status=="succeeded":
            break
    return str(update_status.fine_tuned_model)
def finetune_llama_2(filepath,data):
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
        max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
        "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
        "unsloth/gemma-2b-bnb-4bit",
        "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
        "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    )
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Context:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instructions"]
        inputs       = examples["question"]
        contexts       = examples["data"]
        outputs      = examples["output"]
        texts = []
        for instruction, input,context, output in zip(instructions, inputs,contexts, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input,context, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts,}
    def gen():
      for i in data:
        yield i
    dataset=Dataset.from_generator(gen)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 5,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    trainer_stats = trainer.train()
    model.save_pretrained(filepath)

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
        name=""
        if option=="OpenAI":
            name=fine_tune_openai(dataset.dataset_collection_name.replace("Chroma","OpenAI_data.jsonl"),data)
        else:
            json_data={"data":json.dumps(data),"filepath":dataset.dataset_name+str(uuid.uuid4())}
            req=requests.post("https://c423-35-247-108-133.ngrok-free.app/finetune",data=json_data)
            print(req.text)
    return {"code":200,"name":name,"redirect_url":url_for("view_dataset",item_uuid=item_uuid)}

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
        print(output)
        file.close()
    else:
        embedding = OpenAIEmbeddings()
        vectordb= Chroma(persist_directory=dataset.dataset_collection_name,embedding_function=embedding)
        retriever = vectordb.as_retriever()
        agent=gpt_AgentBuilder()
        resp=agent.invoke(
        {"input": "The Main objective of this bot is to generate data on the documents of a company called "+dataset.dataset_name+". It is described as follows "+dataset.dataset_description+".\
            Create a list of 10 to 11 responses that simulate the following behaviour from an user point of view without referening the bot. '"+dataset.dataset_purpose+"'."},
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
            embedding = OpenAIEmbeddings()

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
