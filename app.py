from BizBotz import *
from BizBotz.Models import *
import uuid
from  flask import url_for
import traceback

@app.route('/')
def index():
    return render_template('create_dataset.html')


@app.route('/view_dataset/<string:item_uuid>/')
def view_dataset(item_uuid):
    item_uuid=uuid.UUID(item_uuid)
    dataset=Dataset.query.filter_by(item_uuid=item_uuid).first()
    embedding = OpenAIEmbeddings()
    vectordb= Chroma(persist_directory=dataset.dataset_collection_name,embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("What are the design goals of the whitepaper? ")
    data=[]
    for i in docs:
        print(i,"\n\n========================================\n\n")
        data.append({"Content":i.page_content})
    return {"data":data}

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
            text_splitter = SemanticChunker(embeddings=HuggingFaceEmbeddings())
            texts = text_splitter.split_documents(documents)
            persist_directory = os.path.join(file_path,"Chroma")
            new_dataset = Dataset(dataset_name=dataset_name, dataset_description=dataset_description, dataset_purpose=dataset_purpose,dataset_collection_name=persist_directory)
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
    app.run()
