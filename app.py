from BizBotz import *
from BizBotz.Models import *
import uuid
from  flask import url_for
@app.route('/')
def index():
    return render_template('create_dataset.html')

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
            new_dataset = Dataset(dataset_name=dataset_name, dataset_description=dataset_description, dataset_purpose=dataset_purpose)
            # Add new_dataset to the database session
            db.session.add(new_dataset)
            # Commit changes to the database
            db.session.commit()
            
            flash('Dataset submitted successfully.', 'success')
        except Exception as e:
            # Rollback changes if an error occurs
            db.session.rollback()
            flash('Error submitting dataset. Please try again.', 'danger')
            print(e)
        return {"code":200}
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
