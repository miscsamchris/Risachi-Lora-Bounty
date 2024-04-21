from BizBotz import db,app,admin,ModelView
from sqlalchemy import types
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql.base import MSBinary
from sqlalchemy.schema import Column
import uuid
class UUID(types.TypeDecorator):
    impl = MSBinary
    def __init__(self):
        self.impl.length = 16
        types.TypeDecorator.__init__(self,length=self.impl.length)

    def process_bind_param(self,value,dialect=None):
        if value and isinstance(value,uuid.UUID):
            return value.bytes
        elif value and not isinstance(value,uuid.UUID):
            raise ValueError('value %s is not a valid uuid.UUID' % value)
        else:
            return None
    def process_result_value(self,value,dialect=None):
        if value:
            return uuid.UUID(bytes=value)
        else:
            return None
    def is_mutable(self):
        return False
app.app_context().push()

class Company(db.Model):
    __tablename__="Company"
    id=db.Column(db.Integer,primary_key=True, autoincrement=True, nullable=False)
    company_name=db.Column(db.Text, nullable=False,server_default='')
    company_description=db.Column(db.Text,server_default='')
    company_narrative=db.Column(db.Text,server_default='')
    item_uuid = db.Column('item_uuid',UUID(),default=uuid.uuid4)
    datasets = db.relationship('Dataset', backref='company', lazy=True)
    models = db.relationship('FinetunedModel', backref='company', lazy=True)

class Dataset(db.Model):
    __tablename__="Dataset"
    id=db.Column(db.Integer,primary_key=True, autoincrement=True, nullable=False)
    dataset_name=db.Column(db.Text, nullable=False,server_default='')
    dataset_description=db.Column(db.Text,server_default='')
    company_id= db.Column(db.Integer, db.ForeignKey('Company.id'))
    dataset_purpose=db.Column(db.Text,server_default='')
    dataset_status=db.Column(db.Text,server_default='Created')
    dataset_collection_name=db.Column(db.Text,server_default='')
    item_uuid = db.Column('item_uuid',UUID(),default=uuid.uuid4)


class FinetunedModel(db.Model):
    __tablename__="FinetunedModel"
    id=db.Column(db.Integer,primary_key=True, autoincrement=True, nullable=False)
    model_name=db.Column(db.Text, nullable=False,server_default='')
    model_descriptione=db.Column(db.Text,server_default='')
    company_id= db.Column(db.Integer, db.ForeignKey('Company.id'))
    model_path=db.Column(db.Text,server_default='')
    item_uuid = db.Column('item_uuid',UUID(),default=uuid.uuid4)


admin.add_view(ModelView(Company, db.session))
admin.add_view(ModelView(Dataset, db.session))
admin.add_view(ModelView(FinetunedModel, db.session))