from BizBotz import db,app
from BizBotz.Models import Company
app.app_context().push()
db.create_all()
ap=Company(company_name="Polygon",company_description="Polygon (formerly Matic Network) is a protocol and framework for building and connecting Ethereum-compatible blockchain networks. It aims to address some of the scalability and usability issues facing the Ethereum network by providing tools and infrastructure for developers to create decentralized applications (dApps) with faster transactions and lower fees.")
db.session.add(ap)
db.session.commit()