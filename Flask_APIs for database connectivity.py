from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from sqlalchemy.exc import SQLAlchemyError


app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Govardhan%4089@localhost:3306/flask_users'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Tenant(db.Model):
    __tablename__ = 'tenant'
    tenant_name = db.Column(db.String(250))
    password = db.Column(db.String(100)) 
    description = db.Column(db.Text)
    phone = db.Column(db.Integer)
    email_id = db.Column(db.String(30), primary_key=True)
    address = db.Column(db.String(300))
    nature = db.Column(db.String(20))
    credits = db.Column(db.Integer)
    validity_start = db.Column(db.String(30))
    validity_end = db.Column(db.String(30))
    status = db.Column(db.String(100))


class User(db.Model):
    __tablename__ = 'users'
    email_id = db.Column(db.String(255), db.ForeignKey('tenant.email_id'), primary_key=True)
    user_name = db.Column(db.String(255))
    password = db.Column(db.String(100))  
    description = db.Column(db.Text)
    phone = db.Column(db.Integer)
    address = db.Column(db.String(300))
    nature = db.Column(db.String(20))
    credits = db.Column(db.Integer)
    validity_start = db.Column(db.String(30))
    validity_end = db.Column(db.String(30))
    status = db.Column(db.String(100))
    tenant = db.relationship('Tenant', backref='users')

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

@app.route('/add_tenant', methods=['POST'])
def add_tenant():
    if request.method == 'POST':
        data = request.get_json()
    
        tenant = Tenant(
            tenant_name=data['tenant_name'],
            password=hash_password(data['password']),
            description=data['description'],
            phone=data['phone'],  
            email_id=data['email_id'],
            address=data['address'],
            nature=data['nature'],
            credits=data['credits'],
            validity_start=data['validity_start'],
            validity_end=data['validity_end'],
            status=data['status']
        )

   
        try:
            db.session.add(tenant)
            db.session.commit()
            return jsonify({'message': 'Tenant added successfully!'}), 201
        except SQLAlchemyError as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500


@app.route('/add_user', methods=['POST'])
def add_user():
    if request.method == 'POST':
        data = request.get_json()
        
    
        user = User(
            user_name=data['user_name'],
            password=hash_password(data['password']),
            description=data['description'],
            phone=data['phone'],
            email_id=data['email_id'],
            address=data['address'],
            nature=data['nature'],
            credits=data['credits'],
            validity_start=data['validity_start'],
            validity_end=data['validity_end'],
            status=data['status']
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            return jsonify({"message": "user added successfully!"}), 201
        except SQLAlchemyError as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500


@app.route('/update_user/<email_id>', methods=['PUT'])
def update_user(email_id):
    if request.method == 'PUT':
            data = request.get_json()
            user = User.query.filter_by(email_id=email_id).first()
            
            if not user:
                return jsonify({"error": "User not found"}), 404

            try:
                user.user_name = data.get('user_name', user.user_name)
                user.password = hash_password(data.get('password', user.password))
                user.description = data.get('description', user.description)
                user.phone = data.get('phone', user.phone)
                user.address = data.get('address', user.address)
                user.nature = data.get('nature', user.nature)
                user.credits = data.get('credits', user.credits)
                user.status = data.get('status', user.status)

                db.session.commit()
                return jsonify({"message": "User updated successfully!"}), 200

            except SQLAlchemyError as e:
                db.session.rollback()
                return jsonify({"error": str(e)}), 500


@app.route('/update_tenant/<email_id>', methods=['PUT'])
def update_tenant(email_id):
    if request.method=='PUT':
            data = request.get_json()
            tenant = Tenant.query.filter_by(email_id=email_id).first()
            
            if not tenant:
                return jsonify({"error": "tenant not found"}), 404

            try:
                tenant.tenant_name = data.get('tenant_name', tenant.tenant_name)
                tenant.password = hash_password(data.get('password',  tenant.password))
                tenant.description = data.get('description',  tenant.description)
                tenant.phone = data.get('phone',  tenant.phone)
                tenant.address = data.get('address',  tenant.address)
                tenant.nature = data.get('nature',  tenant.nature)
                tenant.credits = data.get('credits',  tenant.credits)
                tenant.status = data.get('status',  tenant.status)

                db.session.commit()
                return jsonify({"message": "tenant updated successfully!"}), 200

            except SQLAlchemyError as e:
                db.session.rollback()
                return jsonify({"error": str(e)}), 500

@app.route('/delete_user/<email_id>', methods=['DELETE'])
def delete_user(email_id):
    user = User.query.filter_by(email_id=email_id).first()

    if not user:
        return jsonify({"error": " User not found"}), 404

    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "User deleted successfully!"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/delete_tenant/<email_id>', methods=['DELETE'])
def delete_tenant(email_id):
    tenant = Tenant.query.filter_by(email_id=email_id).first()

    if not tenant:
        return jsonify({"error": "Tenant not found"}), 404

    try:
        db.session.delete(tenant)
        db.session.commit()
        return jsonify({"message": "Tenant deleted successfully!"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/get_tenant_data/<email_id>', methods=['GET'])
def get_tenant_data(email_id):
    tenant = Tenant.query.filter_by(email_id=email_id).first()

    if not tenant:
        return jsonify({"error": "No data found for the given email_id"}), 404

    tenant_data = {
        "tenant_name": tenant.tenant_name,
        "passwor": tenant.password,
        "email_id": tenant.email_id,
        "description": tenant.description,
        "phone": tenant.phone,
        "address": tenant.address,
        "nature": tenant.nature,
        "credits": tenant.credits,
        "validity_start": tenant.validity_start,
        "validity_end": tenant.validity_end,
        "status": tenant.status
    }

    return jsonify(tenant_data)


@app.route('/get_user_data/<email_id>', methods=['GET'])
def get_user_data(email_id):
    user = User.query.filter_by(email_id=email_id).first()

    if not user:
        return jsonify({"error": "No data found for the given email_id"}), 404

    user_data = {
        "user_name": user.user_name,
        "description": user.description,
        "phone": user.phone,
        "email_id": user.email_id,
        "address": user.address,
        "nature": user.nature,
        "credits": user.credits,
        "validity_start": user.validity_start,
        "validity_end": user.validity_end,
        "status": user.status
    }

    return jsonify(user_data)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)








