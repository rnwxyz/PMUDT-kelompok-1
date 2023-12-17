from app import db
from model.userModel import User
import bcrypt

class UserService:
    def createUser(name, password, email):
        # hash password
        hash_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        #  create user
        user = User(name=name, password=hash_password, email=email)
        db.session.add(user)
        db.session.commit()
        return

    def getUser(email):
        # get user by email
        user = User.query.filter_by(email=email).first()
        if user is None:
            return None
        return user
    
    def login(email, password):
        # get user by email
        user = User.query.filter_by(email=email).first()
        if user is None:
            return None
        
        # check password
        if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            return user
        return None

