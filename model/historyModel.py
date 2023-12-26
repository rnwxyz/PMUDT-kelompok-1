from datetime import datetime
import pytz
from database import db

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    data = db.Column(db.Integer, nullable=False)
    aspek_0 = db.Column(db.String(100), nullable=False)
    aspek_1 = db.Column(db.String(100), nullable=False)
    aspek_2 = db.Column(db.String(100), nullable=False)
    aspek_3 = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(pytz.timezone('Asia/Makassar')), nullable=False)

    def __repr__(self):
        return '<History %r>' % self.id
