from datetime import datetime
import pytz
from app import db

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    history = db.Column(db.String(400), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(pytz.timezone('Asia/Makassar')), nullable=False)

    def __repr__(self):
        return '<History %r>' % self.history
