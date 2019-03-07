from base import Base
import sqlalchemy as db


class Atoms(Base):
    """
    atoms table that maps to nanoparticles through structure_id
    """
    __tablename__ = 'atoms'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    index = db.Column('index', db.Integer, nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    z = db.Column(db.Float, nullable=False)
    structure_id = db.Column(db.String(36),
                             db.ForeignKey('nanoparticles.id'))
    nanoparticle = db.orm.relationship('Nanoparticles', backref='atoms')

    def __init__(self, index, x, y, z, nanoparticle):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.nanoparticle = nanoparticle
