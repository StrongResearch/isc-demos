from pathlib import Path
import os
import torch

def atomic_torch_save(obj, f: str | Path, timer=None, **kwargs):
    f = str(f)
    temp_f = f + ".temp"
    torch.save(obj, temp_f, **kwargs)
    if timer is not None:
        timer.report(f'saving temp checkpoint')
    os.replace(temp_f, f)
    if timer is not None:
        timer.report(f'replacing temp checkpoint with checkpoint')
        return timer
    else:
        return
    
# ## ENABLING ACTIVE PROGRESS TRACKING

# from sqlalchemy.orm import Session, sessionmaker
# from sqlmodel import SQLModel, create_engine
# from strenum import StrEnum

# class Experiment(SQLModel, table=True):
#     __tablename__ = "experiments"

#     id: int = Field(primary_key=True, index=True)
#     org_id: str = Field(unique=False)
#     user_id: int = Field(foreign_key="users.id", unique=False)
#     user: User = Relationship(back_populates="experiments")
#     runtime: int | None = Field(unique=False, nullable=True)
#     name: str = Field(unique=False)
#     output_path_used: str | None = Field(unique=False, nullable=True)
#     output_path: str = Field(unique=False)
#     ips: dict[str, int] | None = Field(sa_column=Column(JSON, nullable=True))
#     status: str = Field(nullable=False)
#     gpu_type: str = Field(unique=False)
#     nnodes: int = Field(unique=False)
#     venv_path: str = Field(unique=False)
#     command: str = Field(unique=False)
#     work_dir: str = Field(unique=False)
#     framework: str | None = Field(unique=False, nullable=True)
#     created_at: datetime = Field(default_factory=timestamp_factory, nullable=False)
#     usage_wall_time: int = Field(default=0, nullable=False)
#     last_ran_at: datetime | None = Field(default=None, nullable=True)
#     # started_at: datetime | None = Field(nullable=True)
#     # finished_at: datetime | None = Field(nullable=True)
#     progress: int = Field(default=0, nullable = False) # perhaps


# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/cluster_server"

# def get_db(database_url: str | None = None) -> Session:
#     if database_url is None:
#         database_url = SQLALCHEMY_DATABASE_URL
#     """Returns a session to the database"""
#     engine = create_engine(database_url, isolation_level="AUTOCOMMIT")
#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     return SessionLocal()

# # SessionLocal: Callable[[], Session] = get_db
# SessionLocal = get_db

# class AtomicTorchSave:
#     def __init__(self):
#         self.progress = 0
#         self.experiment_id = os.environ["STRONG_EXPERIMENT_ID"]

#     def commit_progress(self):
#         db = SessionLocal()
#         db_experiment = db.query(Experiment).filter(Experiment.id == self.experiment_id).first()
#         assert db_experiment is not None
#         db_experiment.progress = self.progress
#         db.commit()
#         db.refresh(db_experiment)
#         return db_experiment

#     def save(self, obj, f: str | Path, timer=None, **kwargs):
#         f = str(f)
#         temp_f = f + ".temp"
#         torch.save(obj, temp_f, **kwargs)
#         if timer is not None:
#             timer.report(f'saving temp checkpoint')
#         os.replace(temp_f, f)
#         timer.report(f'replacing temp checkpoint with checkpoint')

#         if self.experiment_id is not None:
#             try:
#                 self.commit_progress()
#             except:
#                 print("Progress commit failed.")
#         else:
#             print("Experiment id not set.")
#         timer.report(f'committing progress to database')

#         if timer is not None:
#             return timer
#         else:
#             return
        
    

