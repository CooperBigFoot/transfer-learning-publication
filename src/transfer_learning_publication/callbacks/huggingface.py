"""Hugging Face integration callbacks for PyTorch Lightning."""

import logging
import os
from pathlib import Path

from huggingface_hub import upload_file
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class HFUploadCallback(Callback):
    """Callback to upload model checkpoints to Hugging Face Hub.
    
    This callback uploads the best model checkpoint to Hugging Face Hub
    after each training epoch. It runs after the ModelCheckpoint callback
    has saved the checkpoint.
    
    Args:
        repo_id: The repository ID on Hugging Face Hub (e.g., "username/repo-name")
        repo_type: Type of repository ("model", "dataset", "space"). Default: "model"
        path_in_repo: Path pattern for uploaded files. Can include {filename} placeholder.
                     Default: "checkpoints/{filename}"
        private: Whether the repository should be private. Default: False
        token: Hugging Face API token. If None, uses the token from huggingface-cli login.
    """
    
    def __init__(
        self,
        repo_id: str,
        repo_type: str = "model",
        path_in_repo: str = "checkpoints/{filename}",
        private: bool = False,
        token: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.path_in_repo = path_in_repo
        self.private = private
        self.token = token
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Upload checkpoint after training epoch ends.
        
        This method runs after the ModelCheckpoint callback has saved
        the best model checkpoint.
        """
        checkpoint_callback = trainer.checkpoint_callback
        
        if not checkpoint_callback:
            logger.debug("No checkpoint callback found, skipping upload")
            return
            
        if not checkpoint_callback.best_model_path:
            logger.debug("No best model path found yet, skipping upload")
            return
            
        checkpoint_path = Path(checkpoint_callback.best_model_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return
            
        # Format the path in repo
        filename = checkpoint_path.name
        upload_path = self.path_in_repo.format(filename=filename)
        
        try:
            logger.info(f"Uploading checkpoint to Hugging Face: {checkpoint_path}")
            
            upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=upload_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.token,
                create_pr=False,
            )
            
            logger.info(
                f"Successfully uploaded checkpoint to {self.repo_id}/{upload_path}"
            )
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to Hugging Face: {e}")
            # Don't raise the exception to avoid interrupting training
            
    def on_fit_end(self, trainer, pl_module):
        """Upload final checkpoint when training completes."""
        logger.info("Training completed, uploading final checkpoint")
        self.on_train_epoch_end(trainer, pl_module)