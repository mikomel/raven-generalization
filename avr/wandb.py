import os
import re
from typing import Dict, List, Optional, Callable

import wandb
from tqdm import tqdm
from wandb.apis.public import Run

from avr.wandbapi.run import RunPredicate


class WandbClient:
    def __init__(self, project_name: str):
        self.api = wandb.Api()
        self.project_name = project_name

    def get_run_by_name(self, run_name: str) -> Run:
        runs = self.api.runs(self.project_name, filters={"display_name": run_name})
        if len(runs) == 1:
            return runs[0]
        else:
            raise ValueError(
                f"Query for runs with display_name='{run_name}' returned {len(runs)} results."
            )

    def get_run_names(self, filters: Dict) -> List[str]:
        runs = self.api.runs(self.project_name, filters=filters)
        return [run.name for run in runs]

    def download_checkpoint_by_artifact_name(self, artifact_name: str) -> str:
        artifact = self.api.artifact(
            f"{self.project_name}/{artifact_name}", type="model"
        )
        return self.download_ckeckpoint(artifact)

    def get_artifact_by_run_name(
        self,
        run_name: str,
        artifact_version: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> wandb.Artifact:
        if artifact_version is not None and alias is not None:
            raise ValueError(f"Provide only artifact_version or alias, not both.")
        run = self.get_run_by_name(run_name)
        artifacts = run.logged_artifacts()
        model_artifacts = [a for a in artifacts if a.type == "model"]
        if model_artifacts:
            if artifact_version:
                artifact = None
                for a in model_artifacts:
                    if a.name.endswith(artifact_version):
                        print(
                            f"Run {run.name} has {len(model_artifacts)} model artifacts. Using provided version: {artifact_version}."
                        )
                        artifact = a
                        break
                if not artifact:
                    raise ValueError(
                        f"Run {run.name} has no artifact with version {artifact_version}"
                    )
            elif alias:
                artifact = None
                for a in model_artifacts:
                    if alias in a.aliases:
                        print(
                            f"Run {run.name} has {len(model_artifacts)} model artifacts. Using provided alias: {alias}."
                        )
                        artifact = a
                        break
                if not artifact:
                    raise ValueError(
                        f"Run {run.name} has no artifact with version {artifact_version}"
                    )
            else:
                print(
                    f"Run {run.name} has {len(model_artifacts)} model artifacts. Using the last one."
                )
                artifact = model_artifacts[-1]
            return artifact
        else:
            raise ValueError(f"Run {run.name} has no artifacts with type model.")

    def download_checkpoint_by_run_name(
        self,
        run_name: str,
        artifact_version: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        artifact = self.get_artifact_by_run_name(run_name, artifact_version, alias)
        return self.download_ckeckpoint(artifact)

    def get_local_checkpoint_by_run_name(self, run_name: str, log_dir: str) -> str:
        run_id = self.get_run_by_name(run_name).id
        dir_name = f"{log_dir}/{self.project_name}/{run_id}/checkpoints/"
        files = os.listdir(dir_name)
        pattern = r"epoch=(\d+)-step=(\d+).ckpt"
        max_epoch = -1
        checkpoint = None
        for f in files:
            match = re.match(pattern, f)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    checkpoint = f
        return os.path.join(dir_name, checkpoint) if checkpoint else None

    def delete_model_artifacts(
        self,
        filters: Dict = dict(),
        run_predicate: RunPredicate = RunPredicate.always_true(),
        do_preserve_fn: Optional[Callable[[wandb.Artifact], bool]] = None,
        dryrun: bool = False,
    ) -> None:
        num_artifacts_deleted = 0
        for run in tqdm(self.api.runs(self.project_name, filters=filters)):
            if run_predicate(run):
                for artifact in run.logged_artifacts():
                    # The wandb-history artifact (a parquet file) stores run's history metrics. By design, wandb doesn't
                    # allow to delete this artifact.
                    if artifact.type == "wandb-history":
                        continue
                    if do_preserve_fn is not None and do_preserve_fn(artifact):
                        print(
                            f"Preserving artifact {artifact.name} with aliases {artifact.aliases}"
                        )
                    else:
                        num_artifacts_deleted += 1
                        if not dryrun:
                            try:
                                artifact.delete(delete_aliases=True)
                            except wandb.errors.CommError as e:
                                print(
                                    f"Failed to delete artifact: run={run.name}, artifact={artifact.name}, message={e.message}"
                                )
        print(f"Deleted artifacts: {num_artifacts_deleted}")

    @staticmethod
    def download_ckeckpoint(artifact: wandb.Artifact) -> str:
        checkpoint_dir = artifact.download()
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
        print(
            f"Success: downloaded wandb artifact={artifact.name} to path={checkpoint_path}"
        )
        return checkpoint_path

    @staticmethod
    def _safe_dict_get(d: Dict, key: str) -> Optional:
        value = d
        for node in key.split("."):
            if value:
                value = value.get(node)
        return value
