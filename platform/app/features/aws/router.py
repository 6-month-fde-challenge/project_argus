"""/api/v1/aws - the AWS stream deployment, driven through the repository's GitHub Actions (no AWS key here).

GET  /aws          GitHub connection, the repository variables the workflows read, recent deploy / destroy / ci runs,
                   whether `python deploy/ops.py up` has run (AWS_ROLE_ARN), the serving mode, the AWS stream's health
POST /aws/sync     .env -> repository variables (SCORING_HOST, ALLOWED_CIDRS, ALERT_EMAIL, TF_VARS) + secret SCORING_TOKEN
POST /aws/deploy   runs deploy.yml on main (tests -> image -> terraform apply). Refused while the Fraud Model API serves
                   a demo setup, unless allow_demo (passed to that one run only)
POST /aws/destroy  runs destroy.yml (only what this project created), with the typed confirmation <project>-<env>
The one-time bootstrap (state bucket, OIDC role) needs AWS credentials, so it stays on your PC: python deploy/ops.py up.
By hand: python deploy/ops.py deploy / status / down (or the Actions tab on GitHub).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...core.container import Container, get_container
from ...core.governance import Approval
from .service import AwsService

router = APIRouter(prefix="/aws", tags=["aws"])


class AwsAction(BaseModel):
    approval: Approval | None = None
    allow_demo: bool = False
    confirm: str = ""


def service(c: Container = Depends(get_container)) -> AwsService:
    return AwsService(c)


@router.get("")
async def status(svc: AwsService = Depends(service)):
    return await svc.status()


@router.post("/sync")
async def sync(req: AwsAction, svc: AwsService = Depends(service)):
    return await svc.sync(req.approval)


@router.post("/deploy")
async def deploy(req: AwsAction, svc: AwsService = Depends(service)):
    return await svc.deploy(req.approval, req.allow_demo)


@router.post("/destroy")
async def destroy(req: AwsAction, svc: AwsService = Depends(service)):
    return await svc.destroy(req.approval, req.confirm)
