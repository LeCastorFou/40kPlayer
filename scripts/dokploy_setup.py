#!/usr/bin/env python3
"""Crée (ou met à jour) l'application 40kPlayer sur une instance Dokploy, puis la déploie.

Idempotent : relancer le script ne duplique rien. Étapes :

1. projet ``--project`` (``other`` par défaut), créé s'il n'existe pas ;
2. application ``--app`` (``40kplayer``) dans ce projet, sur le même serveur que les autres
   applications de l'instance si elles en ont un ;
3. source : ce dépôt GitHub (``--repo``, branche ``--branch``) via un fournisseur GitHub de Dokploy
   qui y a accès ;
4. build ``Dockerfile``, variable ``FORTYK_ACCESS_CODE``, volume nommé monté sur ``/data`` ;
5. domaine : ``--domain`` (HTTPS Let's Encrypt), sinon un domaine traefik.me généré par Dokploy ;
6. déploiement, attente de la fin, puis ``/healthz``.

Variables d'environnement : ``DOKPLOY_URL``, ``DOKPLOY_API_KEY`` (obligatoires),
``FORTYK_ACCESS_CODE`` (facultative). Rien de secret n'est affiché.

    DOKPLOY_URL=… DOKPLOY_API_KEY=… python3 scripts/dokploy_setup.py --repo LeCastorFou/40kPlayer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PORT = 8040


class Dokploy:
    def __init__(self, url: str, key: str):
        self.key = key
        self.base = self._resolve(url.rstrip("/"))

    def _resolve(self, url: str) -> str:
        """Suit les redirections (308 vers https…) une fois, pour ne plus en avoir sur les POST."""
        req = urllib.request.Request(url + "/api/project.all", headers={"x-api-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                final = r.geturl()
        except urllib.error.HTTPError as err:
            final = err.geturl() or url + "/api/project.all"
        return final.split("/api/")[0]

    def call(self, name: str, payload: Optional[Dict[str, Any]] = None, query: Optional[Dict[str, Any]] = None, quiet: bool = False) -> Tuple[int, Any]:
        url = f"{self.base}/api/{name}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST" if payload is not None else "GET",
                                     headers={"x-api-key": self.key, "content-type": "application/json", "accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                code = r.status
        except urllib.error.HTTPError as err:
            raw, code = err.read(), err.code
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = raw.decode(errors="replace")
        if not quiet:
            print(f"  {name} -> {code}")
        return code, body


def fail(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def notice(msg: str) -> None:
    """Ligne visible dans les annotations du run (lisibles par l'API GitHub, sans les logs)."""
    print(f"::notice::{msg}" if os.environ.get("GITHUB_ACTIONS") else msg)


def short(body: Any, n: int = 300) -> str:
    return (json.dumps(body) if not isinstance(body, str) else body)[:n]


def unwrap(body: Any) -> Any:
    """Certaines versions renvoient {"json": …} ou {"result": {"data": …}}."""
    if isinstance(body, dict):
        if "json" in body and len(body) <= 2:
            return body["json"]
        if "result" in body and isinstance(body["result"], dict):
            return body["result"].get("data", body["result"])
    return body


def all_projects(api: Dokploy) -> List[Dict[str, Any]]:
    code, body = api.call("project.all", quiet=True)
    if code != 200:
        fail(f"project.all -> {code} {short(body)}")
    return unwrap(body) or []


def apps_of(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    apps = list(project.get("applications") or [])
    for env in project.get("environments") or []:
        apps += env.get("applications") or []
    return apps


def find_server_id(projects: List[Dict[str, Any]]) -> Optional[str]:
    for p in projects:
        for a in apps_of(p):
            if a.get("serverId"):
                return a["serverId"]
    return None


def ensure_project(api: Dokploy, name: str) -> Dict[str, Any]:
    for p in all_projects(api):
        if p.get("name") == name:
            print(f"projet « {name} » existant : {p.get('projectId')}")
            return p
    code, body = api.call("project.create", {"name": name, "description": "Projets divers (40kPlayer…)"})
    if code != 200:
        fail(f"project.create -> {code} {short(body)}")
    for p in all_projects(api):
        if p.get("name") == name:
            print(f"projet « {name} » créé : {p.get('projectId')}")
            return p
    fail("projet créé mais introuvable")


def environment_id(project: Dict[str, Any]) -> Optional[str]:
    envs = project.get("environments") or []
    if not envs:
        return None
    for e in envs:
        if (e.get("name") or "").lower() in ("production", "prod", "default"):
            return e.get("environmentId")
    return envs[0].get("environmentId")


def ensure_application(api: Dokploy, project: Dict[str, Any], name: str, server_id: Optional[str]) -> str:
    for a in apps_of(project):
        if a.get("name") == name:
            print(f"application « {name} » existante : {a['applicationId']}")
            return a["applicationId"]
    payload: Dict[str, Any] = {"name": name, "appName": f"fortyk-{name}".lower(), "description": "Warhammer 40k V11 — parties à deux, sauvegardées"}
    env_id = environment_id(project)
    if env_id:
        payload["environmentId"] = env_id
    else:
        payload["projectId"] = project["projectId"]
    if server_id:
        payload["serverId"] = server_id
    code, body = api.call("application.create", payload)
    if code != 200:
        fail(f"application.create -> {code} {short(body)}")
    body = unwrap(body)
    if isinstance(body, dict) and body.get("applicationId"):
        print(f"application « {name} » créée : {body['applicationId']}")
        return body["applicationId"]
    for p in all_projects(api):
        if p.get("projectId") == project.get("projectId"):
            for a in apps_of(p):
                if a.get("name") == name:
                    return a["applicationId"]
    fail("application créée mais introuvable")


def github_source(api: Dokploy, app_id: str, owner: str, repo: str, branch: str) -> None:
    code, providers = api.call("github.githubProviders")
    providers = unwrap(providers) or []
    if code != 200 or not providers:
        fail("aucun fournisseur GitHub configuré sur cette instance Dokploy (Settings → Git → GitHub)")
    for prov in providers:
        gid = prov.get("githubId")
        name = (prov.get("gitProvider") or {}).get("name") or gid
        code, repos = api.call("github.getGithubRepositories", query={"githubId": gid}, quiet=True)
        repos = unwrap(repos) or []
        full = {f"{(r.get('owner') or {}).get('login', '')}/{r.get('name', '')}".lower(): r for r in repos if isinstance(r, dict)}
        print(f"  fournisseur GitHub « {name} » : {len(full)} dépôt(s) visibles")
        if f"{owner}/{repo}".lower() in full:
            r = full[f"{owner}/{repo}".lower()]
            payload = {"applicationId": app_id, "githubId": gid, "owner": r["owner"]["login"], "repository": r["name"], "branch": branch,
                       "buildPath": "/", "triggerType": "push", "watchPaths": None, "enableSubmodules": False}
            code, body = api.call("application.saveGithubProvider", payload)
            if code != 200:
                # anciennes versions : pas de triggerType / enableSubmodules
                for k in ("triggerType", "enableSubmodules", "watchPaths"):
                    payload.pop(k, None)
                code, body = api.call("application.saveGithubProvider", payload)
            if code != 200:
                fail(f"application.saveGithubProvider -> {code} {short(body)}")
            print(f"source : github {owner}/{repo}@{branch} via « {name} »")
            return
    fail(f"aucun fournisseur GitHub de Dokploy ne voit {owner}/{repo} : installe l'application GitHub de Dokploy sur ce compte "
         f"(Settings → Git → GitHub) en lui donnant accès au dépôt, puis relance")


def build_and_env(api: Dokploy, app_id: str, access_code: Optional[str]) -> None:
    payload = {"applicationId": app_id, "buildType": "dockerfile", "dockerfile": "Dockerfile", "dockerContextPath": ".",
               "dockerBuildStage": "", "publishDirectory": None, "herokuVersion": None, "isStaticSpa": None, "railpackVersion": None}
    code, body = api.call("application.saveBuildType", payload)
    if code != 200:
        minimal = {k: payload[k] for k in ("applicationId", "buildType", "dockerfile", "dockerContextPath", "dockerBuildStage")}
        code, body = api.call("application.saveBuildType", minimal)
    if code != 200:
        fail(f"application.saveBuildType -> {code} {short(body)}")
    env = f"FORTYK_ACCESS_CODE={access_code}" if access_code else ""
    tries = [
        {"applicationId": app_id, "env": env, "buildArgs": "", "buildSecrets": "", "createEnvFile": False},
        {"applicationId": app_id, "env": env, "buildArgs": "", "buildSecrets": ""},
        {"applicationId": app_id, "env": env, "buildArgs": ""},
        {"applicationId": app_id, "env": env},
    ]
    errors = []
    for payload in tries:
        code, body = api.call("application.saveEnvironment", payload)
        if code == 200:
            break
        text = short(body, 600)
        if access_code:
            text = text.replace(access_code, "***")
        errors.append(text)
    else:
        fail(f"application.saveEnvironment -> {code} ; réponses : {' | '.join(errors)}")
    print("build : Dockerfile ; variable FORTYK_ACCESS_CODE " + ("définie" if access_code else "vide (création de parties ouverte à tous)"))


def app_details(api: Dokploy, app_id: str) -> Dict[str, Any]:
    code, body = api.call("application.one", query={"applicationId": app_id}, quiet=True)
    if code != 200:
        fail(f"application.one -> {code}")
    return unwrap(body)


def ensure_volume(api: Dokploy, app: Dict[str, Any]) -> None:
    if any(m.get("mountPath") == "/data" for m in app.get("mounts") or []):
        print("volume /data déjà monté")
        return
    code, body = api.call("mounts.create", {"type": "volume", "volumeName": "fortyk-data", "mountPath": "/data",
                                            "serviceType": "application", "serviceId": app["applicationId"]})
    if code != 200:
        fail(f"mounts.create -> {code} {short(body)}")
    print("volume nommé fortyk-data monté sur /data")


def ensure_domain(api: Dokploy, app: Dict[str, Any], host: Optional[str]) -> str:
    domains = app.get("domains") or []
    for d in domains:
        if not host or d.get("host") == host:
            scheme = "https" if d.get("https") else "http"
            print(f"domaine existant : {scheme}://{d['host']}")
            return f"{scheme}://{d['host']}"
    https = bool(host)
    if not host:
        payload = {"appName": app.get("appName")}
        if app.get("serverId"):
            payload["serverId"] = app["serverId"]
        code, body = api.call("domain.generateDomain", payload)
        body = unwrap(body)
        if code != 200 or not isinstance(body, str):
            fail(f"domain.generateDomain -> {code} {short(body)}")
        host = body
    payload = {"host": host, "path": "/", "port": PORT, "https": https, "certificateType": "letsencrypt" if https else "none",
               "applicationId": app["applicationId"], "domainType": "application"}
    code, body = api.call("domain.create", payload)
    if code != 200:
        payload.pop("domainType")
        code, body = api.call("domain.create", payload)
    if code != 200:
        fail(f"domain.create -> {code} {short(body)}")
    url = f"{'https' if https else 'http'}://{host}"
    print(f"domaine : {url}")
    return url


def deploy_and_wait(api: Dokploy, app_id: str, url: str, minutes: int = 20) -> None:
    def latest():
        code, body = api.call("deployment.all", query={"applicationId": app_id}, quiet=True)
        deps = unwrap(body) or []
        if not deps:
            return None, None
        d = sorted(deps, key=lambda x: x.get("createdAt") or "")[-1]
        return d.get("deploymentId"), d.get("status")

    before, _ = latest()
    code, body = api.call("application.deploy", {"applicationId": app_id})
    if code != 200:
        fail(f"application.deploy -> {code} {short(body)}")
    for i in range(minutes * 4):
        time.sleep(15)
        dep, status = latest()
        print(f"  [{i + 1}] déploiement {dep} : {status}")
        if dep and dep != before and status == "done":
            break
        if dep and dep != before and status == "error":
            fail("le déploiement Dokploy a échoué (voir ses logs dans l'onglet Deployments)")
    else:
        fail("déploiement pas terminé à temps")
    for _ in range(20):
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=10) as r:
                notice(f"santé : {url}/healthz -> {r.status} {r.read()[:120].decode(errors='replace')}")
                with urllib.request.urlopen(url + "/api/config", timeout=10) as r2:
                    notice(f"config : {r2.read()[:120].decode(errors='replace')}")
                check_access_code(url)
                return
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(10)
    print(f"::warning::{url}/healthz ne répond pas encore ({last}) — le certificat ou le DNS peut prendre quelques minutes")


def check_access_code(url: str) -> None:
    """Le code d'accès du service est-il bien celui du secret ? Sonde sans effet de bord : on demande
    d'enregistrer une liste vide ; un bon code donne « liste vide », un mauvais « code d'accès incorrect »."""
    code = os.environ.get("FORTYK_ACCESS_CODE")
    if not code:
        return
    req = urllib.request.Request(url + "/api/lists/save", method="POST", data=json.dumps({"text": "", "access_code": code}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        err = json.loads(r.read()).get("error", "")
    if "code d'accès" in err:
        fail("le service refuse le code d'accès du secret FORTYK_ACCESS_CODE : la variable de l'application Dokploy ne correspond pas")
    notice(f"code d'accès : accepté par le service ({len(code)} caractères)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="other")
    p.add_argument("--app", default="40kplayer")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "LeCastorFou/40kPlayer"), help="owner/repo")
    p.add_argument("--branch", default="main")
    p.add_argument("--domain", default="", help="nom d'hôte (DNS pointant vers le serveur) ; vide = domaine traefik.me généré")
    p.add_argument("--no-deploy", action="store_true")
    args = p.parse_args(argv)
    url, key = os.environ.get("DOKPLOY_URL"), os.environ.get("DOKPLOY_API_KEY")
    if not url or not key:
        fail("DOKPLOY_URL et DOKPLOY_API_KEY sont requis")
    api = Dokploy(url, key)
    print(f"instance Dokploy : {urllib.parse.urlparse(api.base).netloc}")
    projects = all_projects(api)
    server_id = find_server_id(projects)
    print(f"{len(projects)} projet(s) ; serveur des applications existantes : {server_id or 'hôte Dokploy'}")
    project = ensure_project(api, args.project)
    app_id = ensure_application(api, project, args.app, server_id)
    owner, repo = args.repo.split("/", 1)
    github_source(api, app_id, owner, repo, args.branch)
    build_and_env(api, app_id, os.environ.get("FORTYK_ACCESS_CODE") or None)
    app = app_details(api, app_id)
    ensure_volume(api, app)
    app = app_details(api, app_id)
    site = ensure_domain(api, app, args.domain or None)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not args.no_deploy:
        deploy_and_wait(api, app_id, site)
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### 40kPlayer sur Dokploy\n\n- projet : `{args.project}`\n- application : `{args.app}` (`{app_id}`)\n- adresse : {site}\n")
    notice(f"40kPlayer : projet {args.project}, application {args.app} ({app_id}), adresse {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
