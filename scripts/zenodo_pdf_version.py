#!/usr/bin/env python3
"""
zenodo_pdf_version.py — Crea una nueva versión de un registro Zenodo con el PDF
del artículo como fichero de primer nivel y como vista previa por defecto
(files.default_preview), usando la API InvenioRDM (/api/records).

La API legacy (/api/deposit/depositions) no expone default_preview; por eso se
usa la API de registros.

Modos:
  A) Reutilizar los ficheros de la versión anterior (zip de GitHub) + añadir PDF:
       python zenodo_pdf_version.py --record 21934956 \
           --pdf paper/superstructures.pdf --version 1.0.2
  B) Subir ficheros nuevos (p. ej. zip del tag generado con git archive) + PDF
     (modo usado por el workflow de GitHub Actions):
       python zenodo_pdf_version.py --record 21934956 --files src.zip \
           --pdf paper/superstructures.pdf --version 1.0.2 \
           --github-tree-url https://github.com/USER/REPO/tree/v1.0.2

Token: variable de entorno ZENODO_TOKEN (scopes deposit:write, deposit:actions).
Requiere: requests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import requests

PROD = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"
# Campos que acepta PUT /records/{id}/draft; el resto (links, stats, revision_id…)
# son de solo lectura.
WRITABLE_KEYS = ("metadata", "access", "files", "pids", "custom_fields")


class ZenodoError(RuntimeError):
    pass


class Zenodo:
    def __init__(self, token: str, base: str = PROD, timeout: int = 120,
                 session: requests.Session | None = None):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/json"})

    # -- transporte ---------------------------------------------------------
    def _req(self, method: str, path: str, ok=(200, 201, 202, 204), **kw):
        url = path if path.startswith("http") else f"{self.base}{path}"
        r = self.s.request(method, url, timeout=self.timeout, **kw)
        if r.status_code not in ok:
            raise ZenodoError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:800]}")
        if r.status_code == 204 or not r.content:
            return {}
        return r.json()

    # -- operaciones --------------------------------------------------------
    def latest_id(self, record_id: str) -> str:
        """Resuelve cualquier id de versión (o concept id) a la última publicada."""
        data = self._req("GET", f"/records/{record_id}/versions/latest")
        return str(data["id"])

    def new_version(self, record_id: str) -> dict:
        # Si ya existe un borrador de nueva versión, InvenioRDM lo devuelve.
        return self._req("POST", f"/records/{record_id}/versions")

    def draft(self, rid: str) -> dict:
        return self._req("GET", f"/records/{rid}/draft")

    def draft_files(self, rid: str) -> list[dict]:
        return self._req("GET", f"/records/{rid}/draft/files").get("entries", [])

    def import_previous_files(self, rid: str) -> None:
        self._req("POST", f"/records/{rid}/draft/actions/files-import")

    def delete_file(self, rid: str, key: str) -> None:
        self._req("DELETE", f"/records/{rid}/draft/files/{key}")

    def upload(self, rid: str, path: Path, key: str) -> None:
        self._req("POST", f"/records/{rid}/draft/files", json=[{"key": key}])
        with path.open("rb") as fh:
            self._req("PUT", f"/records/{rid}/draft/files/{key}/content", data=fh,
                      headers={"Content-Type": "application/octet-stream"})
        self._req("POST", f"/records/{rid}/draft/files/{key}/commit")

    def reserve_doi(self, rid: str) -> None:
        self._req("POST", f"/records/{rid}/draft/pids/doi")

    def update_draft(self, rid: str, payload: dict) -> dict:
        return self._req("PUT", f"/records/{rid}/draft", json=payload,
                         headers={"Content-Type": "application/json"})

    def publish(self, rid: str) -> dict:
        return self._req("POST", f"/records/{rid}/draft/actions/publish")


def build_payload(draft: dict, *, version: str, date: str, pdf_key: str,
                  add_rights: list[str], github_tree_url: str | None) -> dict:
    payload = {k: draft[k] for k in WRITABLE_KEYS if k in draft}
    md = payload.setdefault("metadata", {})
    # new_version borra version y publication_date: hay que fijarlos.
    md["version"] = version
    md["publication_date"] = date
    payload["files"] = {"enabled": True, "default_preview": pdf_key}

    if add_rights:
        rights = md.setdefault("rights", [])
        have = {r.get("id") for r in rights}
        rights.extend({"id": rid} for rid in add_rights if rid not in have)

    if github_tree_url:
        rel = [r for r in md.get("related_identifiers", [])
               if not ("github.com" in r.get("identifier", "")
                       and "/tree/" in r.get("identifier", ""))]
        rel.append({"identifier": github_tree_url, "scheme": "url",
                    "relation_type": {"id": "issupplementto"},
                    "resource_type": {"id": "software"}})
        md["related_identifiers"] = rel
    return payload


def run(args: argparse.Namespace, z: Zenodo) -> dict:
    pdf = Path(args.pdf)
    if not pdf.is_file():
        raise SystemExit(f"No existe el PDF: {pdf}")
    extra = [Path(p) for p in args.files]
    for p in extra:
        if not p.is_file():
            raise SystemExit(f"No existe el fichero: {p}")
    pdf_key = args.pdf_name or f"{args.basename}-v{args.version}.pdf"

    latest = z.latest_id(args.record)
    print(f"· Última versión publicada: {latest}")
    rid = str(z.new_version(latest)["id"])
    print(f"· Borrador de nueva versión: {rid}")

    existing = {e["key"] for e in z.draft_files(rid)}
    if extra:
        for key in existing:                    # borrador reanudado: limpiar
            z.delete_file(rid, key)
        for p in extra:
            print(f"· Subiendo {p.name}")
            z.upload(rid, p, p.name)
    elif not (existing - {pdf_key}):            # no hay ficheros heredados aún
        if pdf_key in existing:                 # files-import exige borrador vacío
            z.delete_file(rid, pdf_key)
        print("· Importando ficheros de la versión anterior")
        z.import_previous_files(rid)
    elif pdf_key in existing:                   # reanudado con PDF previo
        z.delete_file(rid, pdf_key)

    print(f"· Subiendo PDF como '{pdf_key}'")
    z.upload(rid, pdf, pdf_key)

    d = z.draft(rid)
    if not (d.get("pids") or {}).get("doi"):
        try:
            z.reserve_doi(rid)
            d = z.draft(rid)
        except ZenodoError as e:  # no fatal: Zenodo acuña el DOI al publicar
            print(f"  (aviso) no se pudo reservar DOI: {e}", file=sys.stderr)

    # Un borrador de nueva version puede llegar sin los campos obligatorios
    # (resource_type, creators, title...): se rellenan desde la ultima version publicada.
    pub_md = z._req("GET", f"/records/{latest}").get("metadata", {})
    md = d.setdefault("metadata", {})
    for key in ("resource_type", "creators", "title", "publisher", "description", "rights"):
        if not md.get(key) and pub_md.get(key):
            md[key] = pub_md[key]
            print(f"  (relleno) metadata.{key} copiado de la version {latest}")

    payload = build_payload(d, version=args.version, date=args.date,
                            pdf_key=pdf_key, add_rights=args.add_license,
                            github_tree_url=args.github_tree_url)
    res = z.update_draft(rid, payload)
    if res.get("errors"):
        raise ZenodoError(f"Validación del borrador: {res['errors']}")
    doi = (res.get("pids") or {}).get("doi", {}).get("identifier", "(al publicar)")
    print(f"· Metadatos OK — versión {args.version}, default_preview={pdf_key}, DOI {doi}")

    if args.no_publish:
        print(f"· Borrador listo para revisar: {z.base.removesuffix('/api')}/uploads/{rid}")
        return res
    pub = z.publish(rid)
    print(f"✔ Publicado: {pub.get('links', {}).get('self_html', rid)}  DOI {doi}")
    return pub


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", required=True,
                   help="id de cualquier versión publicada (se resuelve a la última)")
    p.add_argument("--pdf", required=True, help="ruta local del PDF del artículo")
    p.add_argument("--pdf-name", help="nombre del PDF en Zenodo "
                   "(por defecto <basename>-v<version>.pdf)")
    p.add_argument("--basename", default="fixed-learned-both")
    p.add_argument("--version", required=True, help="p. ej. 1.0.2")
    p.add_argument("--date", default=dt.date.today().isoformat(),
                   help="publication_date ISO (por defecto hoy)")
    p.add_argument("--files", nargs="*", default=[],
                   help="ficheros que sustituyen a los de la versión anterior")
    p.add_argument("--add-license", nargs="*", default=[],
                   help="ids de licencia a añadir, p. ej. cc-by-4.0")
    p.add_argument("--github-tree-url", help="URL tree/<tag> para related_identifiers")
    p.add_argument("--sandbox", action="store_true", help="usar sandbox.zenodo.org")
    p.add_argument("--no-publish", action="store_true",
                   help="dejar el borrador sin publicar para revisarlo en la web")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("Falta ZENODO_TOKEN en el entorno")
    z = Zenodo(token, SANDBOX if args.sandbox else PROD)
    try:
        run(args, z)
    except ZenodoError as e:
        print(f"✘ {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
