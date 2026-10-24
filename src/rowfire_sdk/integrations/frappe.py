"""Frappe (and ERPNext): tag each document write with the request or job that
caused it, and release the resulting events only when the transaction commits.

Frappe has no SQLAlchemy and no unit of work to listen to: its writes go
through `Document`, and the hooks a Frappe app declares in `hooks.py` are where
an app sees them. A Frappe app (`rowfire_frappe`) wires these functions:

    before_request = ["rowfire_sdk.integrations.frappe.before_request"]
    after_request  = ["rowfire_sdk.integrations.frappe.after_request"]
    before_job     = ["rowfire_sdk.integrations.frappe.before_job"]
    after_job      = ["rowfire_sdk.integrations.frappe.after_job"]
    doc_events = {"*": {
        "before_insert": ..., "before_validate": ..., "before_cancel": ...,
        "before_update_after_submit": ..., "before_change": ...,   # before_write
        "on_change":    "rowfire_sdk.integrations.frappe.on_change",
        "after_delete": "rowfire_sdk.integrations.frappe.after_delete",
    }}

and each site turns it on in its `site_config.json`:

    "rowfire_endpoint": "https://rowfire.internal",
    "rowfire_api_key": "rf_...",
    "rowfire": {"spool_dir": "...", "compress": false, ...}   # optional

How Frappe's pieces map onto the SDK's:

- A write is a document's `on_change`, which Frappe runs after every insert,
  save, submit, cancel, update-after-submit and `db_set`, and `after_delete`.
  The table is the doctype; the old values are `get_doc_before_save()`,
  read at the document's first `before_*` hook in the transaction.
  Submit and cancel are updates of `docstatus` (0 -> 1 -> 2), as in SQL.
- One document can change several times in one transaction (a submit whose
  `on_submit` calls `db_set("status", ...)` runs `on_change` twice, both
  against the same snapshot from before the save). So writes are coalesced per
  document: the first "before", the last "after", matched once in
  `frappe.db.before_commit`, sent in `after_commit`, dropped in
  `after_rollback`. One business change, one event.
- A request is named by its route template (`PUT /api/resource/Sales
  Invoice/{name}`), except the desk's generic RPCs (`savedocs`,
  `frappe.client.*`, ...), through which nearly every save, submit and cancel
  in the UI goes: those are named by what they do, `desk Submit Sales
  Invoice`. A job is `job <method>`, a scheduled job `job <job type>`.
- A bench serves many sites from one process, each with its own key, so there
  is one client per site, not the module-level one `rowfire_sdk.init` makes.
- RQ runs each job in a forked process that leaves with `os._exit`, so
  `atexit` never runs there: `after_job` flushes. A job's process is new each
  time, so its subscriptions come from the copy cached in `spool_dir`, or are
  fetched before the job runs when there is none.

Not seen, in this prototype: `frappe.db.set_value`, `frappe.db.sql` and query
builder updates, which bypass `Document` (in ERPNext: stock levels in `Bin`,
an invoice's outstanding amount, valuation updates); and child-table rows (an
invoice's items, a customer's credit limits), which Frappe saves with their
parent without running their hooks. A rollback to a savepoint runs no
callback either: a document that matched a subscription is read back before
commit, so its event is dropped if the savepoint undid it, but the catalog
still counts the write.
"""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
import re
import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import frappe

from rowfire_sdk._client import Change, Client
from rowfire_sdk._context import Operation, current
from rowfire_sdk._matching import Ceiling

log = logging.getLogger("rowfire_sdk")

_clients: dict[str, Client | None] = {}
_TX = "rowfire_tx"
_OP = "rowfire_op"

# Writes that only move these are a save with nothing changed.
_BOOKKEEPING = frozenset({"modified", "modified_by"})

# --- clients, one per site --------------------------------------------------


def get_client() -> Client | None:
    """This site's client, created on first use; None if Rowfire is off here."""
    site = getattr(frappe.local, "site", None)
    if not site:
        return None
    if site in _clients:
        return _clients[site]
    conf = frappe.local.conf or {}
    endpoint, api_key = conf.get("rowfire_endpoint"), conf.get("rowfire_api_key")
    if not endpoint or not api_key:
        return None
    try:
        options = dict(conf.get("rowfire") or {})
        ceiling = options.pop("ceiling", None)
        if ceiling is not None:
            options["ceiling"] = Ceiling(**ceiling)
        options.setdefault("spool_dir", frappe.get_site_path("private", "rowfire"))
        try:
            os.makedirs(options["spool_dir"], exist_ok=True)
        except OSError:
            pass  # the client says why, and keeps retries in memory
        client = Client(endpoint, api_key, **options)
    except Exception:
        # Once per process, not on every request.
        log.exception("rowfire-sdk: could not start for site %s; off until restart", site)
        _clients[site] = None
        return None
    if not _clients:
        # `rowfire_sdk.init`'s atexit closes only its own client.
        atexit.register(_close_all)
    _clients[site] = client
    try:
        _declare(client)
    except Exception:
        log.exception("rowfire-sdk: could not list the site's operations")
    return client


def _close_all() -> None:
    for client in list(_clients.values()):
        if client is not None:
            client.close()
    _clients.clear()


def _declare(client: Client) -> None:
    """Operations that exist before anyone runs them: scheduled jobs, and
    submitting or cancelling each submittable doctype."""
    if client.catalog is None:
        return
    names = [
        f"job {t}" for t in frappe.get_all("Scheduled Job Type", pluck="method", filters={"stopped": 0})
    ]
    for doctype in frappe.get_all("DocType", pluck="name", filters={"is_submittable": 1, "istable": 0}):
        names += [f"desk Submit {doctype}", f"desk Cancel {doctype}"]
    client.declare_operations(sorted(names))


# --- requests --------------------------------------------------------------


def before_request() -> None:
    client = get_client()
    if client is None:
        return
    op = Operation(resolve=_request_name, resolve_params=_request_params)
    frappe.local.rowfire_op = (client, op, current.set(op))


def after_request(response: Any = None, request: Any = None) -> None:
    state = getattr(frappe.local, _OP, None)
    if state is None:
        return
    frappe.local.rowfire_op = None
    client, op, token = state
    try:
        op.status = getattr(response, "status_code", None) or 500
        if op.actor is None:
            op.actor = _actor()
        client.operation_finished(op)
        client.observe_operation(op, op.status >= 500)
    except Exception:
        log.exception("rowfire-sdk: could not finish the request")
    finally:
        current.reset(token)


# The desk's generic RPCs, and what each one does to the document it is given.
_DESK = {
    "frappe.desk.form.save.savedocs": None,  # the verb is the `action` argument
    "frappe.desk.form.save.cancel": "Cancel",
    "frappe.client.save": "Save",
    "frappe.client.insert": "Insert",
    "frappe.client.submit": "Submit",
    "frappe.client.cancel": "Cancel",
    "frappe.client.delete": "Delete",
    "frappe.client.set_value": "Set Value",
    "frappe.desk.reportview.delete_items": "Delete",
    "frappe.model.workflow.apply_workflow": None,  # `action` again
    "frappe.handler.run_doc_method": None,  # `<doctype>.<method>`
    "run_doc_method": None,
}

_CONVERTER = re.compile(r"<(?:[a-z]+:)?(\w+)>")


def _request_name() -> str | None:
    request = frappe.local.request
    method = request.method
    path = request.path
    if path.startswith("/api/"):
        from frappe.api import API_URL_MAP

        try:
            rule, args = API_URL_MAP.bind_to_environ(request.environ).match(return_rule=True)
        except Exception:
            return f"{method} /api/(unknown)"
        rpc = args.get("method")
        if rpc in _DESK:
            name = _desk_name(rpc)
            if name:
                return name
        # The doctype and the RPC are part of what the operation is; the
        # document's name is a parameter.
        def fill(m: re.Match) -> str:
            key = m.group(1)
            return "{name}" if key == "name" else str(args.get(key, "{" + key + "}"))

        return f"{method} {_CONVERTER.sub(fill, rule.rule).rstrip('/')}"
    cmd = frappe.form_dict.get("cmd")
    if cmd:
        return _desk_name(cmd) if cmd in _DESK else f"{method} /api/method/{cmd}"
    # Website pages: one name for all, or every product page is an operation.
    return f"{method} (website)"


def _desk_target() -> tuple[str | None, str | None]:
    form = frappe.form_dict
    doc = form.get("doc") or form.get("docs")
    if doc:
        try:
            doc = json.loads(doc) if isinstance(doc, str) else doc
            return doc.get("doctype"), doc.get("name")
        except Exception:
            return None, None
    return form.get("doctype") or form.get("dt"), form.get("name") or form.get("dn")


def _desk_name(rpc: str) -> str | None:
    doctype, _ = _desk_target()
    if not doctype:
        return None
    verb = _DESK.get(rpc)
    form = frappe.form_dict
    if rpc.endswith("run_doc_method"):
        return f"desk {doctype}.{form.get('method')}"
    if rpc.endswith("apply_workflow"):
        return f"desk Workflow {form.get('action')} {doctype}"
    if verb is None:
        verb = form.get("action") or "Save"
    return f"desk {verb} {doctype}"


def _request_params() -> dict[str, Any]:
    doctype, name = _desk_target()
    if doctype:
        return {"doctype": doctype, "name": name}
    request = frappe.local.request
    if request.path.startswith("/api/"):
        from frappe.api import API_URL_MAP

        try:
            _, args = API_URL_MAP.bind_to_environ(request.environ).match()
        except Exception:
            return {}
        return {k: v for k, v in args.items() if k in ("doctype", "name")}
    return {}


def _actor() -> dict[str, Any]:
    user = getattr(getattr(frappe.local, "session", None), "user", None) or "Guest"
    actor: dict[str, Any] = {"id": user}
    if user in ("Guest", "Administrator"):
        actor["type"] = user
        return actor
    try:
        user_type, profile = frappe.get_cached_value("User", user, ["user_type", "role_profile_name"])
        actor["type"] = user_type
        if profile:
            actor["role_profile"] = profile
    except Exception:
        pass
    return actor


# --- jobs ---------------------------------------------------------------------


def before_job(method: str = "", kwargs: Any = None, transaction_type: str = "job") -> None:
    client = get_client()
    if client is None:
        return
    if not client._static and client.matcher.version is None:
        # A new process per job: nothing cached yet, so ask before the job
        # writes anything, or its writes are matched against no subscription.
        client._poll_config()
    name = method
    if method.endswith("run_scheduled_job") and isinstance(kwargs, dict):
        name = kwargs.get("job_type") or method
    op = Operation(f"job {name}", actor=_actor())
    frappe.local.rowfire_op = (client, op, current.set(op))


def after_job(method: str = "", kwargs: Any = None, result: Any = None) -> None:
    state = getattr(frappe.local, _OP, None)
    if state is None:
        return
    frappe.local.rowfire_op = None
    client, op, token = state
    try:
        failed = sys.exc_info()[0] is not None
        if not failed:
            client.operation_finished(op)
        client.observe_operation(op, failed)
        # The job's process may be a fork that leaves with os._exit: send
        # now, or what is queued dies with it (the disk spool keeps the rest).
        client.flush(timeout=float((frappe.local.conf.get("rowfire") or {}).get("job_flush_timeout", 2.0)))
    except Exception:
        log.exception("rowfire-sdk: could not finish the job")
    finally:
        current.reset(token)


# --- documents -----------------------------------------------------------------


def before_write(doc: Any, method: str | None = None) -> None:
    """`before_insert`, `before_validate`, `before_cancel`,
    `before_update_after_submit`, `before_change` (db_set): remember what the
    document was before this transaction touched it.

    It has to be here, not at `on_change`: by the time a submitted invoice's
    own `on_change` runs, ERPNext has loaded a second copy of it (to update
    its outstanding amount) and `db_set` its status, and that copy's "before"
    was read after the submit's UPDATE.
    """
    _touch(doc, delete=False, begin=True)


def on_change(doc: Any, method: str | None = None) -> None:
    if doc.flags.in_delete:
        return  # `after_delete` reports it, once the row is gone
    _touch(doc, delete=False)


def after_delete(doc: Any, method: str | None = None) -> None:
    _touch(doc, delete=True)


def _touch(doc: Any, delete: bool, begin: bool = False) -> None:
    client = get_client()
    if client is None:
        return
    try:
        meta = doc.meta
        if meta.istable or frappe.flags.in_install:
            return
        op = current.get()
        if op is not None and op.actor is None:
            op.actor = _actor()
        columns, numeric = _columns(meta)
        tx = _transaction(client)
        if begin and not doc.name:
            return  # before_insert: no name yet, and nothing "before" to keep
        key = (doc.doctype, doc.name)
        entry = tx["docs"].get(key)
        now = None if begin else _snapshot(doc, columns, numeric)
        if entry is None:
            before = None if delete else doc.get_doc_before_save()
            if delete:
                kind, old = "delete", now
            elif before is None or doc.flags.in_insert or doc.is_new():
                # `in_insert` too: a `db_set` in the insert's own `on_submit`
                # reloads a "before" from the row the insert just wrote.
                kind, old = "insert", None
            else:
                kind, old = "update", _snapshot(before, columns, numeric)
            tx["docs"][key] = [kind, old, now, op, meta]
        elif not begin:
            entry[2] = now
            if delete:
                entry[0] = "delete"
    except Exception:
        client.counters.lose("internal_error")
        log.exception("rowfire-sdk: could not read a document change")


def _transaction(client: Client) -> dict[str, Any]:
    tx = getattr(frappe.local, _TX, None)
    if tx is None:
        tx = {"docs": {}, "events": [], "records": []}
        frappe.local.rowfire_tx = tx
        db = frappe.db
        db.before_commit.add(lambda: _before_commit(client, tx))
        db.after_commit.add(lambda: _after_commit(client, tx))
        db.after_rollback.add(lambda: _discard(tx))
    return tx


def _before_commit(client: Client, tx: dict[str, Any]) -> None:
    # Writes made from here on (by other before_commit callbacks) start a new
    # transaction record with its own callbacks, which run in this same commit.
    if getattr(frappe.local, _TX, None) is tx:
        frappe.local.rowfire_tx = None
    try:
        for (doctype, name), (kind, old, row, op, meta) in tx["docs"].items():
            if row is None:
                continue  # began, but never written (an error the app caught)
            change = _change(doctype, kind, old, row, meta)
            if change is None:
                continue
            record = client.read_write(op, change)
            if record is not None:
                tx["records"].append(record)
            subs = client.subscriptions_for(op, doctype)
            if subs:
                events = client.match_change(op, subs, change)
                if events and not _still_true(doctype, name, kind, events, subs, meta):
                    continue
                tx["events"].extend(events)
    except Exception:
        client.counters.lose("internal_error")
        log.exception("rowfire-sdk: could not match a transaction's changes")


def _still_true(doctype: str, name: str, kind: str, events: list, subs: list, meta: Any) -> bool:
    """Whether the row still exists and still meets what the matched
    subscriptions asked of it, now that the transaction is about to commit.

    Frappe runs no callback when it rolls back to a savepoint, and ERPNext
    does that inside transactions that then commit (bulk_transaction.py: one
    savepoint per document, rolled back when it fails). The database already
    holds the transaction's final state here, so the documents that matched a
    subscription, and only those, are read back: one query each.
    """
    if kind == "delete":
        return not frappe.db.exists(doctype, name)
    matched = {e["subscription"] for e in events}
    wanted = [(path[4:], expected) for sub in subs if sub.id in matched for path, expected in sub.where if path.startswith("row.")]
    keys = sorted({k for k, _ in wanted} | {"docstatus"})
    stored = frappe.db.get_value(doctype, name, keys, as_dict=True)
    if not stored:
        return False
    _, numeric = _columns(meta)
    now = _snapshot(SimpleNamespace(**stored, doctype=doctype), tuple(keys), numeric)
    return all(now[k] in expected if isinstance(expected, list) else now[k] == expected for k, expected in wanted)


def _after_commit(client: Client, tx: dict[str, Any]) -> None:
    client.emit(tx["events"])
    client.record_writes(tx["records"])
    _discard(tx)


def _discard(tx: dict[str, Any]) -> None:
    tx["docs"].clear()
    tx["events"] = []
    tx["records"] = []
    if getattr(frappe.local, _TX, None) is tx:
        frappe.local.rowfire_tx = None


def _change(doctype: str, kind: str, old: dict | None, row: dict, meta: Any) -> Change | None:
    if kind == "update":
        changed = {k for k, v in row.items() if old.get(k) != v}
        if changed <= _BOOKKEEPING:
            return None  # saved with nothing changed
    elif kind == "insert":
        changed = {k for k, v in row.items() if v not in (None, "")}
    else:
        changed = set()
    return Change(
        doctype,
        kind,
        row.get,
        (lambda key: None) if old is None else old.get,
        lambda: changed,
        columns=row.keys(),
        describe=lambda: describe(meta),
    )


_columns_by_doctype: dict[tuple[str, str], tuple[tuple[str, ...], frozenset]] = {}
_times_by_doctype: dict[tuple[str, str], frozenset] = {}
_NUMERIC = frozenset({"Int", "Float", "Currency", "Percent", "Check", "Rating", "Duration"})


def _columns(meta: Any) -> tuple[tuple[str, ...], frozenset]:
    key = (frappe.local.site, meta.name)
    cols = _columns_by_doctype.get(key)
    if cols is None:
        numeric = {df.fieldname for df in meta.get("fields") if df.fieldtype in _NUMERIC}
        times = {df.fieldname for df in meta.get("fields") if df.fieldtype == "Time"}
        _times_by_doctype[key] = frozenset(times)
        cols = (tuple(meta.get_valid_columns()), frozenset(numeric | {"docstatus", "idx"}))
        _columns_by_doctype[key] = cols
    return cols


def _snapshot(doc: Any, columns: tuple[str, ...], numeric: frozenset) -> dict[str, Any]:
    """The document's values in one form, whether it was built in memory or
    read back from the database: a date is a string in one and a `date` in
    the other, an unset number is None in one and 0 in the other, and an
    unset text is None or "". Compared raw, every save would "change" them."""
    values = doc.__dict__
    times = _times_by_doctype.get((frappe.local.site, doc.doctype), ())
    out = {}
    for k in columns:
        v = values.get(k)
        if v is None or v == "":
            v = 0 if k in numeric else None
        elif isinstance(v, (datetime.datetime, datetime.date, datetime.time, datetime.timedelta)):
            v = _date_text(v)
        elif isinstance(v, Decimal):
            v = float(v)
        elif k in times and isinstance(v, str):
            v = _time_text(v)  # ERPNext writes "14:39:7.130775" as often as "14:39:07.130775"
        out[k] = v
    return out


def _time_text(v: str) -> str:
    hms, _, frac = v.partition(".")
    parts = hms.split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return v
    text = ":".join(f"{int(p):02d}" for p in parts)
    frac = frac.rstrip("0")
    return f"{text}.{frac.ljust(6, '0')}" if frac else text


def _date_text(v: Any) -> str:
    if isinstance(v, datetime.timedelta):  # a Time column, as MariaDB hands it back
        secs = int(v.total_seconds())
        text = f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"
        return _time_text(f"{text}.{v.microseconds:06d}")
    if isinstance(v, datetime.datetime):
        return v.isoformat(sep=" ")
    return v.isoformat()


_STANDARD = {
    "name": {"type": "data", "nullable": False, "primary_key": True},
    "owner": {"type": "link", "nullable": True, "primary_key": False, "references": ["User.name"]},
    "modified_by": {"type": "link", "nullable": True, "primary_key": False, "references": ["User.name"]},
    "creation": {"type": "datetime", "nullable": True, "primary_key": False},
    "modified": {"type": "datetime", "nullable": True, "primary_key": False},
    "docstatus": {
        "type": "int",
        "nullable": False,
        "primary_key": False,
        # 0 draft, 1 submitted, 2 cancelled
        "declared_values": [0, 1, 2],
    },
    "idx": {"type": "int", "nullable": True, "primary_key": False},
}


def describe(meta: Any) -> dict[str, dict[str, Any]]:
    """Each field's declared type, from the DocType: no data involved."""
    columns = {k: dict(v) for k, v in _STANDARD.items()}
    for df in meta.get("fields"):
        if df.fieldname not in meta.get_valid_columns():
            continue
        info: dict[str, Any] = {
            "type": df.fieldtype.lower().replace(" ", "_"),
            "nullable": not df.reqd,
            "primary_key": False,
        }
        if df.label:
            info["label"] = df.label
        if df.fieldtype == "Select" and df.options:
            info["declared_values"] = [o for o in df.options.split("\n") if o]
        elif df.fieldtype == "Link" and df.options:
            info["references"] = [f"{df.options}.name"]
        elif df.fieldtype == "Check":
            info["declared_values"] = [0, 1]
        columns[df.fieldname] = info
    return columns
