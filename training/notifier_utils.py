"""Email notifications for long-running training jobs (e.g. the financial_poc
run, which spans many hours/sessions unattended, on either Kaggle or Colab).

Credentials are never hardcoded and never read from a plain env var as the
primary path - they come from the host platform's own secrets vault (Kaggle
Secrets via kaggle_secrets.UserSecretsClient, or Colab Secrets via
google.colab.userdata), fetched at send-time so they never appear in the
notebook's saved source or output. An environment variable fallback exists
only so this module can be unit-tested / run locally off both platforms; it
is not the intended production path.

Required secrets (attach these under the notebook's secrets panel - Kaggle:
Add-ons -> Secrets; Colab: the key icon in the left sidebar - before relying
on this module):
  NOTIFIER_EMAIL_ADDRESS  - sending account, e.g. a Gmail address
  NOTIFIER_EMAIL_PASSWORD - an app password for that account, NOT its
                            normal login password (Gmail requires 2FA to
                            issue one: https://myaccount.google.com/apppasswords)
Optional:
  NOTIFIER_EMAIL_TO   - recipient address (defaults to NOTIFIER_EMAIL_ADDRESS)
  NOTIFIER_SMTP_HOST  - defaults to smtp.gmail.com
  NOTIFIER_SMTP_PORT  - defaults to 587 (STARTTLS)

Run test_connection() once after attaching secrets to confirm they're wired
correctly before depending on this during an actual long run.
"""

import datetime
import os
import smtplib
import traceback
from email.mime.text import MIMEText


def _get_secret(name: str) -> str:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        pass
    else:
        try:
            return UserSecretsClient().get_secret(name)
        except Exception as e:
            raise RuntimeError(
                f"Kaggle secret '{name}' could not be read ({e}). Attach it under "
                f"this notebook's Add-ons -> Secrets (and grant this notebook "
                f"access if it already exists in your account)."
            ) from e

    try:
        from google.colab import userdata
    except ImportError:
        pass
    else:
        try:
            return userdata.get(name)
        except Exception as e:
            raise RuntimeError(
                f"Colab secret '{name}' could not be read ({e}). Add it under "
                f"this notebook's left sidebar -> Secrets (key icon), and grant "
                f"this notebook access if it already exists."
            ) from e

    value = os.environ.get(name)
    if value:
        return value
    raise RuntimeError(
        f"Secret '{name}' not found: neither kaggle_secrets nor google.colab.userdata "
        f"is importable (not running on Kaggle or Colab) and no '{name}' environment "
        f"variable is set for local testing."
    )


def _get_secret_optional(name: str, default: str) -> str:
    try:
        return _get_secret(name)
    except RuntimeError:
        return default


def send_email(subject: str, body: str, to_addr: str = None) -> None:
    """Send one email via SMTP. Raises on failure - use this directly only
    for test_connection() / manual checks. Training-loop call sites should
    go through the notify_* wrappers below, which swallow failures so a
    flaky network/SMTP hiccup never takes down the training run."""
    from_addr = _get_secret("NOTIFIER_EMAIL_ADDRESS")
    password = _get_secret("NOTIFIER_EMAIL_PASSWORD")
    to_addr = to_addr or _get_secret_optional("NOTIFIER_EMAIL_TO", from_addr)
    host = _get_secret_optional("NOTIFIER_SMTP_HOST", "smtp.gmail.com")
    port = int(_get_secret_optional("NOTIFIER_SMTP_PORT", "587"))

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(from_addr, password)
        server.send_message(msg)


def test_connection() -> None:
    """Send a real test email and raise if anything goes wrong. Meant to be
    run once by hand after attaching the Kaggle Secrets, to catch a
    misconfigured secret or SMTP setting before trusting this during an
    actual multi-hour run."""
    send_email(
        subject="[Aivora AI] Notifier test",
        body=f"Notifier is configured correctly. Sent at "
             f"{datetime.datetime.utcnow().isoformat()}Z.",
    )


def _notify_safely(subject: str, body: str) -> None:
    try:
        send_email(subject, body)
    except Exception:
        print(f"[notifier_utils] Failed to send email '{subject}': "
              f"{traceback.format_exc()}")


def notify_checkpoint_saved(step: int, max_steps: int, train_loss: float, val_loss: float,
                             best_val_loss: float, tokens_processed: int, ckpt_path: str) -> None:
    subject = f"[Aivora AI] Checkpoint saved at step {step:,}/{max_steps:,}"
    body = (
        f"step: {step:,} / {max_steps:,}\n"
        f"train_loss: {train_loss:.4f}\n"
        f"val_loss: {val_loss:.4f}\n"
        f"best_val_loss: {best_val_loss:.4f}\n"
        f"tokens_processed: {tokens_processed:,}\n"
        f"checkpoint: {ckpt_path}\n"
    )
    _notify_safely(subject, body)


def notify_training_complete(final_step: int, train_loss: float, val_loss: float,
                              tokens_processed: int, ckpt_path: str, elapsed_seconds: float) -> None:
    subject = f"[Aivora AI] Training run complete at step {final_step:,}"
    body = (
        f"final_step: {final_step:,}\n"
        f"train_loss: {train_loss:.4f}\n"
        f"val_loss: {val_loss:.4f}\n"
        f"tokens_processed: {tokens_processed:,}\n"
        f"elapsed: {elapsed_seconds:.0f}s\n"
        f"final_checkpoint: {ckpt_path}\n"
    )
    _notify_safely(subject, body)


def notify_training_failed(step: int, error: BaseException, extra_context: str = None) -> None:
    subject = f"[Aivora AI] Training run FAILED at step {step:,}"
    body = (
        f"step: {step:,}\n"
        f"error: {error!r}\n"
        f"traceback:\n{''.join(traceback.format_exception(type(error), error, error.__traceback__))}\n"
    )
    if extra_context:
        body += f"\ncontext: {extra_context}\n"
    _notify_safely(subject, body)
